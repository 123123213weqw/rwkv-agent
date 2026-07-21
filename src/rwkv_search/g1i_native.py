from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Sequence

from .g1i_types import G1ICompletion


class FastRWKV7Completion:
    """Persistent native RWKV7 completion adapter with raw-token tracing."""

    def __init__(self, model_path: str, runtime_dir: str, context: int = 12288) -> None:
        runtime = str(Path(runtime_dir).resolve())
        if runtime not in sys.path:
            sys.path.insert(0, runtime)
        import torch
        from rwkv.utils import PIPELINE
        import rwkv7_fast_v3a as v3a

        v3a.MODEL_PATH, v3a.WKV_MODE, v3a.EMB_DEVICE = model_path, "fp32io16", "cpu"
        v3a.RKV_MODE, v3a.CMIX_SPARSE, v3a.LOWRANK_WEIGHT = "off", "no-fc", "transpose"
        v3a.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
        v3a.load_extensions(v3a.WKV_MODE)
        self.model, self.torch = v3a.RWKV7(), torch
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
        self.context = context

    def route_search(self, query: str, *, threshold: float = 0.7) -> Dict[str, object]:
        """Score one-token ``search``/``chat`` labels without free-form decoding."""

        started = time.perf_counter()
        prompt = (
            "System: Classify whether a request requires live public-web information. "
            "Use search for current or recent information, fact verification, explicit web search, "
            "or requested sources. Use chat for greetings, writing, translation, arithmetic, coding, "
            "or stable knowledge. Reply with one lowercase label.\n\n"
            "User: Hello.\n\nAssistant: chat\n\n"
            "User: Find today's major news and cite sources.\n\nAssistant: search\n\n"
            "User: Write a short poem.\n\nAssistant: chat\n\n"
            "User: What is the current stable Python release?\n\nAssistant: search\n\n"
            f"User: {query.strip()}\n\nAssistant: "
        )
        ids = self.pipeline.encode(prompt)[-self.context :]
        state, logits = self.model.zero_state(1), None
        while ids:
            part, ids = ids[:512], ids[512:]
            tokens = self.torch.tensor(
                part,
                dtype=self.torch.long,
                device="cpu" if self.model.emb_cpu else "cuda",
            )
            logits = self.model.forward(tokens, state).view(-1)
        assert logits is not None
        labels = {"search": self.pipeline.encode("search"), "chat": self.pipeline.encode("chat")}
        if any(len(token_ids) != 1 for token_ids in labels.values()):
            raise RuntimeError("G1I gate labels must each encode to exactly one token")
        scores = {name: float(logits[token_ids[0]].item()) for name, token_ids in labels.items()}
        margin = scores["search"] - scores["chat"]
        return {
            "use_search": margin >= threshold,
            "label": "search" if margin >= threshold else "chat",
            "scores": scores,
            "margin": margin,
            "threshold": threshold,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }

    def __call__(self, prompt: str, stops: Sequence[str], max_tokens: int) -> G1ICompletion:
        started = time.perf_counter()
        torch, ids = self.torch, self.pipeline.encode(prompt)[-self.context :]
        if not ids:
            return G1ICompletion("", elapsed_ms=(time.perf_counter() - started) * 1000)
        state, logits = self.model.zero_state(1), None
        while ids:
            part, ids = ids[:512], ids[512:]
            tokens = torch.tensor(
                part, dtype=torch.long, device="cpu" if self.model.emb_cpu else "cuda"
            )
            logits = self.model.forward(tokens, state).view(-1)
        output_ids, output = [], ""
        assert logits is not None
        for _ in range(max_tokens):
            token = int(torch.argmax(logits).item())
            if token == 0:
                return G1ICompletion(
                    output,
                    "</s>",
                    tuple(output_ids),
                    (time.perf_counter() - started) * 1000,
                )
            output_ids.append(token)
            decoded = self.pipeline.decode(output_ids)
            if "\ufffd" not in decoded:
                output = decoded
                hits = [(output.find(stop), stop) for stop in stops if stop and stop in output]
                if hits:
                    index, stop = min(hits)
                    return G1ICompletion(
                        output[:index],
                        stop,
                        tuple(output_ids),
                        (time.perf_counter() - started) * 1000,
                    )
            tokens = torch.tensor(
                [token], dtype=torch.long, device="cpu" if self.model.emb_cpu else "cuda"
            )
            logits = self.model.forward(tokens, state).view(-1)
        return G1ICompletion(
            output,
            "max_tokens",
            tuple(output_ids),
            (time.perf_counter() - started) * 1000,
        )
