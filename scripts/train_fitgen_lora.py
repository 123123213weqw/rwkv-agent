#!/usr/bin/env python3
"""Train a response-masked QLoRA adapter on FitGen Train with Dev loss only."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("RWKV7_FAST_FORWARD", "0")
os.environ.setdefault("RWKV7_NATIVE_MODEL_JIT", "0")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if "locked" in {part.casefold() for part in resolved.parts}:
        raise ValueError(f"refusing to train from locked data: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{resolved}:{line_number}: row must be object")
            if not isinstance(value.get("prompt"), str) or not isinstance(value.get("response"), str):
                raise ValueError(f"{resolved}:{line_number}: prompt/response must be strings")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty training data: {resolved}")
    return rows


def encode_example(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_length: int,
    max_response_tokens: int = 512,
) -> dict[str, Any]:
    """Left-truncate the prompt while preserving supervised response tokens."""

    prompt_ids = list(tokenizer(str(row["prompt"]), add_special_tokens=False)["input_ids"])
    response_ids = list(tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"])
    if not response_ids:
        raise ValueError(f"empty tokenized response: {row.get('id')}")
    response_ids = response_ids[: max(1, min(int(max_response_tokens), int(max_length) - 1))]
    prompt_budget = max(1, int(max_length) - len(response_ids))
    prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + list(response_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "task": str(row.get("task") or "unknown"),
        "id": str(row.get("id") or ""),
    }


@dataclass(frozen=True)
class TrainConfig:
    model: str
    init_adapter: str | None
    train_data: str
    dev_data: str
    output_dir: str
    seed: int
    max_length: int
    max_response_tokens: int
    bptt_window: int
    max_steps: int
    gradient_accumulation: int
    learning_rate: float
    warmup_steps: int
    lora_rank: int
    lora_alpha: int
    bfcl_repeat: int
    dataset_repeats: dict[str, int]
    include_datasets: list[str]
    include_tasks: list[str]
    eval_samples: int
    eval_every: int
    save_every: int


def _json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def adapter_input_manifest(path: Path) -> dict[str, Any]:
    """Validate and hash an immutable PEFT adapter used for continuation."""

    resolved = path.expanduser().resolve()
    if "locked" in {part.casefold() for part in resolved.parts}:
        raise ValueError(f"refusing to initialize from locked path: {resolved}")
    config_path = resolved / "adapter_config.json"
    weights_path = next(
        (
            candidate
            for candidate in (
                resolved / "adapter_model.safetensors",
                resolved / "adapter_model.bin",
            )
            if candidate.is_file()
        ),
        None,
    )
    if not config_path.is_file() or weights_path is None:
        raise ValueError(f"invalid PEFT adapter checkpoint: {resolved}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "path": str(resolved),
        "config_sha256": sha256(config_path),
        "weights_file": weights_path.name,
        "weights_sha256": sha256(weights_path),
        "r": int(config.get("r") or 0),
        "lora_alpha": int(config.get("lora_alpha") or 0),
    }


def stratified_eval_rows(
    rows: Sequence[Mapping[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Round-robin Dev rows across dataset/task groups without reading Gold."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for value in rows:
        row = dict(value)
        key = (str(row.get("dataset") or ""), str(row.get("task") or ""))
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: str(row.get("id") or ""))
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in groups}
    while len(selected) < min(int(count), len(rows)):
        progressed = False
        for key in sorted(groups):
            offset = offsets[key]
            if offset >= len(groups[key]):
                continue
            selected.append(groups[key][offset])
            offsets[key] = offset + 1
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        help="continue from an existing PEFT adapter while resetting optimizer state",
    )
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--dev-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-response-tokens", type=int, default=192)
    parser.add_argument(
        "--bptt-window",
        type=int,
        default=8,
        help="detach recurrent state between bounded response-training windows",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--bfcl-repeat", type=int, default=3)
    parser.add_argument(
        "--dataset-repeat",
        action="append",
        default=[],
        metavar="DATASET=COUNT",
        help="repeat all rows of one dataset COUNT times in the weighted epoch",
    )
    parser.add_argument("--include-dataset", action="append", default=[])
    parser.add_argument("--include-task", action="append", default=[])
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args(argv)

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if (
        args.max_length < 128
        or args.max_response_tokens <= 0
        or args.bptt_window <= 0
        or args.max_steps <= 0
        or args.gradient_accumulation <= 0
        or args.eval_samples <= 0
    ):
        parser.error("invalid training dimensions")
    dataset_repeats: dict[str, int] = {}
    for value in args.dataset_repeat:
        name, separator, count_text = str(value).partition("=")
        if not separator or not name.strip():
            parser.error(f"invalid --dataset-repeat value: {value!r}")
        try:
            count = int(count_text)
        except ValueError:
            parser.error(f"invalid --dataset-repeat count: {value!r}")
        if count <= 0:
            parser.error(f"dataset repeat must be positive: {value!r}")
        dataset_repeats[name.strip()] = count
    dataset_repeats["bfcl"] = max(
        int(args.bfcl_repeat), dataset_repeats.get("bfcl", 1)
    )
    include_datasets = set(map(str, args.include_dataset))
    include_tasks = set(map(str, args.include_task))
    model_path = args.model.expanduser().resolve()
    init_adapter = (
        adapter_input_manifest(args.init_adapter)
        if args.init_adapter is not None
        else None
    )
    if init_adapter is not None and (
        init_adapter["r"] != int(args.lora_rank)
        or init_adapter["lora_alpha"] != int(args.lora_alpha)
    ):
        parser.error(
            "--init-adapter rank/alpha must match --lora-rank and --lora-alpha"
        )
    train_path = args.train_data.expanduser().resolve()
    dev_path = args.dev_data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        model=str(model_path),
        init_adapter=None if init_adapter is None else str(init_adapter["path"]),
        train_data=str(train_path),
        dev_data=str(dev_path),
        output_dir=str(output_dir),
        seed=args.seed,
        max_length=args.max_length,
        max_response_tokens=args.max_response_tokens,
        bptt_window=args.bptt_window,
        max_steps=args.max_steps,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        bfcl_repeat=args.bfcl_repeat,
        dataset_repeats=dataset_repeats,
        include_datasets=sorted(include_datasets),
        include_tasks=sorted(include_tasks),
        eval_samples=args.eval_samples,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    train_rows = load_records(train_path)
    dev_rows = load_records(dev_path)
    if include_datasets:
        train_rows = [
            row for row in train_rows
            if str(row.get("dataset") or "") in include_datasets
        ]
        dev_rows = [
            row for row in dev_rows
            if str(row.get("dataset") or "") in include_datasets
        ]
    if include_tasks:
        train_rows = [
            row for row in train_rows
            if str(row.get("task") or "") in include_tasks
        ]
        dev_rows = [
            row for row in dev_rows
            if str(row.get("task") or "") in include_tasks
        ]
    if not train_rows or not dev_rows:
        raise ValueError("dataset/task filters produced an empty Train or Dev split")
    weighted_train: list[dict[str, Any]] = []
    for row in train_rows:
        weighted_train.extend(
            [row] * dataset_repeats.get(str(row.get("dataset") or ""), 1)
        )
    rng = random.Random(args.seed)
    rng.shuffle(weighted_train)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    class EncodedDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return encode_example(
                tokenizer,
                self.rows[index],
                max_length=args.max_length,
                max_response_tokens=args.max_response_tokens,
            )

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = min(
            args.max_length,
            int(math.ceil(max(len(row["input_ids"]) for row in batch) / 8) * 8),
        )
        input_ids = []
        attention_mask = []
        labels = []
        for row in batch:
            padding = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [tokenizer.pad_token_id] * padding)
            attention_mask.append(row["attention_mask"] + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.config.fuse_cross_entropy = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if init_adapter is not None:
        model = PeftModel.from_pretrained(
            model,
            str(init_adapter["path"]),
            is_trainable=True,
        )
    else:
        lora = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
        )
        model = get_peft_model(model, lora)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)

    train_loader = DataLoader(
        EncodedDataset(weighted_train),
        batch_size=1,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(args.seed),
    )
    eval_rows = stratified_eval_rows(dev_rows, args.eval_samples)
    dev_loader = DataLoader(
        EncodedDataset(eval_rows),
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
    )

    def lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return max(1e-3, (step + 1) / max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    def detached_bptt_backward(
        batch: Mapping[str, torch.Tensor],
        *,
        loss_scale: float,
    ) -> float:
        """Train only response tokens with bounded recurrent autograd windows.

        Prompt state is prefetched without gradients.  Each response window uses
        the real recurrent cache, then detaches it before the next window.  This
        is truncated BPTT rather than fake independent-token training and keeps
        QLoRA memory bounded on a 16 GiB RTX 4080.
        """

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        if int(input_ids.shape[0]) != 1:
            raise ValueError("detached BPTT currently requires batch_size=1")
        supervised = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
        if supervised.numel() == 0:
            raise ValueError("training example has no supervised next-token labels")
        first_predictor = int(supervised[0].item())
        final_predictor_exclusive = int(supervised[-1].item()) + 1
        total_targets = int(supervised.numel())
        cache = None
        if first_predictor > 0:
            with torch.no_grad():
                prefix = model(
                    input_ids=input_ids[:, :first_predictor],
                    use_cache=True,
                    logits_to_keep=1,
                )
            cache = prefix.past_key_values
            detach = getattr(cache, "detach", None)
            if callable(detach):
                cache = detach(inplace=True)

        raw_loss_sum = 0.0
        position = first_predictor
        while position < final_predictor_exclusive:
            end = min(position + args.bptt_window, final_predictor_exclusive)
            output = model(
                input_ids=input_ids[:, position:end],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            detach = getattr(cache, "detach", None)
            if callable(detach):
                cache = detach(inplace=True)
            targets = labels[:, position + 1 : end + 1]
            logits = output.logits[:, : targets.shape[1], :]
            loss_sum = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            (loss_sum * (float(loss_scale) / total_targets)).backward()
            raw_loss_sum += float(loss_sum.detach().cpu())
            del output, logits, targets, loss_sum
            position = end
        return raw_loss_sum / total_targets
    metadata = {
        "schema_version": "rwkv-agent-fitgen-training.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "inputs": {
            "train_sha256": sha256(train_path),
            "dev_sha256": sha256(dev_path),
            "model_config_sha256": sha256(model_path / "config.json"),
            "init_adapter": init_adapter,
        },
        "records": {"train": len(train_rows), "weighted_train": len(weighted_train), "dev": len(dev_rows)},
        "eval_record_ids": [str(row.get("id") or "") for row in eval_rows],
        "trainable_parameters": trainable_count,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": torch.cuda.get_device_name(0),
    }
    _json_dump(output_dir / "run-config.json", metadata)
    metrics_path = output_dir / "metrics.jsonl"

    @torch.no_grad()
    def evaluate(step: int) -> float:
        model.eval()
        losses = []
        for batch in dev_loader:
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses.append(float(model(**batch, use_cache=False).loss.detach().cpu()))
        model.train()
        value = sum(losses) / max(1, len(losses))
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, "dev_loss": value, "event": "eval"}) + "\n")
        return value

    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    for step in range(1, args.max_steps + 1):
        losses = []
        for _micro in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_value = detached_bptt_backward(
                    batch,
                    loss_scale=1.0 / args.gradient_accumulation,
                )
            losses.append(loss_value)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        event = {
            "step": step,
            "train_loss": sum(losses) / len(losses),
            "learning_rate": scheduler.get_last_lr()[0],
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        print(json.dumps(event, sort_keys=True), flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            checkpoint = output_dir / f"checkpoint-{step:06d}"
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
        if step % args.eval_every == 0 or step == args.max_steps:
            print(json.dumps({"step": step, "dev_loss": evaluate(step)}), flush=True)

    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["final_checkpoint"] = str(output_dir / f"checkpoint-{args.max_steps:06d}")
    metadata["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    _json_dump(output_dir / "run-config.json", metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
