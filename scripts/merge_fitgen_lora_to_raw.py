#!/usr/bin/env python3
"""Merge a FitGen PEFT LoRA adapter into a raw RWKV-7 ``.pth`` checkpoint.

The production native runtime consumes BlinkDL-style raw keys while training
uses the Hugging Face wrapper.  This converter performs the small, explicit
name translation and records hashes so an isolated evaluation sidecar can load
the resulting checkpoint without changing the Stable Release.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence


ADAPTER_KEY = re.compile(
    r"^base_model\.model\.model\.layers\.(?P<layer>\d+)\."
    r"(?P<block>attn|ffn)\.(?P<module>r_proj|k_proj|v_proj|o_proj|key|value)\."
    r"lora_(?P<side>A|B)\.weight$"
)

RAW_SUFFIX = {
    ("attn", "r_proj"): "att.receptance.weight",
    ("attn", "k_proj"): "att.key.weight",
    ("attn", "v_proj"): "att.value.weight",
    ("attn", "o_proj"): "att.output.weight",
    ("ffn", "key"): "ffn.key.weight",
    ("ffn", "value"): "ffn.value.weight",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_key(adapter_key: str) -> tuple[str, str] | None:
    match = ADAPTER_KEY.fullmatch(adapter_key)
    if not match:
        return None
    suffix = RAW_SUFFIX.get((match.group("block"), match.group("module")))
    if suffix is None:
        return None
    return f"blocks.{int(match.group('layer'))}.{suffix}", match.group("side")


def _load_adapter(path: Path) -> tuple[dict[str, Any], Path]:
    safe = path / "adapter_model.safetensors"
    binary = path / "adapter_model.bin"
    if safe.is_file():
        from safetensors.torch import load_file

        return dict(load_file(str(safe), device="cpu")), safe
    if binary.is_file():
        import torch

        return dict(torch.load(binary, map_location="cpu", weights_only=True)), binary
    raise FileNotFoundError(f"adapter weights not found under {path}")


def _state_dict(value: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(value, dict):
        raise TypeError("raw checkpoint must be a dictionary")
    for key in ("state_dict", "model"):
        nested = value.get(key)
        if isinstance(nested, dict) and any(str(name).startswith("blocks.") for name in nested):
            return nested, key
    if any(str(name).startswith("blocks.") for name in value):
        return value, None
    raise ValueError("raw checkpoint contains no blocks.* tensors")


def merge(
    raw_model: Path,
    adapter_dir: Path,
    output_model: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    import torch

    raw_model = raw_model.expanduser().resolve()
    adapter_dir = adapter_dir.expanduser().resolve()
    output_model = output_model.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    if output_model.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite merged model or manifest")
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    scale = alpha / rank
    adapter, adapter_weights_path = _load_adapter(adapter_dir)

    pairs: dict[str, dict[str, Any]] = {}
    unexpected: list[str] = []
    for key, tensor in adapter.items():
        translated = raw_key(str(key))
        if translated is None:
            unexpected.append(str(key))
            continue
        target, side = translated
        pairs.setdefault(target, {})[side] = tensor
    if unexpected:
        raise ValueError(f"unsupported adapter keys: {unexpected[:8]}")
    incomplete = [key for key, value in pairs.items() if set(value) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"incomplete LoRA pairs: {incomplete[:8]}")
    if not pairs:
        raise ValueError("adapter contains no recognized LoRA pairs")

    raw_container = torch.load(
        raw_model,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    state, wrapper_key = _state_dict(raw_container)
    modified: list[dict[str, Any]] = []
    with torch.no_grad():
        for target in sorted(pairs):
            if target not in state:
                raise KeyError(f"raw tensor missing for adapter pair: {target}")
            base = state[target]
            left = pairs[target]["B"]
            right = pairs[target]["A"]
            if int(right.shape[0]) != rank or int(left.shape[1]) != rank:
                raise ValueError(
                    f"rank mismatch for {target}: A={tuple(right.shape)} B={tuple(left.shape)}"
                )
            if tuple(base.shape) != (int(left.shape[0]), int(right.shape[1])):
                raise ValueError(
                    f"shape mismatch for {target}: raw={tuple(base.shape)} "
                    f"A={tuple(right.shape)} B={tuple(left.shape)}"
                )
            delta = torch.matmul(left.float(), right.float()).mul_(scale)
            # BlinkDL raw checkpoints pack nearly every BF16 model tensor as a
            # view into one contiguous storage.  Replacing 192 dictionary
            # values breaks that aliasing and bloats a 14GB checkpoint to
            # roughly 26GB.  MAP_PRIVATE mmap storage is writable by this
            # process, so update the original view in place: torch.save then
            # preserves the compact shared storage without changing the input
            # file on disk.
            base.add_(delta.to(dtype=base.dtype))
            modified.append(
                {
                    "raw_key": target,
                    "shape": list(base.shape),
                    "dtype": str(base.dtype),
                }
            )
            del delta
    if wrapper_key is not None:
        raw_container[wrapper_key] = state
    else:
        raw_container = state
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw_container, output_model)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "rwkv-agent-fitgen-lora-raw-merge.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_input": str(raw_model),
        "raw_input_sha256": sha256(raw_model),
        "adapter_dir": str(adapter_dir),
        "adapter_config_sha256": sha256(config_path),
        "adapter_weights_sha256": sha256(adapter_weights_path),
        "rank": rank,
        "alpha": alpha,
        "scale": scale,
        "modified_tensors": len(modified),
        "modified": modified,
        "output_model": str(output_model),
        "output_sha256": sha256(output_model),
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-model", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    value = merge(args.raw_model, args.adapter_dir, args.output_model, args.manifest)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
