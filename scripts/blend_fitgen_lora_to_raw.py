#!/usr/bin/env python3
"""Merge an interpolation of two FitGen LoRA adapters into a raw checkpoint.

This is a validation-time line search between a known-stable adapter and a new
adapter.  Interpolation is performed in weight-update space, not by averaging
LoRA factors::

    W = W_raw + (1 - weight) * scale_a * (B_a @ A_a)
              + weight       * scale_b * (B_b @ A_b)

The raw input is never modified and every source/output hash is frozen in the
manifest.  The command refuses extrapolation and existing outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.merge_fitgen_lora_to_raw import (
    _load_adapter,
    _state_dict,
    raw_key,
    sha256,
)


def _load_pairs(adapter_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    tensors, weights_path = _load_adapter(adapter_dir)
    pairs: dict[str, dict[str, Any]] = {}
    unexpected: list[str] = []
    for key, tensor in tensors.items():
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
    for target, value in pairs.items():
        left, right = value["B"], value["A"]
        if int(right.shape[0]) != rank or int(left.shape[1]) != rank:
            raise ValueError(
                f"rank mismatch for {target}: A={tuple(right.shape)} B={tuple(left.shape)}"
            )
    return pairs, {
        "dir": str(adapter_dir),
        "config_sha256": sha256(config_path),
        "weights_sha256": sha256(weights_path),
        "rank": rank,
        "alpha": alpha,
        "scale": alpha / rank,
    }


def blend(
    raw_model: Path,
    reference_adapter_dir: Path,
    candidate_adapter_dir: Path,
    candidate_weight: float,
    output_model: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    raw_model = raw_model.expanduser().resolve()
    reference_adapter_dir = reference_adapter_dir.expanduser().resolve()
    candidate_adapter_dir = candidate_adapter_dir.expanduser().resolve()
    output_model = output_model.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    weight = float(candidate_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("candidate_weight must be between 0 and 1")

    import torch

    if output_model.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite blended model or manifest")

    reference, reference_meta = _load_pairs(reference_adapter_dir)
    candidate, candidate_meta = _load_pairs(candidate_adapter_dir)
    if set(reference) != set(candidate):
        missing_reference = sorted(set(candidate) - set(reference))[:8]
        missing_candidate = sorted(set(reference) - set(candidate))[:8]
        raise ValueError(
            "adapter target mismatch: "
            f"missing_reference={missing_reference} "
            f"missing_candidate={missing_candidate}"
        )

    raw_container = torch.load(
        raw_model,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    state, wrapper_key = _state_dict(raw_container)
    modified: list[dict[str, Any]] = []
    reference_weight = 1.0 - weight
    with torch.no_grad():
        for target in sorted(reference):
            if target not in state:
                raise KeyError(f"raw tensor missing for adapter pair: {target}")
            base = state[target]
            a_ref, b_ref = reference[target]["A"], reference[target]["B"]
            a_new, b_new = candidate[target]["A"], candidate[target]["B"]
            if tuple(a_ref.shape) != tuple(a_new.shape) or tuple(b_ref.shape) != tuple(
                b_new.shape
            ):
                raise ValueError(f"adapter shape mismatch for {target}")
            expected = (int(b_ref.shape[0]), int(a_ref.shape[1]))
            if tuple(base.shape) != expected:
                raise ValueError(
                    f"shape mismatch for {target}: raw={tuple(base.shape)} expected={expected}"
                )
            delta = torch.matmul(b_ref.float(), a_ref.float()).mul_(
                reference_weight * float(reference_meta["scale"])
            )
            delta.add_(
                torch.matmul(b_new.float(), a_new.float()),
                alpha=weight * float(candidate_meta["scale"]),
            )
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
        "schema_version": "rwkv-agent-fitgen-lora-raw-blend.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_input": str(raw_model),
        "raw_input_sha256": sha256(raw_model),
        "reference_adapter": reference_meta,
        "candidate_adapter": candidate_meta,
        "reference_weight": reference_weight,
        "candidate_weight": weight,
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
    parser.add_argument("--reference-adapter-dir", type=Path, required=True)
    parser.add_argument("--candidate-adapter-dir", type=Path, required=True)
    parser.add_argument("--candidate-weight", type=float, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    value = blend(
        args.raw_model,
        args.reference_adapter_dir,
        args.candidate_adapter_dir,
        args.candidate_weight,
        args.output_model,
        args.manifest,
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
