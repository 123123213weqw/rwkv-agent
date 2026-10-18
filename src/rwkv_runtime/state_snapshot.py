"""Safe, bounded wire encoding for exact RWKV recurrent State tensors.

The format intentionally contains only a small canonical JSON manifest and a
``safetensors`` payload.  It never imports Python objects from a checkpoint and
therefore avoids the code-execution semantics of pickle/``torch.load``.
Scheduler backends remain responsible for validating every tensor name, shape
and dtype against the currently loaded model before installing a snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import json
import struct
from typing import Any


MAGIC = b"RWKVSTATE\x01"
MAX_MANIFEST_BYTES = 64 * 1024
DEFAULT_MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024


def encode_state_snapshot(
    *,
    manifest: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> bytes:
    """Encode a backend-produced manifest and CPU tensor map."""

    from safetensors.torch import save

    if not tensors:
        raise ValueError("snapshot tensor map must not be empty")
    names = list(tensors)
    if len(names) != len(set(names)) or any(
        not name or len(name) > 128 or not name.isascii() for name in names
    ):
        raise ValueError("snapshot tensor names must be unique bounded ASCII")
    normalized = {}
    for name, tensor in tensors.items():
        if not hasattr(tensor, "detach"):
            raise TypeError(f"snapshot value {name} is not a tensor")
        normalized[name] = tensor.detach().to(device="cpu").contiguous()
    document = dict(manifest)
    document["tensor_names"] = sorted(names)
    encoded_manifest = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded_manifest) > MAX_MANIFEST_BYTES:
        raise ValueError("snapshot manifest is too large")
    encoded_tensors = save(normalized)
    body = MAGIC + struct.pack(">I", len(encoded_manifest)) + encoded_manifest + encoded_tensors
    return body + hashlib.sha256(body).digest()


def decode_state_snapshot(
    payload: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode a bounded snapshot without constructing arbitrary objects."""

    from safetensors.torch import load

    if max_bytes < 1 or len(payload) > max_bytes:
        raise ValueError("snapshot exceeds configured byte limit")
    header_size = len(MAGIC) + 4
    if len(payload) <= header_size + 32 or not payload.startswith(MAGIC):
        raise ValueError("unsupported RWKV State snapshot format")
    body, supplied_digest = payload[:-32], payload[-32:]
    if not hmac.compare_digest(hashlib.sha256(body).digest(), supplied_digest):
        raise ValueError("snapshot integrity digest mismatch")
    manifest_size = struct.unpack(">I", payload[len(MAGIC) : header_size])[0]
    if manifest_size < 2 or manifest_size > MAX_MANIFEST_BYTES:
        raise ValueError("invalid snapshot manifest size")
    manifest_end = header_size + manifest_size
    if manifest_end >= len(body):
        raise ValueError("truncated snapshot payload")
    try:
        manifest = json.loads(payload[header_size:manifest_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid snapshot manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("snapshot manifest must be an object")
    try:
        tensors = load(body[manifest_end:])
    except Exception as exc:
        raise ValueError("invalid safetensors snapshot payload") from exc
    declared = manifest.get("tensor_names")
    if not isinstance(declared, list) or declared != sorted(tensors):
        raise ValueError("snapshot tensor manifest mismatch")
    return manifest, tensors
