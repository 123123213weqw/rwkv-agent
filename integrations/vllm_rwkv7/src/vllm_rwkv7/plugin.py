"""Lazy vLLM model-registry entry point."""

from __future__ import annotations


def register() -> None:
    """Register without importing torch/CUDA in vLLM's discovery process."""

    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "RWKV7ForCausalLM",
        "vllm_rwkv7.model:VllmRWKV7ForCausalLM",
    )


__all__ = ["register"]

