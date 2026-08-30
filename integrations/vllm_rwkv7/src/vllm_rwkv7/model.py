"""vLLM 0.7 recurrent-cache adapter for Hugging Face RWKV-7 models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from importlib.metadata import version
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsAttentionFree
from vllm.model_executor.models.mamba_cache import MambaCacheManager
from vllm.model_executor.sampling_metadata import SamplingMetadata


def _require_supported_vllm() -> None:
    release = version("vllm")
    if not release.startswith("0.7."):
        raise RuntimeError(
            "rwkv7-vllm-plugin 0.1 supports vLLM 0.7.x; "
            f"found {release}"
        )


def _query_bounds(
    input_ids: torch.Tensor,
    state_indices: torch.Tensor,
    attn_metadata: Any,
) -> list[tuple[int, int]]:
    """Return one flattened token interval for every recurrent State row."""

    token_count = int(input_ids.numel())
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if query_start_loc is None:
        bounds = [(index, index + 1) for index in range(token_count)]
    else:
        offsets = [
            int(value) for value in query_start_loc.detach().cpu().tolist()
        ]
        bounds = list(zip(offsets[:-1], offsets[1:], strict=True))
    if len(bounds) != int(state_indices.numel()):
        raise RuntimeError(
            "RWKV State rows do not match vLLM query metadata: "
            f"{len(bounds)} queries for {int(state_indices.numel())} rows"
        )
    if not bounds or bounds[0][0] != 0 or bounds[-1][1] != token_count:
        raise RuntimeError("invalid vLLM query_start_loc for RWKV input")
    if any(start < 0 or end <= start for start, end in bounds):
        raise RuntimeError("RWKV does not accept empty or reversed query chunks")
    return bounds


class VllmRWKV7ForCausalLM(nn.Module, HasInnerState, IsAttentionFree):
    """Use vLLM scheduling/sampling with the model repository's RWKV kernel."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        del prefix
        _require_supported_vllm()
        super().__init__()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = self.model_config.hf_config
        parallel = vllm_config.parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
        ):
            raise ValueError("RWKV-7 vLLM adapter currently requires TP=1 and PP=1")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("RWKV-7 vLLM adapter does not support prefix caching")
        if not self.model_config.enforce_eager:
            raise ValueError("RWKV-7 vLLM adapter requires --enforce-eager")
        if vllm_config.lora_config is not None:
            raise ValueError("RWKV-7 vLLM adapter does not support LoRA")
        if vllm_config.quant_config is not None:
            raise ValueError("RWKV-7 vLLM adapter does not support quantization")
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "RWKV-7 vLLM adapter does not support speculative decoding"
            )

        self.hf_model = AutoModelForCausalLM.from_config(
            self.config,
            trust_remote_code=True,
        )
        if self.hf_model.__class__.__name__ != "RWKV7ForCausalLM":
            raise TypeError(
                "model repository must expose the RWKV7ForCausalLM architecture"
            )
        cache_class = self.hf_model.model.forward.__globals__.get("RWKV7Cache")
        if cache_class is None:
            raise TypeError("model repository does not expose canonical RWKV7Cache")
        self.cache_class = cache_class
        self.unpadded_vocab_size = int(self.config.vocab_size)
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            self.unpadded_vocab_size,
            logits_as_input=True,
        )
        self.sampler = get_sampler()
        self.mamba_cache: MambaCacheManager | None = None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.hf_model.model.embeddings(input_ids)

    def _cache_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        shift = (2, int(self.config.hidden_size))
        recurrent = (
            int(self.config.num_heads),
            int(self.config.head_dim),
            int(self.config.head_dim),
        )
        return shift, recurrent

    def _ensure_cache(self) -> MambaCacheManager:
        if self.mamba_cache is None:
            shift_shape, recurrent_shape = self._cache_shapes()
            self.mamba_cache = MambaCacheManager(
                self.vllm_config,
                torch.float32,
                int(self.config.num_hidden_layers),
                shift_shape,
                recurrent_shape,
            )
        return self.mamba_cache

    @staticmethod
    def _zero_new_rows(
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        slots: torch.Tensor,
        starts: torch.Tensor,
    ) -> None:
        new_rows = slots[starts.eq(0)]
        if int(new_rows.numel()) == 0:
            return
        conv_state.index_fill_(1, new_rows, 0)
        ssm_state.index_fill_(1, new_rows, 0)

    def _make_cache(
        self,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        slots: torch.Tensor,
        seen_tokens: int,
    ):
        return self.cache_class(
            recurrent_state=[
                layer.index_select(0, slots) for layer in ssm_state
            ],
            attention_shift=[
                layer[:, 0].index_select(0, slots) for layer in conv_state
            ],
            ffn_shift=[
                layer[:, 1].index_select(0, slots) for layer in conv_state
            ],
            seen_tokens=seen_tokens,
        )

    @staticmethod
    def _store_cache(
        output_cache: Any,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        for layer_idx, recurrent in enumerate(output_cache.recurrent_state):
            if recurrent is None:
                raise RuntimeError(
                    "RWKV model returned an incomplete recurrent cache"
                )
            ssm_state[layer_idx].index_copy_(
                0, slots, recurrent.to(dtype=ssm_state.dtype)
            )
            conv_state[layer_idx, :, 0].index_copy_(
                0,
                slots,
                output_cache.attention_shift[layer_idx].to(
                    dtype=conv_state.dtype
                ),
            )
            conv_state[layer_idx, :, 1].index_copy_(
                0,
                slots,
                output_cache.ffn_shift[layer_idx].to(dtype=conv_state.dtype),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[Any],
        attn_metadata: Any,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kv_caches
        if intermediate_tensors is not None:
            raise ValueError(
                "RWKV-7 adapter does not support pipeline intermediates"
            )
        if inputs_embeds is not None:
            raise ValueError("RWKV-7 adapter currently requires input_ids")
        if input_ids.ndim != 1 or positions.ndim != 1:
            raise ValueError(
                "vLLM must provide flattened input_ids and positions"
            )

        cache_params = self._ensure_cache().current_run_tensors(**kwargs)
        state_indices = cache_params.state_indices_tensor.to(dtype=torch.long)
        negative = state_indices.lt(0)
        if bool(negative.any().item()):
            # vLLM's memory profiler deliberately marks every synthetic
            # request as already finished, so MambaCacheManager returns only
            # PAD_SLOT_ID rows even in eager mode. Use all cache rows as
            # scratch for that all-padding pass; mixed real/padding rows remain
            # rejected because they could alias a live request State.
            if not bool(negative.all().item()):
                raise RuntimeError(
                    "mixed padded RWKV cache rows require --enforce-eager"
                )
            if int(state_indices.numel()) > int(cache_params.conv_state.shape[1]):
                raise RuntimeError("RWKV profile batch exceeds recurrent cache")
            state_indices = torch.arange(
                int(state_indices.numel()),
                device=state_indices.device,
                dtype=torch.long,
            )
        bounds = _query_bounds(input_ids, state_indices, attn_metadata)

        groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for row, (start, end) in enumerate(bounds):
            groups[end - start].append((row, start, end))

        hidden = torch.empty(
            (int(input_ids.numel()), int(self.config.hidden_size)),
            device=input_ids.device,
            dtype=self.hf_model.model.embeddings.weight.dtype,
        )
        for query_length, items in groups.items():
            rows = torch.tensor(
                [row for row, _start, _end in items],
                device=state_indices.device,
                dtype=torch.long,
            )
            slots = state_indices.index_select(0, rows)
            starts = torch.stack(
                [positions[start] for _row, start, _end in items]
            )
            self._zero_new_rows(
                cache_params.conv_state,
                cache_params.ssm_state,
                slots,
                starts,
            )
            batch_ids = torch.stack(
                [input_ids[start:end] for _row, start, end in items]
            )
            state = self._make_cache(
                cache_params.conv_state,
                cache_params.ssm_state,
                slots,
                int(starts.min().item()),
            )
            outputs = self.hf_model.model(
                input_ids=batch_ids,
                past_key_values=state,
                use_cache=True,
                return_dict=True,
            )
            self._store_cache(
                outputs.past_key_values,
                cache_params.conv_state,
                cache_params.ssm_state,
                slots,
            )
            if tuple(outputs.last_hidden_state.shape[:2]) != (
                len(items),
                query_length,
            ):
                raise RuntimeError(
                    "RWKV model returned an unexpected hidden shape"
                )
            for output_row, (_row, start, end) in enumerate(items):
                hidden[start:end].copy_(outputs.last_hidden_state[output_row])
        return hidden

    def copy_inputs_before_cuda_graphs(
        self, input_buffers: Any, **kwargs: Any
    ):
        return self._ensure_cache().copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs
        )

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self._ensure_cache().get_seqlen_agnostic_capture_inputs(batch_size)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor | None:
        selected = sampling_metadata.selected_token_indices
        if selected is not None:
            hidden_states = hidden_states.index_select(0, selected)
        logits = self.hf_model.lm_head(hidden_states)
        return self.logits_processor(None, logits, sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor | None,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.sampler(logits, sampling_metadata)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        parameters = dict(self.hf_model.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            parameter = parameters.get(name)
            if parameter is None:
                raise KeyError(f"unexpected RWKV checkpoint tensor: {name}")
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, weight)
            loaded.add(name)
        missing = set(parameters).difference(loaded)
        if missing:
            preview = ", ".join(sorted(missing)[:8])
            raise KeyError(f"RWKV checkpoint is missing parameters: {preview}")
        # vLLM validates the names returned by this hook against the wrapper's
        # full ``named_parameters()`` paths, not the checkpoint paths passed
        # into the hook.
        return {f"hf_model.{name}" for name in loaded}


__all__ = ["VllmRWKV7ForCausalLM"]
