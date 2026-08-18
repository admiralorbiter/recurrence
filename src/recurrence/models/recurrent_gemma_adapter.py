"""Upstream RecurrentGemma Model Adapter & State Plumbing (Sprint S10 / S10.1).

Instruments the official upstream Hugging Face RecurrentGemma model (`google/recurrentgemma-2b`),
wrapping module-internal recurrence (rg_lru.recurrent_states), temporal convolution (conv1d_state),
and attention sliding KV cache into the clean layer-indexed RecurrentStateSnapshot.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    DynamicCache,
    RecurrentGemmaConfig,
    RecurrentGemmaForCausalLM,
)

from recurrence.state.temporal_inventory import RecurrentStateSnapshot


class RecurrentGemmaAdapter:
    """Adapter wrapping upstream RecurrentGemmaForCausalLM to expose explicit temporal state."""

    def __init__(
        self,
        model: Optional[RecurrentGemmaForCausalLM] = None,
        config: Optional[RecurrentGemmaConfig] = None,
        tokenizer: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if model is not None:
            self.model = model
            self.config = model.config
            # S10.1: Preserve caller device and dtype if not explicitly provided
            first_param = next(model.parameters(), None)
            self.device = torch.device(device) if device is not None else (first_param.device if first_param is not None else torch.device("cpu"))
            self.dtype = dtype if dtype is not None else (first_param.dtype if first_param is not None else torch.float32)
            if device is not None or dtype is not None:
                self.model = self.model.to(device=self.device, dtype=self.dtype)
        elif config is not None:
            self.config = config
            self.device = torch.device(device) if device is not None else torch.device("cpu")
            self.dtype = dtype if dtype is not None else torch.float32
            self.model = RecurrentGemmaForCausalLM(config).to(device=self.device, dtype=self.dtype)
        else:
            # Default lightweight reference configuration for testing
            self.config = RecurrentGemmaConfig(
                num_hidden_layers=4,
                hidden_size=128,
                intermediate_size=256,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                lru_width=128,
                conv1d_width=4,
                sliding_window=8,
                block_types=["recurrent", "recurrent", "attention", "recurrent"],
                vocab_size=1000,
            )
            self.device = torch.device(device) if device is not None else torch.device("cpu")
            self.dtype = dtype if dtype is not None else torch.float32
            self.model = RecurrentGemmaForCausalLM(self.config).to(device=self.device, dtype=self.dtype)

        self.model.eval()
        self.tokenizer = tokenizer

    def create_canonical_initial_state(
        self,
        batch_size: int = 1,
    ) -> RecurrentStateSnapshot:
        """Create a canonical zero-initialized state snapshot for all layers."""
        rglru: Dict[int, torch.Tensor] = {}
        conv: Dict[int, torch.Tensor] = {}
        kv: Dict[int, Dict[str, Any]] = {}

        for l_idx, layer in enumerate(self.model.model.layers):
            block = layer.temporal_block
            if hasattr(block, "rg_lru"):
                lru_width = getattr(self.config, "lru_width", self.config.hidden_size)
                rglru[l_idx] = torch.zeros(
                    (batch_size, lru_width),
                    device=self.device,
                    dtype=torch.float32,
                )
                conv1d_width = getattr(self.config, "conv1d_width", 4)
                conv[l_idx] = torch.zeros(
                    (batch_size, self.config.hidden_size, conv1d_width - 1),
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                # Attention layer
                kv[l_idx] = {
                    "key": torch.empty(
                        (batch_size, self.config.num_key_value_heads, 0, self.config.head_dim),
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    "value": torch.empty(
                        (batch_size, self.config.num_key_value_heads, 0, self.config.head_dim),
                        device=self.device,
                        dtype=self.dtype,
                    ),
                    "cumulative_length": 0,
                    "sliding_window": getattr(self.config, "sliding_window", None),
                }

        return RecurrentStateSnapshot(
            rglru=rglru,
            conv=conv,
            kv=kv,
            cache_position=0,
            device=self.device,
            dtype=self.dtype,
            metadata={"batch_size": batch_size, "initial_state": True},
        )

    def extract_state_snapshot(
        self,
        past_key_values: Optional[Any] = None,
        cache_position: int = 0,
    ) -> RecurrentStateSnapshot:
        """Extract current layer-indexed state from model modules and KV cache."""
        rglru: Dict[int, torch.Tensor] = {}
        conv: Dict[int, torch.Tensor] = {}
        kv: Dict[int, Dict[str, Any]] = {}

        for l_idx, layer in enumerate(self.model.model.layers):
            block = layer.temporal_block
            if hasattr(block, "rg_lru"):
                if block.rg_lru.recurrent_states is not None:
                    rglru[l_idx] = block.rg_lru.recurrent_states.detach().clone()
                if block.conv1d_state is not None:
                    conv[l_idx] = block.conv1d_state.detach().clone()
            else:
                # Attention layer: extract from past_key_values cache
                if past_key_values is not None and hasattr(past_key_values, "layers"):
                    if l_idx < len(past_key_values.layers):
                        layer_cache = past_key_values.layers[l_idx]
                        if getattr(layer_cache, "is_initialized", False) and getattr(layer_cache, "keys", None) is not None:
                            kv[l_idx] = {
                                "key": layer_cache.keys.detach().clone(),
                                "value": layer_cache.values.detach().clone(),
                                "cumulative_length": getattr(layer_cache, "cumulative_length", layer_cache.keys.shape[-2]),
                                "sliding_window": getattr(layer_cache, "sliding_window", getattr(self.config, "sliding_window", None)),
                            }
                        else:
                            kv[l_idx] = {
                                "key": torch.empty((1, self.config.num_key_value_heads, 0, self.config.head_dim), device=self.device, dtype=self.dtype),
                                "value": torch.empty((1, self.config.num_key_value_heads, 0, self.config.head_dim), device=self.device, dtype=self.dtype),
                                "cumulative_length": 0,
                                "sliding_window": getattr(self.config, "sliding_window", None),
                            }

        return RecurrentStateSnapshot(
            rglru=rglru,
            conv=conv,
            kv=kv,
            cache_position=cache_position,
            device=self.device,
            dtype=self.dtype,
        )

    def inject_state_snapshot(
        self,
        snapshot: RecurrentStateSnapshot,
    ) -> DynamicCache:
        """Inject state snapshot into model modules and construct past_key_values cache."""
        cache = DynamicCache(config=self.config)

        for l_idx, layer in enumerate(self.model.model.layers):
            block = layer.temporal_block
            if hasattr(block, "rg_lru"):
                if l_idx in snapshot.rglru:
                    block.rg_lru.recurrent_states = snapshot.rglru[l_idx].detach().clone().to(device=self.device)
                if l_idx in snapshot.conv:
                    block.conv1d_state = snapshot.conv[l_idx].detach().clone().to(device=self.device, dtype=self.dtype)
            else:
                # Attention layer: populate into cache layers
                if l_idx in snapshot.kv and l_idx < len(cache.layers):
                    k = snapshot.kv[l_idx]["key"].detach().clone().to(device=self.device, dtype=self.dtype)
                    v = snapshot.kv[l_idx]["value"].detach().clone().to(device=self.device, dtype=self.dtype)
                    layer_cache = cache.layers[l_idx]
                    layer_cache.keys = k
                    layer_cache.values = v
                    layer_cache.is_initialized = True
                    layer_cache.device = self.device
                    layer_cache.dtype = self.dtype
                    layer_cache.cumulative_length = snapshot.kv[l_idx].get("cumulative_length", k.shape[-2])
                    if "sliding_window" in snapshot.kv[l_idx] and hasattr(layer_cache, "sliding_window"):
                        layer_cache.sliding_window = snapshot.kv[l_idx]["sliding_window"]

        return cache

    @torch.no_grad()
    def step(
        self,
        token_id: int,
        snapshot: RecurrentStateSnapshot,
    ) -> Tuple[torch.Tensor, RecurrentStateSnapshot]:
        """Execute a single token step and return next-token logits and advanced state snapshot."""
        cache = self.inject_state_snapshot(snapshot)
        input_ids = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
        pos = snapshot.cache_position
        position_ids = torch.tensor([[pos]], device=self.device, dtype=torch.long)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        next_logits = outputs.logits[:, -1, :]
        new_snapshot = self.extract_state_snapshot(
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else cache,
            cache_position=pos + 1,
        )

        return next_logits, new_snapshot

    @torch.no_grad()
    def encode_sequence(
        self,
        token_ids: List[int],
        initial_snapshot: Optional[RecurrentStateSnapshot] = None,
    ) -> Tuple[torch.Tensor, RecurrentStateSnapshot]:
        """Unroll a token sequence step-by-step and return final logits and state snapshot."""
        state = initial_snapshot if initial_snapshot is not None else self.create_canonical_initial_state()
        last_logits = torch.empty(0, device=self.device)

        for tok in token_ids:
            last_logits, state = self.step(tok, state)

        return last_logits, state

    def generate_greedy(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 10,
        initial_snapshot: Optional[RecurrentStateSnapshot] = None,
    ) -> Tuple[List[int], RecurrentStateSnapshot]:
        """Generate tokens autoregressively using explicit single-token stepping."""
        logits, state = self.encode_sequence(prompt_token_ids, initial_snapshot=initial_snapshot)
        generated_ids: List[int] = []

        for _ in range(max_new_tokens):
            next_token = int(torch.argmax(logits, dim=-1).item())
            generated_ids.append(next_token)
            logits, state = self.step(next_token, state)

        return generated_ids, state
