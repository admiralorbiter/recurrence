"""Sprint S12: Multi-Store Surgical State Swaps Evaluation Harness.

Executes causal factorial channel interventions at key lag checkpoints
(e.g., L=8, L=W+1=2049, L=2W=4096) to establish causal store attribution:
evaluating whether transplanted RGLRU, Conv, or KV channels flip the model's
behavioral cloze retrieval choice.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Set
import torch

from recurrence.interventions.surgical_swaps import swap_stores
from recurrence.loop.latent_impulse_harness import (
    compute_continuation_log_likelihood,
)
from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.state.temporal_inventory import RecurrentStateSnapshot
from recurrence.tasks.impulse_stimuli import (
    ImpulseStimulusPair,
    build_audited_vocabulary_pool,
    get_filler_tokens_for_regime,
)


class SwapCondition(str, Enum):
    INTACT_A = "intact_a"
    INTACT_B = "intact_b"
    WHOLE_SWAP_A_INTO_B = "whole_swap_a_into_b"
    WHOLE_SWAP_B_INTO_A = "whole_swap_b_into_a"
    RGLRU_ONLY_A_INTO_B = "rglru_only_a_into_b"
    RGLRU_ONLY_B_INTO_A = "rglru_only_b_into_a"
    CONV_ONLY_A_INTO_B = "conv_only_a_into_b"
    CONV_ONLY_B_INTO_A = "conv_only_b_into_a"
    KV_ONLY_A_INTO_B = "kv_only_a_into_b"
    KV_ONLY_B_INTO_A = "kv_only_b_into_a"
    RECURRENT_CORE_A_INTO_B = "recurrent_core_a_into_b"
    SHAM_A2_INTO_A1 = "sham_a2_into_a1"


@dataclass
class SurgicalSwapRecord:
    """Evaluation result for a single surgical intervention condition at lag L."""
    pair_id: str
    regime: str
    lag: int
    condition: str
    target_donor: str  # 'A', 'B', or 'none'
    ll_target_a: float
    ll_target_b: float
    cloze_margin: float
    target_choice: str
    causal_attribution_index: float  # alpha in [0, 1] relative to intact endpoints


def evaluate_surgical_swaps(
    adapter: RecurrentGemmaAdapter,
    pair: ImpulseStimulusPair,
    regime: str,
    target_lags: List[int],
    seed: int = 42,
    tokenizer: Optional[Any] = None,
    audited_pool: Optional[List[int]] = None,
) -> List[SurgicalSwapRecord]:
    """Evaluate full causal surgical swap factorial across specified lag checkpoints."""
    # 1. Tokenize inputs
    if tokenizer is not None:
        prefix_tokens = tokenizer.encode(pair.prefix, add_special_tokens=False)
        event_a_tokens = tokenizer.encode(pair.event_a, add_special_tokens=False)
        event_b_tokens = tokenizer.encode(pair.event_b, add_special_tokens=False)
        cloze_tokens = tokenizer.encode(pair.cloze_prompt, add_special_tokens=False)
        target_a_tokens = tokenizer.encode(" " + pair.target_a.strip(), add_special_tokens=False)
        target_b_tokens = tokenizer.encode(" " + pair.target_b.strip(), add_special_tokens=False)
    else:
        prefix_tokens = [10, 11]
        event_a_tokens = [20, 21, 22]
        event_b_tokens = [30, 31, 32]
        cloze_tokens = [40, 41]
        target_a_tokens = [22]
        target_b_tokens = [32]

    max_lag = max(target_lags)
    pair_excluded = set(target_a_tokens + target_b_tokens + event_a_tokens + event_b_tokens + prefix_tokens)
    if audited_pool is None:
        audited_pool, _ = build_audited_vocabulary_pool(tokenizer, excluded_token_ids=pair_excluded)

    filler_tokens = get_filler_tokens_for_regime(
        regime=regime,
        length=max_lag,
        seed=seed,
        audited_pool=audited_pool,
        tokenizer=tokenizer,
        excluded_token_ids=pair_excluded,
    )

    # 2. Unroll prefix
    _, init_state = adapter.encode_sequence(prefix_tokens, step_by_step=False)

    # 3. Unroll Branch A, Branch B, and Sham A2
    logits_a, state_a = adapter.encode_sequence(event_a_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    logits_b, state_b = adapter.encode_sequence(event_b_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    logits_sham, state_sham = adapter.encode_sequence(event_a_tokens, initial_snapshot=init_state.clone(), step_by_step=False)

    records: List[SurgicalSwapRecord] = []

    prev_lag = 0
    for current_lag in sorted(target_lags):
        if current_lag > prev_lag:
            chunk = filler_tokens[prev_lag:current_lag]
            logits_a, state_a = adapter.encode_sequence(chunk, initial_snapshot=state_a, step_by_step=False)
            logits_b, state_b = adapter.encode_sequence(chunk, initial_snapshot=state_b, step_by_step=False)
            logits_sham, state_sham = adapter.encode_sequence(chunk, initial_snapshot=state_sham, step_by_step=False)
            prev_lag = current_lag

        # Evaluate Intact Endpoints
        ll_a_a = compute_continuation_log_likelihood(adapter, state_a, cloze_tokens, target_a_tokens)
        ll_a_b = compute_continuation_log_likelihood(adapter, state_a, cloze_tokens, target_b_tokens)
        margin_intact_a = ll_a_a - ll_a_b

        ll_b_a = compute_continuation_log_likelihood(adapter, state_b, cloze_tokens, target_a_tokens)
        ll_b_b = compute_continuation_log_likelihood(adapter, state_b, cloze_tokens, target_b_tokens)
        margin_intact_b = ll_b_a - ll_b_b

        total_dynamic_range = max(margin_intact_a - margin_intact_b, 1e-6)

        # Define surgical intervention variants
        interventions = [
            (SwapCondition.INTACT_A, state_a.clone(), "A", 1.0),
            (SwapCondition.INTACT_B, state_b.clone(), "B", 0.0),
            (SwapCondition.WHOLE_SWAP_A_INTO_B, swap_stores(state_b, state_a, "all"), "A", None),
            (SwapCondition.WHOLE_SWAP_B_INTO_A, swap_stores(state_a, state_b, "all"), "B", None),
            (SwapCondition.RGLRU_ONLY_A_INTO_B, swap_stores(state_b, state_a, "rglru"), "A", None),
            (SwapCondition.RGLRU_ONLY_B_INTO_A, swap_stores(state_a, state_b, "rglru"), "B", None),
            (SwapCondition.CONV_ONLY_A_INTO_B, swap_stores(state_b, state_a, "conv"), "A", None),
            (SwapCondition.CONV_ONLY_B_INTO_A, swap_stores(state_a, state_b, "conv"), "B", None),
            (SwapCondition.KV_ONLY_A_INTO_B, swap_stores(state_b, state_a, "kv"), "A", None),
            (SwapCondition.KV_ONLY_B_INTO_A, swap_stores(state_a, state_b, "kv"), "B", None),
            (SwapCondition.RECURRENT_CORE_A_INTO_B, swap_stores(state_b, state_a, ["rglru", "conv"]), "A", None),
            (SwapCondition.SHAM_A2_INTO_A1, swap_stores(state_a, state_sham, "all"), "none", None),
        ]

        for cond, grafted_state, donor_label, fixed_alpha in interventions:
            ll_a = compute_continuation_log_likelihood(adapter, grafted_state, cloze_tokens, target_a_tokens)
            ll_b = compute_continuation_log_likelihood(adapter, grafted_state, cloze_tokens, target_b_tokens)
            m = ll_a - ll_b
            choice = "A" if m > 0 else "B"

            if fixed_alpha is not None:
                alpha = fixed_alpha
            else:
                alpha = float((m - margin_intact_b) / total_dynamic_range)

            records.append(
                SurgicalSwapRecord(
                    pair_id=pair.pair_id,
                    regime=regime,
                    lag=current_lag,
                    condition=cond.value,
                    target_donor=donor_label,
                    ll_target_a=round(ll_a, 4),
                    ll_target_b=round(ll_b, 4),
                    cloze_margin=round(m, 4),
                    target_choice=choice,
                    causal_attribution_index=round(alpha, 4),
                )
            )

    return records
