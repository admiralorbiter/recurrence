"""Sprint S12: Multi-Store Surgical State Swaps & Mediational Propagation Harness.

Executes causal factorial channel interventions at key lag checkpoints (e.g., L=8,
L=W+1=2049, L=2W=4096) and mediational dynamic propagation experiments to establish:
1. Donor-aligned signed graft effect Delta_C and absolute directional displacement P_C.
2. Normalized directional projection alpha_C^logit alongside attribution eligibility counts.
3. True Frobenius-matched Gaussian noise control projected on the real donor axis.
4. Unrelated-donor and permuted-donor specificity controls.
5. Dynamic post-graft and full-turnover KV cache measurements (RGLRU_B -> KV_future_B).
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Set
import torch
import torch.nn.functional as F

from recurrence.interventions.surgical_swaps import swap_stores, add_intervention_matched_noise
from recurrence.loop.latent_impulse_harness import (
    compute_continuation_log_likelihood,
)
from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.state.temporal_inventory import RecurrentStateSnapshot
from recurrence.tasks.impulse_stimuli import (
    ImpulseStimulusPair,
    CANONICAL_STIMULI_PAIRS,
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
    NOISE_CONTROL_RGLRU_SEED1 = "noise_control_rglru_seed1"
    NOISE_CONTROL_RGLRU_SEED2 = "noise_control_rglru_seed2"
    UNRELATED_DONOR_RGLRU = "unrelated_donor_rglru"
    PERMUTED_DONOR_RGLRU = "permuted_donor_rglru"


@dataclass
class SurgicalSwapRecord:
    """Evaluation result for a single surgical intervention condition at lag L."""
    pair_id: str
    regime: str
    lag: int
    condition: str
    target_donor: str  # 'A', 'B', 'unrelated', 'permuted', 'noise', or 'none'
    ll_target_a: float
    ll_target_b: float
    cloze_margin: float
    target_choice: str
    signed_graft_effect: float  # Donor-aligned effect: (m_graft - m_rec) if donor=A, (m_rec - m_graft) if donor=B
    absolute_displacement: float  # P_C = (z_G - z_R) . (z_D - z_R) / ||z_D - z_R||
    donor_recipient_norm: float  # ||z_D - z_R||
    logit_directional_projection: float  # alpha_C^logit = P_C / ||z_D - z_R||
    is_eligible_for_attribution: bool  # |m_donor - m_recipient| >= delta
    causal_attribution_index: Optional[float]  # secondary normalized attribution fraction (unclamped)


def compute_directional_displacement_and_projection(
    z_recipient: torch.Tensor,
    z_donor: torch.Tensor,
    z_graft: torch.Tensor,
) -> Tuple[float, float, float]:
    """Compute (absolute_displacement P_C, projection_fraction alpha_C, norm ||z_D - z_R||)."""
    diff_d = (z_donor - z_recipient).flatten().float()
    diff_g = (z_graft - z_recipient).flatten().float()
    norm_d = float(torch.norm(diff_d).item())
    if norm_d < 1e-6:
        return 0.0, 0.0, 0.0
    unit_d = diff_d / norm_d
    abs_disp = float(torch.sum(diff_g * unit_d).item())
    alpha_proj = float(abs_disp / norm_d)
    return abs_disp, alpha_proj, norm_d


def evaluate_surgical_swaps(
    adapter: RecurrentGemmaAdapter,
    pair: ImpulseStimulusPair,
    regime: str,
    target_lags: List[int],
    seed: int = 42,
    tokenizer: Optional[Any] = None,
    audited_pool: Optional[List[int]] = None,
    unrelated_pair: Optional[ImpulseStimulusPair] = None,
    permuted_pair: Optional[ImpulseStimulusPair] = None,
    eligibility_threshold: float = 0.5,
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

    assert len(event_a_tokens) == len(event_b_tokens), (
        f"Event A ({len(event_a_tokens)}) and Event B ({len(event_b_tokens)}) have mismatched token lengths!"
    )
    assert len(target_a_tokens) == len(target_b_tokens), (
        f"Target A ({len(target_a_tokens)}) and Target B ({len(target_b_tokens)}) have mismatched token lengths!"
    )

    # 2. Tokenize unrelated & permuted control pairs
    if unrelated_pair is None:
        for cp in CANONICAL_STIMULI_PAIRS:
            if cp.pair_id != pair.pair_id:
                unrelated_pair = cp
                break

    if permuted_pair is None:
        # Default to another distinct pair
        idx = next((i for i, cp in enumerate(CANONICAL_STIMULI_PAIRS) if cp.pair_id == pair.pair_id), 0)
        permuted_pair = CANONICAL_STIMULI_PAIRS[(idx + 1) % len(CANONICAL_STIMULI_PAIRS)]

    if tokenizer is not None:
        unrel_event_tokens = tokenizer.encode(unrelated_pair.event_a, add_special_tokens=False)
        perm_event_tokens = tokenizer.encode(permuted_pair.event_a, add_special_tokens=False)
    else:
        unrel_event_tokens = [50, 51, 52]
        perm_event_tokens = [60, 61, 62]

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

    # 3. Unroll prefix
    _, init_state = adapter.encode_sequence(prefix_tokens, step_by_step=False)

    # 4. Unroll Branch A, Branch B, Sham A2, Unrelated Control, and Permuted Control
    logits_a, state_a = adapter.encode_sequence(event_a_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    logits_b, state_b = adapter.encode_sequence(event_b_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    logits_sham, state_sham = adapter.encode_sequence(event_a_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    _, state_unrel = adapter.encode_sequence(unrel_event_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    _, state_perm = adapter.encode_sequence(perm_event_tokens, initial_snapshot=init_state.clone(), step_by_step=False)

    records: List[SurgicalSwapRecord] = []

    prev_lag = 0
    for current_lag in sorted(target_lags):
        if current_lag > prev_lag:
            chunk = filler_tokens[prev_lag:current_lag]
            logits_a, state_a = adapter.encode_sequence(chunk, initial_snapshot=state_a, step_by_step=False)
            logits_b, state_b = adapter.encode_sequence(chunk, initial_snapshot=state_b, step_by_step=False)
            logits_sham, state_sham = adapter.encode_sequence(chunk, initial_snapshot=state_sham, step_by_step=False)
            _, state_unrel = adapter.encode_sequence(chunk, initial_snapshot=state_unrel, step_by_step=False)
            _, state_perm = adapter.encode_sequence(chunk, initial_snapshot=state_perm, step_by_step=False)
            prev_lag = current_lag

        # Evaluate Intact Endpoints
        p_a = state_a.clone()
        z_a, _ = adapter.encode_sequence(cloze_tokens, initial_snapshot=p_a, step_by_step=False)
        ll_a_a = compute_continuation_log_likelihood(adapter, state_a, cloze_tokens, target_a_tokens)
        ll_a_b = compute_continuation_log_likelihood(adapter, state_a, cloze_tokens, target_b_tokens)
        margin_intact_a = ll_a_a - ll_a_b

        p_b = state_b.clone()
        z_b, _ = adapter.encode_sequence(cloze_tokens, initial_snapshot=p_b, step_by_step=False)
        ll_b_a = compute_continuation_log_likelihood(adapter, state_b, cloze_tokens, target_a_tokens)
        ll_b_b = compute_continuation_log_likelihood(adapter, state_b, cloze_tokens, target_b_tokens)
        margin_intact_b = ll_b_a - ll_b_b

        contrast_range = abs(margin_intact_a - margin_intact_b)
        is_eligible = contrast_range >= eligibility_threshold

        # LIVE WHOLE-STATE SWAP VERIFICATION GATE: S^{B <- A}_all must match state_a logits
        whole_a_into_b = swap_stores(state_b, state_a, "all")
        p_whole = whole_a_into_b.clone()
        z_whole, _ = adapter.encode_sequence(cloze_tokens, initial_snapshot=p_whole, step_by_step=False)
        assert torch.allclose(z_whole, z_a, atol=1e-4), "Live whole-state swap failed to reproduce donor logits!"

        # Define genuine intervention-matched noise controls on recipient B pointing along B -> A axis
        noise_state_b_s1 = add_intervention_matched_noise(recipient=state_b, donor=state_a, channel="rglru", seed=seed)
        noise_state_b_s2 = add_intervention_matched_noise(recipient=state_b, donor=state_a, channel="rglru", seed=seed + 1)

        # Define surgical intervention variants
        interventions = [
            (SwapCondition.INTACT_A, state_a.clone(), margin_intact_a, margin_intact_a, z_a, z_a, "A"),
            (SwapCondition.INTACT_B, state_b.clone(), margin_intact_b, margin_intact_b, z_b, z_b, "B"),
            (SwapCondition.WHOLE_SWAP_A_INTO_B, whole_a_into_b, margin_intact_b, margin_intact_a, z_b, z_a, "A"),
            (SwapCondition.WHOLE_SWAP_B_INTO_A, swap_stores(state_a, state_b, "all"), margin_intact_a, margin_intact_b, z_a, z_b, "B"),
            (SwapCondition.RGLRU_ONLY_A_INTO_B, swap_stores(state_b, state_a, "rglru"), margin_intact_b, margin_intact_a, z_b, z_a, "A"),
            (SwapCondition.RGLRU_ONLY_B_INTO_A, swap_stores(state_a, state_b, "rglru"), margin_intact_a, margin_intact_b, z_a, z_b, "B"),
            (SwapCondition.CONV_ONLY_A_INTO_B, swap_stores(state_b, state_a, "conv"), margin_intact_b, margin_intact_a, z_b, z_a, "A"),
            (SwapCondition.CONV_ONLY_B_INTO_A, swap_stores(state_a, state_b, "conv"), margin_intact_a, margin_intact_b, z_a, z_b, "B"),
            (SwapCondition.KV_ONLY_A_INTO_B, swap_stores(state_b, state_a, "kv"), margin_intact_b, margin_intact_a, z_b, z_a, "A"),
            (SwapCondition.KV_ONLY_B_INTO_A, swap_stores(state_a, state_b, "kv"), margin_intact_a, margin_intact_b, z_a, z_b, "B"),
            (SwapCondition.RECURRENT_CORE_A_INTO_B, swap_stores(state_b, state_a, ["rglru", "conv"]), margin_intact_b, margin_intact_a, z_b, z_a, "A"),
            (SwapCondition.SHAM_A2_INTO_A1, swap_stores(state_a, state_sham, "all"), margin_intact_a, margin_intact_a, z_a, z_a, "none"),
            (SwapCondition.NOISE_CONTROL_RGLRU_SEED1, noise_state_b_s1, margin_intact_b, margin_intact_a, z_b, z_a, "noise"),
            (SwapCondition.NOISE_CONTROL_RGLRU_SEED2, noise_state_b_s2, margin_intact_b, margin_intact_a, z_b, z_a, "noise"),
            (SwapCondition.UNRELATED_DONOR_RGLRU, swap_stores(state_b, state_unrel, "rglru"), margin_intact_b, margin_intact_a, z_b, z_a, "unrelated"),
            (SwapCondition.PERMUTED_DONOR_RGLRU, swap_stores(state_b, state_perm, "rglru"), margin_intact_b, margin_intact_a, z_b, z_a, "permuted"),
        ]

        for cond, grafted_state, rec_m, don_m, z_rec, z_don, donor_label in interventions:
            p_graft = grafted_state.clone()
            z_graft, _ = adapter.encode_sequence(cloze_tokens, initial_snapshot=p_graft, step_by_step=False)

            ll_a = compute_continuation_log_likelihood(adapter, grafted_state, cloze_tokens, target_a_tokens)
            ll_b = compute_continuation_log_likelihood(adapter, grafted_state, cloze_tokens, target_b_tokens)
            m = ll_a - ll_b
            choice = "A" if m > 0 else "B"

            # Donor-Aligned Signed Graft Effect Delta_C
            if donor_label == "A":
                signed_delta = m - rec_m
            elif donor_label == "B":
                signed_delta = rec_m - m
            else:
                signed_delta = m - rec_m

            # Absolute Directional Displacement P_C & Projection Fraction alpha_C^logit
            abs_disp, alpha_logit, norm_d = compute_directional_displacement_and_projection(z_rec, z_don, z_graft)

            # Secondary Normalized Cloze Attribution Fraction (unclamped)
            if is_eligible and abs(don_m - rec_m) > 1e-6:
                if donor_label == "A":
                    alpha_cloze = (m - rec_m) / (don_m - rec_m)
                elif donor_label == "B":
                    alpha_cloze = (rec_m - m) / (rec_m - don_m)
                else:
                    alpha_cloze = (m - rec_m) / (don_m - rec_m)
            else:
                alpha_cloze = None

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
                    signed_graft_effect=round(signed_delta, 4),
                    absolute_displacement=round(abs_disp, 4),
                    donor_recipient_norm=round(norm_d, 4),
                    logit_directional_projection=round(alpha_logit, 4),
                    is_eligible_for_attribution=is_eligible,
                    causal_attribution_index=round(alpha_cloze, 4) if alpha_cloze is not None else None,
                )
            )

    return records


def evaluate_mediational_propagation(
    adapter: RecurrentGemmaAdapter,
    pair: ImpulseStimulusPair,
    regime: str = "constant",
    initial_lag: int = 8,
    future_tokens: int = 512,
    seed: int = 42,
    tokenizer: Optional[Any] = None,
    audited_pool: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Causally test if RG-LRU(B) propagates history into subsequently generated post-graft KV representations.

    At initial_lag (e.g. L=8), inject RGLRU_B into Recipient A -> S_0 = (R^B, C^A, K^A).
    Unroll future_tokens of identical filler across S_A, S_B, and S_graft.
    Measure distances strictly over the newly generated post-graft KV slice.
    """
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

    pair_excluded = set(target_a_tokens + target_b_tokens + event_a_tokens + event_b_tokens + prefix_tokens)
    if audited_pool is None:
        audited_pool, _ = build_audited_vocabulary_pool(tokenizer, excluded_token_ids=pair_excluded)

    total_tokens = initial_lag + future_tokens
    filler = get_filler_tokens_for_regime(
        regime=regime,
        length=total_tokens,
        seed=seed,
        audited_pool=audited_pool,
        tokenizer=tokenizer,
        excluded_token_ids=pair_excluded,
    )

    # 1. Unroll prefix
    _, init_state = adapter.encode_sequence(prefix_tokens, step_by_step=False)

    # 2. Unroll events
    _, state_a = adapter.encode_sequence(event_a_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    _, state_b = adapter.encode_sequence(event_b_tokens, initial_snapshot=init_state.clone(), step_by_step=False)

    # 3. Unroll to initial_lag
    init_chunk = filler[:initial_lag]
    _, state_a = adapter.encode_sequence(init_chunk, initial_snapshot=state_a, step_by_step=False)
    _, state_b = adapter.encode_sequence(init_chunk, initial_snapshot=state_b, step_by_step=False)

    # 4. Create grafted mediational state at initial_lag: (R^B, C^A, K^A)
    state_med = swap_stores(recipient=state_a, donor=state_b, channels="rglru")

    # 5. Unroll future_tokens on all 3 branches
    future_chunk = filler[initial_lag:total_tokens]
    _, end_state_a = adapter.encode_sequence(future_chunk, initial_snapshot=state_a.clone(), step_by_step=False)
    _, end_state_b = adapter.encode_sequence(future_chunk, initial_snapshot=state_b.clone(), step_by_step=False)
    _, end_state_med = adapter.encode_sequence(future_chunk, initial_snapshot=state_med.clone(), step_by_step=False)

    # 6. Measure Key distances strictly on POST-GRAFT generated tokens
    d_post_med_to_a = 0.0
    d_post_med_to_b = 0.0
    d_post_a_to_b = 0.0

    d_full_med_to_a = 0.0
    d_full_med_to_b = 0.0
    d_full_a_to_b = 0.0

    n_layers = len(end_state_a.kv)
    num_new = len(future_chunk)

    for l in end_state_a.kv:
        k_a = end_state_a.kv[l]["key"].float()
        k_b = end_state_b.kv[l]["key"].float()
        k_med = end_state_med.kv[l]["key"].float()

        # Full cache distance
        d_full_med_to_a += float(torch.norm(k_med - k_a) / (torch.norm(k_a) + 1e-8))
        d_full_med_to_b += float(torch.norm(k_med - k_b) / (torch.norm(k_b) + 1e-8))
        d_full_a_to_b += float(torch.norm(k_a - k_b) / (torch.norm(k_b) + 1e-8))

        # Sliced post-graft distance (last num_new tokens)
        k_a_post = k_a[..., -num_new:, :] if k_a.shape[-2] >= num_new else k_a
        k_b_post = k_b[..., -num_new:, :] if k_b.shape[-2] >= num_new else k_b
        k_med_post = k_med[..., -num_new:, :] if k_med.shape[-2] >= num_new else k_med

        d_post_med_to_a += float(torch.norm(k_med_post - k_a_post) / (torch.norm(k_a_post) + 1e-8))
        d_post_med_to_b += float(torch.norm(k_med_post - k_b_post) / (torch.norm(k_b_post) + 1e-8))
        d_post_a_to_b += float(torch.norm(k_a_post - k_b_post) / (torch.norm(k_b_post) + 1e-8))

    d_full_med_to_a /= max(n_layers, 1)
    d_full_med_to_b /= max(n_layers, 1)
    d_full_a_to_b /= max(n_layers, 1)

    d_post_med_to_a /= max(n_layers, 1)
    d_post_med_to_b /= max(n_layers, 1)
    d_post_a_to_b /= max(n_layers, 1)

    # Migration indices: positive means grafted KV migrated toward donor B
    post_migration_index = (d_post_med_to_a - d_post_med_to_b) / (d_post_a_to_b + 1e-8) if d_post_a_to_b > 1e-6 else 0.0
    full_migration_index = (d_full_med_to_a - d_full_med_to_b) / (d_full_a_to_b + 1e-8) if d_full_a_to_b > 1e-6 else 0.0

    return {
        "pair_id": pair.pair_id,
        "regime": regime,
        "initial_lag": initial_lag,
        "future_tokens": future_tokens,
        "is_full_window_turnover": bool(future_tokens >= 2048),
        "d_post_med_to_a": round(d_post_med_to_a, 4),
        "d_post_med_to_b": round(d_post_med_to_b, 4),
        "d_post_a_to_b": round(d_post_a_to_b, 4),
        "post_migration_index": round(post_migration_index, 4),
        "d_full_med_to_a": round(d_full_med_to_a, 4),
        "d_full_med_to_b": round(d_full_med_to_b, 4),
        "d_full_a_to_b": round(d_full_a_to_b, 4),
        "full_migration_index": round(full_migration_index, 4),
    }
