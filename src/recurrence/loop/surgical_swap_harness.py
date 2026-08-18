"""Sprint S12: Multi-Store Surgical State Swaps & Mediational Propagation Harness.

Executes fully symmetric causal factorial channel interventions at key lag checkpoints (e.g., L=8,
L=W+1=2049, L=2W=4096) and mediational dynamic propagation experiments to establish:
1. Symmetrized donor-aligned signed graft effect Delta_C and directional displacement P_C.
2. Normalized directional projection alpha_C^logit alongside attribution eligibility counts.
3. True Frobenius-matched Gaussian noise control projected on the real donor axis.
4. Balanced cyclic derangement cross-pair controls (Unrelated Shift +1, Permuted Shift +7).
5. Dynamic post-graft and full-turnover KV cache measurements (RGLRU_B -> KV_future_B).
"""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
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
    # Endpoints
    INTACT_A = "intact_a"
    INTACT_B = "intact_b"
    # Whole State Swaps
    WHOLE_SWAP_A_INTO_B = "whole_swap_a_into_b"
    WHOLE_SWAP_B_INTO_A = "whole_swap_b_into_a"
    # Single Store Swaps (Matching Donor)
    RGLRU_ONLY_A_INTO_B = "rglru_only_a_into_b"
    RGLRU_ONLY_B_INTO_A = "rglru_only_b_into_a"
    CONV_ONLY_A_INTO_B = "conv_only_a_into_b"
    CONV_ONLY_B_INTO_A = "conv_only_b_into_a"
    KV_ONLY_A_INTO_B = "kv_only_a_into_b"
    KV_ONLY_B_INTO_A = "kv_only_b_into_a"
    # Multi-Store Composite
    RECURRENT_CORE_A_INTO_B = "recurrent_core_a_into_b"
    RECURRENT_CORE_B_INTO_A = "recurrent_core_b_into_a"
    # Sham Controls
    SHAM_A2_INTO_A1 = "sham_a2_into_a1"
    SHAM_B2_INTO_B1 = "sham_b2_into_b1"
    # Intervention-Matched Frobenius Noise Controls (Seed 1 & Seed 2)
    NOISE_RGLRU_A_INTO_B_S1 = "noise_rglru_a_into_b_s1"
    NOISE_RGLRU_B_INTO_A_S1 = "noise_rglru_b_into_a_s1"
    NOISE_RGLRU_A_INTO_B_S2 = "noise_rglru_a_into_b_s2"
    NOISE_RGLRU_B_INTO_A_S2 = "noise_rglru_b_into_a_s2"
    # Balanced Cyclic Derangement 1 (Primary Unrelated Donor: Shift +1)
    UNRELATED_RGLRU_A_INTO_B = "unrelated_rglru_a_into_b"
    UNRELATED_RGLRU_B_INTO_A = "unrelated_rglru_b_into_a"
    # Balanced Cyclic Derangement 2 (Secondary Permuted Donor: Shift +7)
    PERMUTED_RGLRU_A_INTO_B = "permuted_rglru_a_into_b"
    PERMUTED_RGLRU_B_INTO_A = "permuted_rglru_b_into_a"


@dataclass
class SurgicalSwapRecord:
    """Evaluation result for a single surgical intervention condition at lag L."""
    pair_id: str
    regime: str
    lag: int
    condition: str
    target_donor: str  # 'A', 'B', 'unrelated_A', 'unrelated_B', 'permuted_A', 'permuted_B', 'noise_A', 'noise_B', 'none'
    ll_target_a: float
    ll_target_b: float
    cloze_margin: float
    target_choice: str
    signed_graft_effect: float  # Donor-aligned effect: (m_graft - m_rec) if donor=A, (m_rec - m_graft) if donor=B
    directional_displacement: float  # P_C = (z_G - z_R) . (z_D - z_R) / ||z_D - z_R||
    donor_recipient_norm: float  # ||z_D - z_R||
    logit_directional_projection: float  # alpha_C^logit = P_C / ||z_D - z_R||
    is_eligible_for_attribution: bool  # |m_donor - m_recipient| >= delta
    causal_attribution_index: Optional[float]  # secondary normalized attribution fraction (unclamped)


def compute_directional_displacement_and_projection(
    z_recipient: torch.Tensor,
    z_donor: torch.Tensor,
    z_graft: torch.Tensor,
) -> Tuple[float, float, float]:
    """Compute (directional_displacement P_C, projection_fraction alpha_C, norm ||z_D - z_R||)."""
    diff_d = (z_donor - z_recipient).flatten().float()
    diff_g = (z_graft - z_recipient).flatten().float()
    norm_d = float(torch.norm(diff_d).item())
    if norm_d < 1e-6:
        return 0.0, 0.0, 0.0
    unit_d = diff_d / norm_d
    dir_disp = float(torch.sum(diff_g * unit_d).item())
    alpha_proj = float(dir_disp / norm_d)
    return dir_disp, alpha_proj, norm_d


def get_balanced_donor_pairs(
    all_pairs: List[ImpulseStimulusPair],
    target_pair: ImpulseStimulusPair,
) -> Tuple[ImpulseStimulusPair, ImpulseStimulusPair]:
    """Get balanced cyclic derangements for unrelated (+1) and permuted (+7) cross-pair donors."""
    idx = next((i for i, p in enumerate(all_pairs) if p.pair_id == target_pair.pair_id), 0)
    n = len(all_pairs)
    unrelated_idx = (idx + 1) % n
    permuted_idx = (idx + 7) % n if n > 1 else 0
    return all_pairs[unrelated_idx], all_pairs[permuted_idx]


def evaluate_surgical_swaps(
    adapter: RecurrentGemmaAdapter,
    pair: ImpulseStimulusPair,
    regime: str,
    target_lags: List[int],
    seed: int = 42,
    tokenizer: Optional[Any] = None,
    audited_pool: Optional[List[int]] = None,
    all_pairs: Optional[List[ImpulseStimulusPair]] = None,
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

    # 2. Balanced cyclic derangement pairs
    pair_panel = all_pairs if all_pairs is not None else CANONICAL_STIMULI_PAIRS
    unrelated_pair, permuted_pair = get_balanced_donor_pairs(pair_panel, pair)

    if tokenizer is not None:
        unrel_a_tokens = tokenizer.encode(unrelated_pair.event_a, add_special_tokens=False)
        unrel_b_tokens = tokenizer.encode(unrelated_pair.event_b, add_special_tokens=False)
        perm_a_tokens = tokenizer.encode(permuted_pair.event_a, add_special_tokens=False)
        perm_b_tokens = tokenizer.encode(permuted_pair.event_b, add_special_tokens=False)
    else:
        unrel_a_tokens = [50, 51, 52]
        unrel_b_tokens = [53, 54, 55]
        perm_a_tokens = [60, 61, 62]
        perm_b_tokens = [63, 64, 65]

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

    # 4. Unroll Intact Branches, Sham Controls, and Cross-Pair Controls
    _, state_a = adapter.encode_sequence(event_a_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    _, state_b = adapter.encode_sequence(event_b_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    _, state_sham_a = adapter.encode_sequence(event_a_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    _, state_sham_b = adapter.encode_sequence(event_b_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    _, state_unrel_a = adapter.encode_sequence(unrel_a_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    _, state_unrel_b = adapter.encode_sequence(unrel_b_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    _, state_perm_a = adapter.encode_sequence(perm_a_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    _, state_perm_b = adapter.encode_sequence(perm_b_tokens, initial_snapshot=init_state.clone(), step_by_step=False)

    records: List[SurgicalSwapRecord] = []

    prev_lag = 0
    for current_lag in sorted(target_lags):
        if current_lag > prev_lag:
            chunk = filler_tokens[prev_lag:current_lag]
            _, state_a = adapter.encode_sequence(chunk, initial_snapshot=state_a, step_by_step=False)
            _, state_b = adapter.encode_sequence(chunk, initial_snapshot=state_b, step_by_step=False)
            _, state_sham_a = adapter.encode_sequence(chunk, initial_snapshot=state_sham_a, step_by_step=False)
            _, state_sham_b = adapter.encode_sequence(chunk, initial_snapshot=state_sham_b, step_by_step=False)
            _, state_unrel_a = adapter.encode_sequence(chunk, initial_snapshot=state_unrel_a, step_by_step=False)
            _, state_unrel_b = adapter.encode_sequence(chunk, initial_snapshot=state_unrel_b, step_by_step=False)
            _, state_perm_a = adapter.encode_sequence(chunk, initial_snapshot=state_perm_a, step_by_step=False)
            _, state_perm_b = adapter.encode_sequence(chunk, initial_snapshot=state_perm_b, step_by_step=False)
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
        p_whole_ab = whole_a_into_b.clone()
        z_whole_ab, _ = adapter.encode_sequence(cloze_tokens, initial_snapshot=p_whole_ab, step_by_step=False)
        assert torch.allclose(z_whole_ab, z_a, atol=1e-4), "Live whole-state swap failed to reproduce donor logits!"

        whole_b_into_a = swap_stores(state_a, state_b, "all")
        p_whole_ba = whole_b_into_a.clone()
        z_whole_ba, _ = adapter.encode_sequence(cloze_tokens, initial_snapshot=p_whole_ba, step_by_step=False)
        assert torch.allclose(z_whole_ba, z_b, atol=1e-4), "Live whole-state swap failed to reproduce donor logits!"

        # Define genuine intervention-matched noise controls for both directions
        noise_b_s1 = add_intervention_matched_noise(recipient=state_b, donor=state_a, channel="rglru", seed=seed)
        noise_a_s1 = add_intervention_matched_noise(recipient=state_a, donor=state_b, channel="rglru", seed=seed)
        noise_b_s2 = add_intervention_matched_noise(recipient=state_b, donor=state_a, channel="rglru", seed=seed + 1)
        noise_a_s2 = add_intervention_matched_noise(recipient=state_a, donor=state_b, channel="rglru", seed=seed + 1)

        # Define symmetric surgical intervention variants
        # Tuple: (condition, grafted_state, recipient_margin, donor_margin, recipient_z, donor_z, donor_label)
        interventions = [
            (SwapCondition.INTACT_A, state_a.clone(), margin_intact_a, margin_intact_a, z_a, z_a, "A"),
            (SwapCondition.INTACT_B, state_b.clone(), margin_intact_b, margin_intact_b, z_b, z_b, "B"),
            (SwapCondition.WHOLE_SWAP_A_INTO_B, whole_a_into_b, margin_intact_b, margin_intact_a, z_b, z_a, "A"),
            (SwapCondition.WHOLE_SWAP_B_INTO_A, whole_b_into_a, margin_intact_a, margin_intact_b, z_a, z_b, "B"),
            (SwapCondition.RGLRU_ONLY_A_INTO_B, swap_stores(state_b, state_a, "rglru"), margin_intact_b, margin_intact_a, z_b, z_a, "A"),
            (SwapCondition.RGLRU_ONLY_B_INTO_A, swap_stores(state_a, state_b, "rglru"), margin_intact_a, margin_intact_b, z_a, z_b, "B"),
            (SwapCondition.CONV_ONLY_A_INTO_B, swap_stores(state_b, state_a, "conv"), margin_intact_b, margin_intact_a, z_b, z_a, "A"),
            (SwapCondition.CONV_ONLY_B_INTO_A, swap_stores(state_a, state_b, "conv"), margin_intact_a, margin_intact_b, z_a, z_b, "B"),
            (SwapCondition.KV_ONLY_A_INTO_B, swap_stores(state_b, state_a, "kv"), margin_intact_b, margin_intact_a, z_b, z_a, "A"),
            (SwapCondition.KV_ONLY_B_INTO_A, swap_stores(state_a, state_b, "kv"), margin_intact_a, margin_intact_b, z_a, z_b, "B"),
            (SwapCondition.RECURRENT_CORE_A_INTO_B, swap_stores(state_b, state_a, ["rglru", "conv"]), margin_intact_b, margin_intact_a, z_b, z_a, "A"),
            (SwapCondition.RECURRENT_CORE_B_INTO_A, swap_stores(state_a, state_b, ["rglru", "conv"]), margin_intact_a, margin_intact_b, z_a, z_b, "B"),
            (SwapCondition.SHAM_A2_INTO_A1, swap_stores(state_a, state_sham_a, "all"), margin_intact_a, margin_intact_a, z_a, z_a, "none"),
            (SwapCondition.SHAM_B2_INTO_B1, swap_stores(state_b, state_sham_b, "all"), margin_intact_b, margin_intact_b, z_b, z_b, "none"),
            (SwapCondition.NOISE_RGLRU_A_INTO_B_S1, noise_b_s1, margin_intact_b, margin_intact_a, z_b, z_a, "noise_A"),
            (SwapCondition.NOISE_RGLRU_B_INTO_A_S1, noise_a_s1, margin_intact_a, margin_intact_b, z_a, z_b, "noise_B"),
            (SwapCondition.NOISE_RGLRU_A_INTO_B_S2, noise_b_s2, margin_intact_b, margin_intact_a, z_b, z_a, "noise_A"),
            (SwapCondition.NOISE_RGLRU_B_INTO_A_S2, noise_a_s2, margin_intact_a, margin_intact_b, z_a, z_b, "noise_B"),
            (SwapCondition.UNRELATED_RGLRU_A_INTO_B, swap_stores(state_b, state_unrel_a, "rglru"), margin_intact_b, margin_intact_a, z_b, z_a, "unrelated_A"),
            (SwapCondition.UNRELATED_RGLRU_B_INTO_A, swap_stores(state_a, state_unrel_b, "rglru"), margin_intact_a, margin_intact_b, z_a, z_b, "unrelated_B"),
            (SwapCondition.PERMUTED_RGLRU_A_INTO_B, swap_stores(state_b, state_perm_a, "rglru"), margin_intact_b, margin_intact_a, z_b, z_a, "permuted_A"),
            (SwapCondition.PERMUTED_RGLRU_B_INTO_A, swap_stores(state_a, state_perm_b, "rglru"), margin_intact_a, margin_intact_b, z_a, z_b, "permuted_B"),
        ]

        for cond, grafted_state, rec_m, don_m, z_rec, z_don, donor_label in interventions:
            p_graft = grafted_state.clone()
            z_graft, _ = adapter.encode_sequence(cloze_tokens, initial_snapshot=p_graft, step_by_step=False)

            ll_a = compute_continuation_log_likelihood(adapter, grafted_state, cloze_tokens, target_a_tokens)
            ll_b = compute_continuation_log_likelihood(adapter, grafted_state, cloze_tokens, target_b_tokens)
            m = ll_a - ll_b
            choice = "A" if m > 0 else "B"

            # Donor-Aligned Signed Graft Effect Delta_C
            if donor_label in ("A", "unrelated_A", "permuted_A", "noise_A"):
                signed_delta = m - rec_m
            elif donor_label in ("B", "unrelated_B", "permuted_B", "noise_B"):
                signed_delta = rec_m - m
            else:
                signed_delta = m - rec_m

            # Directional Displacement P_C & Projection Fraction alpha_C^logit
            dir_disp, alpha_logit, norm_d = compute_directional_displacement_and_projection(z_rec, z_don, z_graft)

            # Secondary Normalized Cloze Attribution Fraction (unclamped)
            if is_eligible and abs(don_m - rec_m) > 1e-6:
                if donor_label in ("A", "unrelated_A", "permuted_A", "noise_A"):
                    alpha_cloze = (m - rec_m) / (don_m - rec_m)
                elif donor_label in ("B", "unrelated_B", "permuted_B", "noise_B"):
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
                    directional_displacement=round(dir_disp, 4),
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
    """Causally test if RG-LRU(B) propagates history into subsequently generated post-graft KV representations."""
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
