"""Latent Impulse Response & Store Localization Harness (Sprint S11 Hardened).

Evaluates physical persistence, layer-level spatial store traces, behavioral divergence,
and task usability across RG-LRU, Conv1D, and KV cache without off-manifold injections.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import RecurrentGemmaConfig

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.state.temporal_inventory import RecurrentStateSnapshot
from recurrence.tasks.impulse_stimuli import (
    ImpulseStimulusPair,
    build_audited_vocabulary_pool,
    get_filler_tokens_for_regime,
)


def generate_dynamic_lag_grid(config: RecurrentGemmaConfig) -> List[int]:
    """Generate dynamic lag steps aligned with model's physical architectural boundaries."""
    conv1d_width = getattr(config, "conv1d_width", 4)
    # Check official RecurrentGemmaConfig field attention_window_size first
    sliding_window = getattr(config, "attention_window_size", getattr(config, "sliding_window", 2048))

    raw_lags = [
        0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512,
        sliding_window // 2,
        sliding_window - 8,
        sliding_window - 1,
        sliding_window,
        sliding_window + 1,
        2 * sliding_window,
    ]
    # Filter valid non-negative unique sorted lags
    lags = sorted(list({int(l) for l in raw_lags if l >= 0}))
    return lags


def compute_rmsdiff(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """Compute Root Mean Square Difference between two tensors."""
    if t1.numel() == 0 or t2.numel() == 0 or t1.shape != t2.shape:
        return 0.0
    diff = t1.float() - t2.float()
    return float(torch.sqrt(torch.mean(diff ** 2)).item())


def compute_scale_relative_dist(t1: torch.Tensor, t2: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute scale-relative distance D_rel = ||t1 - t2||_2 / (sqrt(||t1||_2^2 + ||t2||_2^2) + eps)."""
    if t1.numel() == 0 or t2.numel() == 0 or t1.shape != t2.shape:
        return 0.0
    norm_diff = torch.norm(t1.float() - t2.float(), p=2)
    norm_sum = torch.sqrt(torch.norm(t1.float(), p=2) ** 2 + torch.norm(t2.float(), p=2) ** 2)
    return float((norm_diff / (norm_sum + eps)).item())


def compute_cossim(t1: torch.Tensor, t2: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute Cosine Similarity between flattened tensors."""
    if t1.numel() == 0 or t2.numel() == 0 or t1.shape != t2.shape:
        return 1.0
    f1 = t1.flatten().float()
    f2 = t2.flatten().float()
    sim = F.cosine_similarity(f1, f2, dim=0, eps=eps)
    return float(sim.item())


def compute_jensen_shannon_div(logits1: torch.Tensor, logits2: torch.Tensor) -> float:
    """Compute bounded symmetric Jensen-Shannon divergence D_JS(P1 || P2) in nats."""
    p1 = F.softmax(logits1.float(), dim=-1)
    p2 = F.softmax(logits2.float(), dim=-1)
    m = 0.5 * (p1 + p2)

    kl1 = F.kl_div(torch.log(m + 1e-12), p1, reduction="batchmean")
    kl2 = F.kl_div(torch.log(m + 1e-12), p2, reduction="batchmean")
    return float((0.5 * (kl1 + kl2)).item())


@dataclass
class StoreMetrics:
    """Detailed layer-level physical divergence metrics."""
    rmsdiff: float = 0.0
    scale_relative_dist: float = 0.0
    cossim: float = 1.0
    frobenius: float = 0.0
    retention_ratio: float = 1.0
    # Additional separate K / V and recent-entry metrics (for KV channel)
    key_d_rel: float = 0.0
    val_d_rel: float = 0.0
    recent_kv_d_rel: float = 0.0


@dataclass
class LayerTraceRecord:
    """Per-layer, per-store metric record for layer x lag anatomy."""
    pair_id: str
    regime: str
    lag: int
    channel: str  # 'rglru', 'conv', 'kv', 'k', 'v'
    layer_idx: int
    rmsdiff: float
    scale_relative_dist: float
    cossim: float
    frobenius: float
    retention_ratio: float
    key_d_rel: float = 0.0
    val_d_rel: float = 0.0
    recent_kv_d_rel: float = 0.0


@dataclass
class LagCheckpointRecord:
    """Recorded summary metrics at a single lag step L."""
    lag: int
    conv_directly_resident: bool
    kv_directly_resident: bool
    rglru_layers: Dict[int, StoreMetrics] = field(default_factory=dict)
    conv_layers: Dict[int, StoreMetrics] = field(default_factory=dict)
    kv_layers: Dict[int, StoreMetrics] = field(default_factory=dict)
    mean_rglru_d_rel: float = 0.0
    mean_conv_d_rel: float = 0.0
    mean_kv_d_rel: float = 0.0
    mean_k_d_rel: float = 0.0
    mean_v_d_rel: float = 0.0
    mean_recent_kv_d_rel: float = 0.0
    mean_rglru_retention: float = 1.0
    mean_conv_retention: float = 1.0
    mean_kv_retention: float = 1.0
    mean_k_retention: float = 1.0
    mean_v_retention: float = 1.0
    jensen_shannon_div: float = 0.0
    top1_disagreement: bool = False
    twoway_2afc_margin: float = 0.0
    twoway_2afc_accuracy: float = 0.0
    sham_mean_d_rel: float = 0.0
    sham_jensen_shannon_div: float = 0.0


def compute_continuation_log_likelihood(
    adapter: RecurrentGemmaAdapter,
    base_snapshot: RecurrentStateSnapshot,
    cloze_tokens: List[int],
    target_tokens: List[int],
) -> float:
    """Compute exact log P(target_tokens | cloze_tokens, base_snapshot) using multi-token unroll."""
    if not target_tokens:
        return 0.0

    probe = base_snapshot.clone()
    logits, probe = adapter.encode_sequence(cloze_tokens, initial_snapshot=probe, step_by_step=False)

    total_log_prob = 0.0
    for t_idx, tok_id in enumerate(target_tokens):
        log_probs = torch.log_softmax(logits, dim=-1)
        if tok_id < log_probs.shape[-1]:
            tok_log_prob = log_probs[0, tok_id].item()
        else:
            tok_log_prob = -100.0
        total_log_prob += tok_log_prob

        if t_idx < len(target_tokens) - 1:
            logits, probe = adapter.step(tok_id, probe)

    return float(total_log_prob / max(len(target_tokens), 1))


def evaluate_cloze_retrieval(
    adapter: RecurrentGemmaAdapter,
    snapshot_a: RecurrentStateSnapshot,
    snapshot_b: RecurrentStateSnapshot,
    cloze_tokens: List[int],
    target_a_tokens: List[int],
    target_b_tokens: List[int],
) -> Tuple[float, float]:
    """Evaluate 2AFC cloze retrieval log-likelihood margin and accuracy from cloned snapshots."""
    # 1. Probe Branch A (detached clone)
    ll_a_target_a = compute_continuation_log_likelihood(adapter, snapshot_a, cloze_tokens, target_a_tokens)
    ll_a_target_b = compute_continuation_log_likelihood(adapter, snapshot_a, cloze_tokens, target_b_tokens)
    margin_a = ll_a_target_a - ll_a_target_b

    # 2. Probe Branch B (detached clone)
    ll_b_target_b = compute_continuation_log_likelihood(adapter, snapshot_b, cloze_tokens, target_b_tokens)
    ll_b_target_a = compute_continuation_log_likelihood(adapter, snapshot_b, cloze_tokens, target_a_tokens)
    margin_b = ll_b_target_b - ll_b_target_a

    mean_margin = (margin_a + margin_b) / 2.0
    accuracy = 1.0 if (margin_a > 0 and margin_b > 0) else (0.5 if (margin_a > 0 or margin_b > 0) else 0.0)
    return float(mean_margin), float(accuracy)


def evaluate_impulse_trajectory(
    adapter: RecurrentGemmaAdapter,
    pair: ImpulseStimulusPair,
    regime: str,
    lag_grid: List[int],
    seed: int = 42,
    tokenizer: Optional[Any] = None,
    audited_pool: Optional[List[int]] = None,
) -> Tuple[List[LagCheckpointRecord], List[LayerTraceRecord]]:
    """Execute matched trajectory impulse response and return summary and per-layer records."""
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

    max_lag = max(lag_grid)
    # Pair-disjoint token exclusions
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

    conv1d_width = getattr(adapter.config, "conv1d_width", 4)
    sliding_window = getattr(adapter.config, "attention_window_size", getattr(adapter.config, "sliding_window", 2048))

    # 2. Unroll prefix
    _, init_state = adapter.encode_sequence(prefix_tokens, step_by_step=False)

    # 3. Unroll Branch A, Branch B, and Sham A2
    logits_a, state_a = adapter.encode_sequence(event_a_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    logits_b, state_b = adapter.encode_sequence(event_b_tokens, initial_snapshot=init_state.clone(), step_by_step=False)
    logits_sham, state_sham = adapter.encode_sequence(event_a_tokens, initial_snapshot=init_state.clone(), step_by_step=False)

    records: List[LagCheckpointRecord] = []
    layer_records: List[LayerTraceRecord] = []

    initial_d_rel_rglru: Dict[int, float] = {}
    initial_d_rel_conv: Dict[int, float] = {}
    initial_d_rel_kv: Dict[int, float] = {}
    initial_d_rel_k: Dict[int, float] = {}
    initial_d_rel_v: Dict[int, float] = {}

    prev_lag = 0
    for current_lag in sorted(lag_grid):
        if current_lag > prev_lag:
            chunk = filler_tokens[prev_lag:current_lag]
            logits_a, state_a = adapter.encode_sequence(chunk, initial_snapshot=state_a, step_by_step=False)
            logits_b, state_b = adapter.encode_sequence(chunk, initial_snapshot=state_b, step_by_step=False)
            logits_sham, state_sham = adapter.encode_sequence(chunk, initial_snapshot=state_sham, step_by_step=False)
            prev_lag = current_lag

        # Check direct physical residency
        conv_resident = current_lag < (conv1d_width - 1)
        kv_resident = current_lag < (sliding_window - 1)

        # Measure layer-wise channel metrics
        rglru_metrics: Dict[int, StoreMetrics] = {}
        conv_metrics: Dict[int, StoreMetrics] = {}
        kv_metrics: Dict[int, StoreMetrics] = {}

        # RGLRU
        for l in state_a.rglru:
            if l in state_b.rglru:
                t1, t2 = state_a.rglru[l], state_b.rglru[l]
                d_rel = compute_scale_relative_dist(t1, t2)
                if current_lag == 0:
                    initial_d_rel_rglru[l] = d_rel
                init_d = initial_d_rel_rglru.get(l, d_rel)
                retention = d_rel / (init_d + 1e-8) if init_d > 0 else 1.0
                m = StoreMetrics(
                    rmsdiff=compute_rmsdiff(t1, t2),
                    scale_relative_dist=d_rel,
                    cossim=compute_cossim(t1, t2),
                    frobenius=float(torch.norm(t1.float() - t2.float()).item()),
                    retention_ratio=retention,
                )
                rglru_metrics[l] = m
                layer_records.append(
                    LayerTraceRecord(
                        pair_id=pair.pair_id,
                        regime=regime,
                        lag=current_lag,
                        channel="rglru",
                        layer_idx=l,
                        rmsdiff=m.rmsdiff,
                        scale_relative_dist=m.scale_relative_dist,
                        cossim=m.cossim,
                        frobenius=m.frobenius,
                        retention_ratio=m.retention_ratio,
                    )
                )

        # Conv
        for l in state_a.conv:
            if l in state_b.conv:
                t1, t2 = state_a.conv[l], state_b.conv[l]
                d_rel = compute_scale_relative_dist(t1, t2)
                if current_lag == 0:
                    initial_d_rel_conv[l] = d_rel
                init_d = initial_d_rel_conv.get(l, d_rel)
                retention = d_rel / (init_d + 1e-8) if init_d > 0 else 1.0
                m = StoreMetrics(
                    rmsdiff=compute_rmsdiff(t1, t2),
                    scale_relative_dist=d_rel,
                    cossim=compute_cossim(t1, t2),
                    frobenius=float(torch.norm(t1.float() - t2.float()).item()),
                    retention_ratio=retention,
                )
                conv_metrics[l] = m
                layer_records.append(
                    LayerTraceRecord(
                        pair_id=pair.pair_id,
                        regime=regime,
                        lag=current_lag,
                        channel="conv",
                        layer_idx=l,
                        rmsdiff=m.rmsdiff,
                        scale_relative_dist=m.scale_relative_dist,
                        cossim=m.cossim,
                        frobenius=m.frobenius,
                        retention_ratio=m.retention_ratio,
                    )
                )

        # KV (Separate K and V, plus aligned recent-entry divergence)
        for l in state_a.kv:
            if l in state_b.kv:
                k1, k2 = state_a.kv[l]["key"], state_b.kv[l]["key"]
                v1, v2 = state_a.kv[l]["value"], state_b.kv[l]["value"]
                
                # Whole-cache Key and Value distances
                d_k = compute_scale_relative_dist(k1, k2)
                d_v = compute_scale_relative_dist(v1, v2)
                d_kv_comb = (d_k + d_v) / 2.0
                
                # Aligned recent-entry divergence (last min(cache_len, 16) tokens)
                recent_len = min(k1.shape[-2], 16) if k1.numel() > 0 else 0
                if recent_len > 0:
                    rec_k1 = k1[..., -recent_len:, :]
                    rec_k2 = k2[..., -recent_len:, :]
                    rec_v1 = v1[..., -recent_len:, :]
                    rec_v2 = v2[..., -recent_len:, :]
                    rec_d_k = compute_scale_relative_dist(rec_k1, rec_k2)
                    rec_d_v = compute_scale_relative_dist(rec_v1, rec_v2)
                    recent_kv_d = (rec_d_k + rec_d_v) / 2.0
                else:
                    recent_kv_d = 0.0

                if current_lag == 0:
                    initial_d_rel_kv[l] = d_kv_comb
                    initial_d_rel_k[l] = d_k
                    initial_d_rel_v[l] = d_v

                init_d_kv = initial_d_rel_kv.get(l, d_kv_comb)
                retention_kv = d_kv_comb / (init_d_kv + 1e-8) if init_d_kv > 0 else 1.0

                m = StoreMetrics(
                    rmsdiff=compute_rmsdiff(k1, k2),
                    scale_relative_dist=d_kv_comb,
                    cossim=compute_cossim(k1, k2),
                    frobenius=float(torch.norm(k1.float() - k2.float()).item()),
                    retention_ratio=retention_kv,
                    key_d_rel=d_k,
                    val_d_rel=d_v,
                    recent_kv_d_rel=recent_kv_d,
                )
                kv_metrics[l] = m
                layer_records.append(
                    LayerTraceRecord(
                        pair_id=pair.pair_id,
                        regime=regime,
                        lag=current_lag,
                        channel="kv",
                        layer_idx=l,
                        rmsdiff=m.rmsdiff,
                        scale_relative_dist=m.scale_relative_dist,
                        cossim=m.cossim,
                        frobenius=m.frobenius,
                        retention_ratio=m.retention_ratio,
                        key_d_rel=d_k,
                        val_d_rel=d_v,
                        recent_kv_d_rel=recent_kv_d,
                    )
                )

        # Means across layers
        mean_rglru_d = sum(m.scale_relative_dist for m in rglru_metrics.values()) / max(len(rglru_metrics), 1)
        mean_conv_d = sum(m.scale_relative_dist for m in conv_metrics.values()) / max(len(conv_metrics), 1)
        mean_kv_d = sum(m.scale_relative_dist for m in kv_metrics.values()) / max(len(kv_metrics), 1)
        mean_k_d = sum(m.key_d_rel for m in kv_metrics.values()) / max(len(kv_metrics), 1)
        mean_v_d = sum(m.val_d_rel for m in kv_metrics.values()) / max(len(kv_metrics), 1)
        mean_recent_kv_d = sum(m.recent_kv_d_rel for m in kv_metrics.values()) / max(len(kv_metrics), 1)

        mean_rglru_ret = sum(m.retention_ratio for m in rglru_metrics.values()) / max(len(rglru_metrics), 1)
        mean_conv_ret = sum(m.retention_ratio for m in conv_metrics.values()) / max(len(conv_metrics), 1)
        mean_kv_ret = sum(m.retention_ratio for m in kv_metrics.values()) / max(len(kv_metrics), 1)
        
        init_k_mean = sum(initial_d_rel_k.values()) / max(len(initial_d_rel_k), 1)
        init_v_mean = sum(initial_d_rel_v.values()) / max(len(initial_d_rel_v), 1)
        mean_k_ret = mean_k_d / (init_k_mean + 1e-8) if init_k_mean > 0 else 1.0
        mean_v_ret = mean_v_d / (init_v_mean + 1e-8) if init_v_mean > 0 else 1.0

        # Behavioral divergence
        js_div = compute_jensen_shannon_div(logits_a, logits_b)
        pred_disagree = bool(torch.argmax(logits_a).item() != torch.argmax(logits_b).item())

        # 2AFC Cloze Probing (multi-token log-likelihood)
        margin_2afc, acc_2afc = evaluate_cloze_retrieval(
            adapter=adapter,
            snapshot_a=state_a,
            snapshot_b=state_b,
            cloze_tokens=cloze_tokens,
            target_a_tokens=target_a_tokens,
            target_b_tokens=target_b_tokens,
        )

        # Sham floor
        sham_dists = [
            compute_scale_relative_dist(state_a.rglru[l], state_sham.rglru[l])
            for l in state_a.rglru if l in state_sham.rglru
        ]
        sham_mean_d = sum(sham_dists) / max(len(sham_dists), 1)
        sham_js = compute_jensen_shannon_div(logits_a, logits_sham)

        records.append(
            LagCheckpointRecord(
                lag=current_lag,
                conv_directly_resident=conv_resident,
                kv_directly_resident=kv_resident,
                rglru_layers=rglru_metrics,
                conv_layers=conv_metrics,
                kv_layers=kv_metrics,
                mean_rglru_d_rel=mean_rglru_d,
                mean_conv_d_rel=mean_conv_d,
                mean_kv_d_rel=mean_kv_d,
                mean_k_d_rel=mean_k_d,
                mean_v_d_rel=mean_v_d,
                mean_recent_kv_d_rel=mean_recent_kv_d,
                mean_rglru_retention=mean_rglru_ret,
                mean_conv_retention=mean_conv_ret,
                mean_kv_retention=mean_kv_ret,
                mean_k_retention=mean_k_ret,
                mean_v_retention=mean_v_ret,
                jensen_shannon_div=js_div,
                top1_disagreement=pred_disagree,
                twoway_2afc_margin=margin_2afc,
                twoway_2afc_accuracy=acc_2afc,
                sham_mean_d_rel=sham_mean_d,
                sham_jensen_shannon_div=sham_js,
            )
        )

    return records, layer_records
