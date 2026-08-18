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


@dataclass
class LayerTraceRecord:
    """Per-layer, per-store metric record for layer x lag anatomy."""
    pair_id: str
    regime: str
    lag: int
    channel: str  # 'rglru', 'conv', 'kv'
    layer_idx: int
    rmsdiff: float
    scale_relative_dist: float
    cossim: float
    frobenius: float
    retention_ratio: float


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
    mean_rglru_retention: float = 1.0
    mean_conv_retention: float = 1.0
    mean_kv_retention: float = 1.0
    jensen_shannon_div: float = 0.0
    top1_disagreement: bool = False
    twoway_2afc_margin: float = 0.0
    twoway_2afc_accuracy: float = 0.0
    sham_mean_d_rel: float = 0.0
    sham_jensen_shannon_div: float = 0.0


def evaluate_cloze_retrieval(
    adapter: RecurrentGemmaAdapter,
    snapshot_a: RecurrentStateSnapshot,
    snapshot_b: RecurrentStateSnapshot,
    cloze_tokens: List[int],
    target_a_id: int,
    target_b_id: int,
) -> Tuple[float, float]:
    """Evaluate 2AFC cloze retrieval logit margin and accuracy from cloned snapshots."""
    # 1. Probe Branch A (detached clone)
    probe_a = snapshot_a.clone()
    logits_a, _ = adapter.encode_sequence(cloze_tokens, initial_snapshot=probe_a)
    score_a_target_a = logits_a[0, target_a_id].item() if logits_a.numel() > target_a_id else 0.0
    score_a_target_b = logits_a[0, target_b_id].item() if logits_a.numel() > target_b_id else 0.0
    margin_a = score_a_target_a - score_a_target_b

    # 2. Probe Branch B (detached clone)
    probe_b = snapshot_b.clone()
    logits_b, _ = adapter.encode_sequence(cloze_tokens, initial_snapshot=probe_b)
    score_b_target_b = logits_b[0, target_b_id].item() if logits_b.numel() > target_b_id else 0.0
    score_b_target_a = logits_b[0, target_a_id].item() if logits_b.numel() > target_a_id else 0.0
    margin_b = score_b_target_b - score_b_target_a

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
        target_a_id = target_a_tokens[0] if target_a_tokens else 10
        target_b_id = target_b_tokens[0] if target_b_tokens else 11
    else:
        prefix_tokens = [10, 11]
        event_a_tokens = [20, 21, 22]
        event_b_tokens = [30, 31, 32]
        cloze_tokens = [40, 41]
        target_a_id = 22
        target_b_id = 32

    assert len(event_a_tokens) == len(event_b_tokens), (
        f"Event A ({len(event_a_tokens)}) and Event B ({len(event_b_tokens)}) have mismatched token lengths!"
    )

    max_lag = max(lag_grid)
    exclude_set = {target_a_id, target_b_id}
    if audited_pool is None:
        audited_pool, _ = build_audited_vocabulary_pool(tokenizer, excluded_token_ids=exclude_set)

    filler_tokens = get_filler_tokens_for_regime(
        regime=regime,
        length=max_lag,
        seed=seed,
        audited_pool=audited_pool,
        tokenizer=tokenizer,
    )

    conv1d_width = getattr(adapter.config, "conv1d_width", 4)
    sliding_window = getattr(adapter.config, "attention_window_size", getattr(adapter.config, "sliding_window", 2048))

    # 2. Unroll prefix
    _, init_state = adapter.encode_sequence(prefix_tokens)

    # 3. Unroll Branch A, Branch B, and Sham A2
    logits_a, state_a = adapter.encode_sequence(event_a_tokens, initial_snapshot=init_state.clone())
    logits_b, state_b = adapter.encode_sequence(event_b_tokens, initial_snapshot=init_state.clone())
    logits_sham, state_sham = adapter.encode_sequence(event_a_tokens, initial_snapshot=init_state.clone())

    records: List[LagCheckpointRecord] = []
    layer_records: List[LayerTraceRecord] = []

    initial_d_rel_rglru: Dict[int, float] = {}
    initial_d_rel_conv: Dict[int, float] = {}
    initial_d_rel_kv: Dict[int, float] = {}

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

        # KV
        for l in state_a.kv:
            if l in state_b.kv:
                k1, k2 = state_a.kv[l]["key"], state_b.kv[l]["key"]
                d_rel = compute_scale_relative_dist(k1, k2)
                if current_lag == 0:
                    initial_d_rel_kv[l] = d_rel
                init_d = initial_d_rel_kv.get(l, d_rel)
                retention = d_rel / (init_d + 1e-8) if init_d > 0 else 1.0
                m = StoreMetrics(
                    rmsdiff=compute_rmsdiff(k1, k2),
                    scale_relative_dist=d_rel,
                    cossim=compute_cossim(k1, k2),
                    frobenius=float(torch.norm(k1.float() - k2.float()).item()),
                    retention_ratio=retention,
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
                    )
                )

        # Means
        mean_rglru_d = sum(m.scale_relative_dist for m in rglru_metrics.values()) / max(len(rglru_metrics), 1)
        mean_conv_d = sum(m.scale_relative_dist for m in conv_metrics.values()) / max(len(conv_metrics), 1)
        mean_kv_d = sum(m.scale_relative_dist for m in kv_metrics.values()) / max(len(kv_metrics), 1)

        mean_rglru_ret = sum(m.retention_ratio for m in rglru_metrics.values()) / max(len(rglru_metrics), 1)
        mean_conv_ret = sum(m.retention_ratio for m in conv_metrics.values()) / max(len(conv_metrics), 1)
        mean_kv_ret = sum(m.retention_ratio for m in kv_metrics.values()) / max(len(kv_metrics), 1)

        # Behavioral divergence
        js_div = compute_jensen_shannon_div(logits_a, logits_b)
        pred_disagree = bool(torch.argmax(logits_a).item() != torch.argmax(logits_b).item())

        # 2AFC Cloze Probing (from cloned snapshots)
        margin_2afc, acc_2afc = evaluate_cloze_retrieval(
            adapter=adapter,
            snapshot_a=state_a,
            snapshot_b=state_b,
            cloze_tokens=cloze_tokens,
            target_a_id=target_a_id,
            target_b_id=target_b_id,
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
                mean_rglru_retention=mean_rglru_ret,
                mean_conv_retention=mean_conv_ret,
                mean_kv_retention=mean_kv_ret,
                jensen_shannon_div=js_div,
                top1_disagreement=pred_disagree,
                twoway_2afc_margin=margin_2afc,
                twoway_2afc_accuracy=acc_2afc,
                sham_mean_d_rel=sham_mean_d,
                sham_jensen_shannon_div=sham_js,
            )
        )

    return records, layer_records
