"""Sprint S13: Controlled Task-Irrelevant Recurrent Dynamics Module.

Provides:
1. S13.0 Token-clock identity invariant verification (T_theta^(0)(S) = S).
2. Standardized N=0 common-origin state preparation using 2W random filler history.
3. Frozen 2048-token single-stream prefix generator for 4 drive regimes.
4. Frozen baseline output axis (u_0) computation and projection utilities.
5. Causal recurrent carry clamping (rglru_carry_clamped) execution helper.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import torch

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.state.temporal_inventory import RecurrentStateSnapshot
from recurrence.tasks.impulse_stimuli import (
    get_filler_tokens_for_regime,
    build_audited_vocabulary_pool,
)
from recurrence.tasks.specificity_microscope import MicroscopePair


def verify_token_clock_invariance(adapter: RecurrentGemmaAdapter) -> bool:
    """Verify Phase S13.0 invariant: T_theta^(0)(S) = S for empty token sequences."""
    state = adapter.create_canonical_initial_state()

    # 1. Test encode_sequence with empty list
    logits, state_out = adapter.encode_sequence([], initial_snapshot=state.clone(), step_by_step=False)
    assert logits.numel() == 0, f"Expected 0 logits for empty sequence, got {logits.shape}"
    assert state_out.cache_position == state.cache_position, "Cache position mutated on empty sequence"

    # Verify layer-by-layer bit identity
    for l_idx in state.rglru:
        assert torch.equal(state.rglru[l_idx], state_out.rglru[l_idx]), f"RGLRU mutated at layer {l_idx}"
    for l_idx in state.conv:
        assert torch.equal(state.conv[l_idx], state_out.conv[l_idx]), f"Conv mutated at layer {l_idx}"

    return True


def generate_single_drive_stream(
    length: int = 2048,
    regime: str = "random",
    seed: int = 42,
    tokenizer: Optional[Any] = None,
    audited_pool: Optional[List[int]] = None,
    excluded_token_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Generate a single frozen drive stream of specified length from which prefixes are sliced."""
    return get_filler_tokens_for_regime(
        regime=regime,
        length=length,
        seed=seed,
        audited_pool=audited_pool,
        tokenizer=tokenizer,
        excluded_token_ids=excluded_token_ids,
    )


def compute_frozen_axis(
    z_don_0: torch.Tensor,
    z_rec_0: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    """Compute frozen baseline unit vector u_0 = (z_D(0) - z_R(0)) / ||z_D(0) - z_R(0)|| and its norm."""
    diff = (z_don_0 - z_rec_0).flatten().float()
    norm = float(torch.norm(diff).item())
    if norm < 1e-6:
        unit = torch.zeros_like(diff)
    else:
        unit = diff / norm
    return unit, norm


def project_onto_axis(
    z_intervened: torch.Tensor,
    z_recipient: torch.Tensor,
    unit_axis: torch.Tensor,
    norm_baseline: float,
) -> Tuple[float, float]:
    """Project displacement (z_intervened - z_recipient) onto a specified unit axis.
    
    Returns:
        (directional_displacement, normalized_projection_fraction)
    """
    diff = (z_intervened - z_recipient).flatten().float()
    dir_disp = float(torch.sum(diff * unit_axis).item())
    proj_fraction = float(dir_disp / norm_baseline) if norm_baseline > 1e-6 else 0.0
    return dir_disp, proj_fraction


def compute_logit_axis_cosine(
    u_0: torch.Tensor,
    u_N: torch.Tensor,
) -> float:
    """Compute cosine similarity between frozen baseline axis u_0 and contemporaneous axis u_N.
    
    Since both u_0 and u_N are unit vectors, C_logit(N) = u_0^T u_N.
    """
    return float(torch.sum(u_0.flatten().float() * u_N.flatten().float()).item())


def compute_recurrent_state_diff_vec(
    state_don: RecurrentStateSnapshot,
    state_rec: RecurrentStateSnapshot,
) -> torch.Tensor:
    """Concatenate layerwise RG-LRU recurrent state differences into a single 1D tensor r in R^D."""
    diffs = []
    for l_idx in sorted(state_don.rglru.keys()):
        d = (state_don.rglru[l_idx].float() - state_rec.rglru[l_idx].float()).flatten()
        diffs.append(d)
    if not diffs:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(diffs, dim=0)


def compute_recurrent_geometry(
    r_0: torch.Tensor,
    r_N: torch.Tensor,
) -> Tuple[float, float]:
    """Compute recurrent state rotation cosine C_R(N) and magnitude retention quotient Q_R(N).
    
    Returns:
        (C_R, Q_R) where:
            C_R(N) = (r_0^T r_N) / (||r_0||_2 * ||r_N||_2)
            Q_R(N) = ||r_N||_2 / ||r_0||_2
    """
    norm_0 = float(torch.norm(r_0).item())
    norm_N = float(torch.norm(r_N).item())

    if norm_0 < 1e-6 or norm_N < 1e-6:
        c_r = 0.0
    else:
        c_r = float((torch.sum(r_0 * r_N) / (norm_0 * norm_N)).item())

    q_r = float(norm_N / norm_0) if norm_0 > 1e-6 else 0.0
    return c_r, q_r


@torch.no_grad()
def advance_stream(
    adapter: RecurrentGemmaAdapter,
    initial_snapshot: RecurrentStateSnapshot,
    token_ids: List[int],
    arm: str = "intact_recurrence",
) -> RecurrentStateSnapshot:
    """Advance a state snapshot through token_ids under specified causal arm."""
    if not token_ids:
        return initial_snapshot.clone()

    if arm == "intact_recurrence":
        chunk_size = 512
        state = initial_snapshot.clone()
        for i in range(0, len(token_ids), chunk_size):
            end_idx = min(i + chunk_size, len(token_ids))
            chunk = token_ids[i:end_idx]
            _, state = adapter.encode_sequence(chunk, initial_snapshot=state, step_by_step=False)
        return state

    elif arm == "rglru_carry_clamped":
        cache = adapter.inject_state_snapshot(initial_snapshot)
        s0_rglru = {l: initial_snapshot.rglru[l].detach().clone().to(device=adapter.device) for l in initial_snapshot.rglru}
        pos = initial_snapshot.cache_position

        for i, tok in enumerate(token_ids):
            input_ids = torch.tensor([[tok]], device=adapter.device, dtype=torch.long)
            position_ids = torch.tensor([[pos + i]], device=adapter.device, dtype=torch.long)

            adapter.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

            # Restore RG-LRU carry to S_0 directly on layer modules
            for l_idx, layer in enumerate(adapter.model.model.layers):
                block = layer.temporal_block
                if hasattr(block, "rg_lru") and l_idx in s0_rglru:
                    block.rg_lru.recurrent_states = s0_rglru[l_idx].detach().clone()

        state_out = adapter.extract_state_snapshot(
            past_key_values=cache,
            cache_position=pos + len(token_ids),
        )
        state_out.metadata["arm"] = "rglru_carry_clamped"
        return state_out

    else:
        raise ValueError(f"Unknown causal arm '{arm}'")


@torch.no_grad()
def advance_stream_along_horizons(
    adapter: RecurrentGemmaAdapter,
    initial_snapshot: RecurrentStateSnapshot,
    stream_2048: List[int],
    horizons: List[int] = [0, 16, 64, 256, 1024, 2048],
    arm: str = "intact_recurrence",
) -> Dict[int, RecurrentStateSnapshot]:
    """Advance a state snapshot sequentially across horizon checkpoints along a single stream."""
    snapshots = {0: initial_snapshot.clone()}
    if not stream_2048:
        return {h: initial_snapshot.clone() for h in horizons}

    if arm == "intact_recurrence":
        cur_state = initial_snapshot.clone()
        prev_h = 0
        for h in horizons:
            if h == 0:
                continue
            interval_tokens = stream_2048[prev_h:h]
            chunk_size = 512
            for i in range(0, len(interval_tokens), chunk_size):
                end_idx = min(i + chunk_size, len(interval_tokens))
                chunk = interval_tokens[i:end_idx]
                _, cur_state = adapter.encode_sequence(chunk, initial_snapshot=cur_state, step_by_step=False)
            snapshots[h] = cur_state.clone()
            prev_h = h
        return snapshots

    elif arm == "rglru_carry_clamped":
        cache = adapter.inject_state_snapshot(initial_snapshot)
        s0_rglru = {l: initial_snapshot.rglru[l].detach().clone().to(device=adapter.device) for l in initial_snapshot.rglru}
        pos = initial_snapshot.cache_position
        max_h = max(horizons)
        stream_tokens = stream_2048[:max_h]

        for i, tok in enumerate(stream_tokens):
            cur_pos = pos + i
            input_ids = torch.tensor([[tok]], device=adapter.device, dtype=torch.long)
            position_ids = torch.tensor([[cur_pos]], device=adapter.device, dtype=torch.long)

            adapter.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )

            # Restore RG-LRU carry to S_0 directly on layer modules
            for l_idx, layer in enumerate(adapter.model.model.layers):
                block = layer.temporal_block
                if hasattr(block, "rg_lru") and l_idx in s0_rglru:
                    block.rg_lru.recurrent_states = s0_rglru[l_idx].detach().clone()

            step_h = i + 1
            if step_h in horizons:
                snap = adapter.extract_state_snapshot(
                    past_key_values=cache,
                    cache_position=pos + step_h,
                )
                snap.metadata["arm"] = "rglru_carry_clamped"
                snapshots[step_h] = snap

        return snapshots

    else:
        raise ValueError(f"Unknown causal arm '{arm}'")
