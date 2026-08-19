"""Sprint S13: Controlled Task-Irrelevant Recurrent Dynamics Runner (Final Protocol S13.2).

Tracks the longitudinal trajectory V(N) = P_match(N) - P_wrong_val(N) across horizons
N in {0, 16, 64, 256, 1024, 2048} under 4 future drive regimes starting from a standardized
random 2W baseline at N=0, comparing Intact Recurrence vs RG-LRU Carry Clamped processing.

Protocol Architecture:
1. Canonical B=1 Common-Origin Preparation:
   - Prompts and 4096-token random filler history unrolled at B=1 (bit-identical to S12c/scout).
2. Canonical B=1 Common-Origin Measurement Probes:
   - Baseline intact outputs, u_0, and N=0 condition probes evaluated at B=1.
3. Optimized B=5 Future-Drive Evolution:
   - Parallel 5-branch execution (S_A, S_B, S_C, S_D, S_cross) along each 2048-token single drive stream.
   - Vectorized RG-LRU carry restore during clamped arm.
4. Canonical B=1 Horizon Probes:
   - All condition probes at horizons N in {0, 16, 64, 256, 1024, 2048} evaluated at B=1 with logits_to_keep=1.
5. Invariance & Geometry Diagnostics:
   - Token-clock audit, C_logit(N), C_R(N), Q_R(N), Cloze margins, and physical distances.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from recurrence.models.recurrent_gemma_adapter import (
    RecurrentGemmaAdapter,
    RecurrentGemmaConfig,
    RecurrentStateSnapshot,
    stack_snapshots,
    unstack_snapshot,
)
from recurrence.interventions.surgical_swaps import swap_stores, add_intervention_matched_noise
from recurrence.tasks.impulse_stimuli import (
    get_filler_tokens_for_regime,
    build_audited_vocabulary_pool,
)
from recurrence.tasks.specificity_microscope import (
    build_microscope_pairs,
    audit_microscope_panel,
    MicroscopePair,
    MICROSCOPE_FAMILIES,
)
from recurrence.tasks.controlled_drive import (
    verify_token_clock_invariance,
    generate_single_drive_stream,
    compute_frozen_axis,
    project_onto_axis,
    compute_logit_axis_cosine,
    compute_recurrent_state_diff_vec,
    compute_recurrent_geometry,
    advance_stream,
    advance_stream_along_horizons,
    advance_batched_stream_along_horizons,
)


def compute_condition_metrics(
    z_intervened: torch.Tensor,
    z_recipient: torch.Tensor,
    u_0: torch.Tensor,
    norm_0: float,
    u_N: torch.Tensor,
    norm_N: float,
    tok_rec_id: int,
    tok_don_id: int,
) -> Dict[str, Any]:
    """Compute projection metrics onto both frozen baseline u_0 and contemporaneous u_N axes."""
    disp_0, proj_0 = project_onto_axis(z_intervened, z_recipient, u_0, norm_0)
    disp_N, proj_N = project_onto_axis(z_intervened, z_recipient, u_N, norm_N)

    logit_rec = float(z_intervened[tok_rec_id].item())
    logit_don = float(z_intervened[tok_don_id].item())
    cloze_margin = float(logit_don - logit_rec)
    donor_is_top1 = bool(torch.argmax(z_intervened).item() == tok_don_id)

    return {
        "directional_displacement_u0": disp_0,
        "normalized_projection_u0": proj_0,
        "directional_displacement_uN": disp_N,
        "normalized_projection_uN": proj_N,
        "cloze_margin": cloze_margin,
        "donor_is_top1": donor_is_top1,
    }


def select_scout_pairs(all_pairs: List[MicroscopePair]) -> List[MicroscopePair]:
    """Select 4 scout pairs, exactly 1 from each of the 4 template families."""
    family_firsts = {}
    for p in all_pairs:
        if p.family_id not in family_firsts:
            family_firsts[p.family_id] = p
    return list(family_firsts.values())


@torch.inference_mode()
def run_experiment(
    phase: str = "confirmatory",
    model_id: str = "google/recurrentgemma-2b",
    dtype_str: str = "bfloat16",
    device: str = "cuda",
    output_dir: Optional[str] = None,
    seed: int = 42,
    pair_ids: Optional[List[str]] = None,
    pair_start: Optional[int] = None,
    pair_end: Optional[int] = None,
    resume: bool = False,
    dry_run: bool = False,
) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        run_name = f"run_e13_{phase}_{timestamp}"
        out_path = Path("results") / "e13_controlled_recurrent_dynamics" / run_name
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    trace_file = out_path / "dynamics_trace.jsonl"
    summary_file = out_path / "summary.json"

    # Git Provenance
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        is_clean = (len(git_status) == 0)
    except Exception:
        git_sha = "unknown"
        is_clean = False

    # Protocol code hashing
    code_hasher = hashlib.sha256()
    for fname in ["run.py", "analyze.py"]:
        fpath = Path("experiments") / "e13_controlled_recurrent_dynamics" / fname
        if fpath.exists():
            code_hasher.update(fpath.read_bytes())
    for frel in ["models/recurrent_gemma_adapter.py", "tasks/controlled_drive.py", "tasks/specificity_microscope.py", "interventions/surgical_swaps.py"]:
        fpath = Path("src") / "recurrence" / frel
        if fpath.exists():
            code_hasher.update(fpath.read_bytes())
    protocol_code_sha = code_hasher.hexdigest()

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    base_lag = 4096
    horizons = [0, 16, 64, 256, 1024, 2048]
    regimes = ["constant", "random", "natural", "interfering"]
    arms = ["intact_recurrence", "rglru_carry_clamped"]

    all_pairs = build_microscope_pairs()
    if phase == "scout":
        pairs = select_scout_pairs(all_pairs)
    else:
        pairs = all_pairs

    # Filtering / Sharding
    if pair_ids is not None and len(pair_ids) > 0:
        pairs = [p for p in pairs if p.pair_id in pair_ids]
    elif pair_start is not None or pair_end is not None:
        p_s = pair_start if pair_start is not None else 0
        p_e = pair_end if pair_end is not None else len(pairs)
        pairs = pairs[p_s:p_e]

    # Dry-Run Compute Gate Report
    if dry_run:
        n_pairs = len(pairs)
        total_rows = n_pairs * len(regimes) * len(arms) * 60
        batched_calls_per_pair = (len(regimes) * 4) + (len(regimes) * 2048)
        total_batched_calls = n_pairs * batched_calls_per_pair
        print("=" * 80)
        print(f"DRY-RUN COMPUTE GATE REPORT -- Sprint S13 ({phase.upper()})")
        print("=" * 80)
        print(f"Model ID:                {model_id} ({dtype_str} on {device})")
        print(f"Total Selected Pairs:    {n_pairs} (of {len(all_pairs)} total)")
        print(f"Drive Regimes (4):       {regimes}")
        print(f"Causal Arms (2):         {arms}")
        print(f"Horizons (6):            {horizons}")
        print(f"Total Rows to Write:     {total_rows:,} rows")
        print(f"Preparation Mode:        B=1 Canonical (56s/pair, exact scout bit-identity)")
        print(f"Measurement Probes:      B=1 Canonical (exact numerical stability)")
        print(f"Future Drive Execution:  B=5 Branch Batching (~114s/regime)")
        print(f"Estimated Time per Pair: ~8.8 minutes")
        print(f"Estimated Total Runtime: ~{n_pairs * 8.8 / 60:.2f} hours (for {n_pairs} pairs)")
        print(f"Clean Worktree Status:   {'CLEAN' if is_clean else 'DIRTY'}")
        print("=" * 80)
        return

    print(f"[E13] Initializing {phase.upper()} Controlled Recurrent Dynamics on device={device}, dtype={dtype_str}...")

    if phase == "confirmatory":
        if not is_clean:
            raise RuntimeError(
                "[E13 Fail-Closed Gate] Confirmatory run requires a completely clean git worktree. "
                "Commit or stash all uncommitted changes before launching."
            )
        if pair_ids is None and pair_start is None and pair_end is None:
            assert len(pairs) == 24, f"[E13 Fail-Closed Gate] Confirmatory requires exactly 24 pairs, got {len(pairs)}"
        assert model_id == "google/recurrentgemma-2b", f"Confirmatory requires google/recurrentgemma-2b, got {model_id}"
        assert dtype_str == "bfloat16", f"Confirmatory requires bfloat16, got {dtype_str}"
        assert device.startswith("cuda"), f"Confirmatory requires cuda, got {device}"
        assert torch.cuda.is_available(), "CUDA is not available"

    tokenizer = None
    if model_id is None or model_id == "reference_random":
        print("[E13] Running lightweight reference model scout.")
        config = RecurrentGemmaConfig(
            num_hidden_layers=4,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            lru_width=64,
            conv1d_width=4,
            sliding_window=16,
            block_types=["recurrent", "recurrent", "attention", "recurrent"],
            vocab_size=200,
        )
        torch.manual_seed(seed)
        adapter = RecurrentGemmaAdapter(config=config, device=device, dtype=torch.float32)
        model_provenance = {
            "model_id": "reference_random",
            "is_reference_model": True,
            "vocab_size": config.vocab_size,
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "model_revision": "local_mock",
        }
    else:
        print(f"[E13] Loading pretrained model '{model_id}' (FAIL-CLOSED)...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
            adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=device, dtype=dtype)
            model_commit = getattr(model.config, "_commit_hash", None) or getattr(tokenizer, "_commit_hash", "resolved_hub_head")
            model_provenance = {
                "model_id": model_id,
                "is_reference_model": False,
                "model_revision": str(model_commit),
                "model_class": model.__class__.__name__,
                "vocab_size": getattr(model.config, "vocab_size", None),
                "num_hidden_layers": getattr(model.config, "num_hidden_layers", 26),
                "hidden_size": getattr(model.config, "hidden_size", 2560),
                "conv1d_width": getattr(model.config, "conv1d_width", 4),
                "attention_window_size": getattr(model.config, "sliding_window", 2048),
            }
        except Exception as e:
            raise RuntimeError(f"[E13 Fail-Closed Gate] Pretrained model load failed: {e}") from e

    # Phase S13.0 Token-Clock Identity Audit
    print("[E13] Executing S13.0 Token-Clock Invariance Audit (T_theta^(0)(S) = S)...")
    token_clock_valid = verify_token_clock_invariance(adapter)
    print(f"[E13] S13.0 Token-Clock Invariance Audit: {'PASSED' if token_clock_valid else 'FAILED'}")
    if not token_clock_valid:
        raise RuntimeError("[E13 Fail-Closed Gate] S13.0 Token-Clock Invariance Audit failed!")

    # Audit microscope panel
    is_valid, panel_hash, panel_audit = audit_microscope_panel(adapter.tokenizer)
    print(f"[E13] Specificity Panel Audit: {panel_audit['status']} (SHA256: {panel_hash})")
    if phase == "confirmatory" and not is_valid:
        raise ValueError(f"[E13 Fail-Closed Gate] Panel audit failed: {panel_audit['status']}")

    # Audited pool
    audited_pool = None
    if adapter.tokenizer is not None:
        audited_pool, _ = build_audited_vocabulary_pool(adapter.tokenizer)

    start_time = datetime.datetime.now()
    records_written = 0
    seen_cells: Set[Tuple[str, str, str, int, str]] = set()

    # Resumability handling
    completed_pairs: Set[str] = set()
    open_mode = "w"
    if resume and trace_file.exists():
        print(f"[E13] Resuming from existing trace at {trace_file}...")
        pair_row_counts: Dict[str, int] = {}
        with open(trace_file, "r", encoding="utf-8") as f_prev:
            for line in f_prev:
                if line.strip():
                    r = json.loads(line)
                    p_id = r["pair_id"]
                    cell_k = (p_id, r["regime"], r["arm"], r["horizon"], r["condition"])
                    seen_cells.add(cell_k)
                    pair_row_counts[p_id] = pair_row_counts.get(p_id, 0) + 1
                    records_written += 1
        for p_id, count in pair_row_counts.items():
            if count == 480:
                completed_pairs.add(p_id)
        print(f"[E13] Found {len(completed_pairs)} already completed pairs ({records_written} rows).")
        open_mode = "a"

    with open(trace_file, open_mode, encoding="utf-8") as f_trace:
        for p_idx, pair in enumerate(pairs):
            if pair.pair_id in completed_pairs:
                print(f"[E13] Skipping already completed pair {p_idx+1:02d}/{len(pairs)} ({pair.pair_id})")
                continue

            pair_start = datetime.datetime.now()

            # Tokenize prompts
            if adapter.tokenizer is not None:
                toks_prefix = adapter.tokenizer.encode(pair.prefix, add_special_tokens=False)
                toks_prompt_a = adapter.tokenizer.encode(pair.prompt_a, add_special_tokens=False)
                toks_prompt_b = adapter.tokenizer.encode(pair.prompt_b, add_special_tokens=False)
                toks_prompt_c = adapter.tokenizer.encode(pair.prompt_c, add_special_tokens=False)
                toks_prompt_d = adapter.tokenizer.encode(pair.prompt_d, add_special_tokens=False)
                toks_prompt_cross = adapter.tokenizer.encode(pair.prompt_cross, add_special_tokens=False)

                toks_query = adapter.tokenizer.encode(pair.query, add_special_tokens=False)
                tok_a_id = adapter.tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
                tok_b_id = adapter.tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]
            else:
                toks_prefix = [1, 2]
                toks_prompt_a = [1, 2, 10, 11]
                toks_prompt_b = [1, 2, 20, 21]
                toks_prompt_c = [1, 2, 30, 31]
                toks_prompt_d = [1, 2, 40, 41]
                toks_prompt_cross = [1, 2, 50, 51]
                toks_query = [100, 101]
                tok_a_id = 11
                tok_b_id = 21

            pair_excluded = set(toks_prompt_a + toks_prompt_b + toks_prompt_c + toks_prompt_d + toks_prompt_cross + [tok_a_id, tok_b_id])

            # 1. Canonical B=1 Common-Origin Preparation: Random Regime at L_0 = 4096
            cur_seed = seed + p_idx * 100
            random_init_filler = get_filler_tokens_for_regime(
                regime="random",
                length=base_lag,
                seed=cur_seed,
                audited_pool=audited_pool,
                tokenizer=adapter.tokenizer,
                excluded_token_ids=pair_excluded,
            )

            # Unroll initial prompts (B=1 state-only base model execution)
            _, state_a_0 = adapter.encode_sequence(toks_prompt_a, step_by_step=False, return_logits=False)
            _, state_b_0 = adapter.encode_sequence(toks_prompt_b, step_by_step=False, return_logits=False)
            _, state_c_0 = adapter.encode_sequence(toks_prompt_c, step_by_step=False, return_logits=False)
            _, state_d_0 = adapter.encode_sequence(toks_prompt_d, step_by_step=False, return_logits=False)
            _, state_cross_0 = adapter.encode_sequence(toks_prompt_cross, step_by_step=False, return_logits=False)

            # Unroll 4096 random filler tokens at B=1 (bit-identical to S12c/scout canonical origin)
            chunk_size = 512
            for i in range(0, base_lag, chunk_size):
                end_idx = min(i + chunk_size, base_lag)
                chunk = random_init_filler[i:end_idx]
                _, state_a_0 = adapter.encode_sequence(chunk, initial_snapshot=state_a_0, step_by_step=False, return_logits=False)
                _, state_b_0 = adapter.encode_sequence(chunk, initial_snapshot=state_b_0, step_by_step=False, return_logits=False)
                _, state_c_0 = adapter.encode_sequence(chunk, initial_snapshot=state_c_0, step_by_step=False, return_logits=False)
                _, state_d_0 = adapter.encode_sequence(chunk, initial_snapshot=state_d_0, step_by_step=False, return_logits=False)
                _, state_cross_0 = adapter.encode_sequence(chunk, initial_snapshot=state_cross_0, step_by_step=False, return_logits=False)

            # Baseline Intact Outputs at N=0 (evaluated at B=1 with logits_to_keep=1)
            out_intact_a_0, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_a_0.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
            out_intact_b_0, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_b_0.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
            z_intact_a_0 = out_intact_a_0[0]
            z_intact_b_0 = out_intact_b_0[0]

            # 2. Compute and Cache Frozen Baseline Axes & State Difference Vectors at N=0
            u_0_a2b, norm_0_a2b = compute_frozen_axis(z_intact_a_0, z_intact_b_0)
            u_0_b2a, norm_0_b2a = compute_frozen_axis(z_intact_b_0, z_intact_a_0)
            r_0_a2b = compute_recurrent_state_diff_vec(state_a_0, state_b_0)
            r_0_b2a = compute_recurrent_state_diff_vec(state_b_0, state_a_0)

            # 3. Fork into Future Drive Regimes
            for reg in regimes:
                # Generate single 2048-token frozen drive stream from which all prefixes are taken
                drive_seed = cur_seed + 5000
                drive_stream_2048 = generate_single_drive_stream(
                    length=2048,
                    regime=reg,
                    seed=drive_seed,
                    tokenizer=adapter.tokenizer,
                    audited_pool=audited_pool,
                    excluded_token_ids=pair_excluded,
                )

                for arm in arms:
                    # Advance all 5 branches in parallel along the single stream (B=5)
                    branch_snaps = advance_batched_stream_along_horizons(
                        adapter,
                        [state_a_0, state_b_0, state_c_0, state_d_0, state_cross_0],
                        drive_stream_2048,
                        horizons=horizons,
                        arm=arm,
                    )
                    snaps_a = branch_snaps[0]
                    snaps_b = branch_snaps[1]
                    snaps_c = branch_snaps[2]
                    snaps_d = branch_snaps[3]
                    snaps_cross = branch_snaps[4]

                    for horizon in horizons:
                        state_a_N = snaps_a[horizon]
                        state_b_N = snaps_b[horizon]
                        state_c_N = snaps_c[horizon]
                        state_d_N = snaps_d[horizon]
                        state_cross_N = snaps_cross[horizon]

                        # Physical RG-LRU divergence
                        dist_ab = float(sum(torch.norm(state_a_N.rglru[l].float() - state_b_N.rglru[l].float()).item() for l in state_a_N.rglru))
                        dist_ac = float(sum(torch.norm(state_a_N.rglru[l].float() - state_c_N.rglru[l].float()).item() for l in state_a_N.rglru))

                        # Baseline Intact Outputs at horizon N (B=1)
                        out_intact_a_N, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_a_N.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
                        out_intact_b_N, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_b_N.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
                        z_intact_a_N = out_intact_a_N[0]
                        z_intact_b_N = out_intact_b_N[0]

                        # Contemporaneous axes and state diff vectors at horizon N
                        u_N_a2b, norm_N_a2b = compute_frozen_axis(z_intact_a_N, z_intact_b_N)
                        u_N_b2a, norm_N_b2a = compute_frozen_axis(z_intact_b_N, z_intact_a_N)
                        r_N_a2b = compute_recurrent_state_diff_vec(state_a_N, state_b_N)
                        r_N_b2a = compute_recurrent_state_diff_vec(state_b_N, state_a_N)

                        c_logit_a2b = compute_logit_axis_cosine(u_0_a2b, u_N_a2b)
                        c_logit_b2a = compute_logit_axis_cosine(u_0_b2a, u_N_b2a)

                        c_r_a2b, q_r_a2b = compute_recurrent_geometry(r_0_a2b, r_N_a2b)
                        c_r_b2a, q_r_b2a = compute_recurrent_geometry(r_0_b2a, r_N_b2a)

                        # Build and evaluate condition states at B=1
                        # Direction A -> B (Recipient B, Donor A/C/D)
                        state_match_a2b = swap_stores(recipient=state_b_N, donor=state_a_N, channels="rglru")
                        out_match_a2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_match_a2b, step_by_step=False, return_logits=True, logits_to_keep=1)
                        z_match_a2b = out_match_a2b[0]

                        state_wrong_c2b = swap_stores(recipient=state_b_N, donor=state_c_N, channels="rglru")
                        out_wrong_c2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_wrong_c2b, step_by_step=False, return_logits=True, logits_to_keep=1)
                        z_wrong_c2b = out_wrong_c2b[0]

                        state_wrong_d2b = swap_stores(recipient=state_b_N, donor=state_d_N, channels="rglru")
                        out_wrong_d2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_wrong_d2b, step_by_step=False, return_logits=True, logits_to_keep=1)
                        z_wrong_d2b = out_wrong_d2b[0]

                        eval_conditions_a2b = [
                            ("intact_b", z_intact_b_N, "none"),
                            ("matching_rglru_a_into_b", z_match_a2b, "A"),
                            ("same_template_wrong_c_into_b", z_wrong_c2b, "C"),
                            ("same_template_wrong_d_into_b", z_wrong_d2b, "D"),
                        ]

                        # Direction B -> A (Recipient A, Donor B/C/D)
                        state_match_b2a = swap_stores(recipient=state_a_N, donor=state_b_N, channels="rglru")
                        out_match_b2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_match_b2a, step_by_step=False, return_logits=True, logits_to_keep=1)
                        z_match_b2a = out_match_b2a[0]

                        state_wrong_c2a = swap_stores(recipient=state_a_N, donor=state_c_N, channels="rglru")
                        out_wrong_c2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_wrong_c2a, step_by_step=False, return_logits=True, logits_to_keep=1)
                        z_wrong_c2a = out_wrong_c2a[0]

                        state_wrong_d2a = swap_stores(recipient=state_a_N, donor=state_d_N, channels="rglru")
                        out_wrong_d2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_wrong_d2a, step_by_step=False, return_logits=True, logits_to_keep=1)
                        z_wrong_d2a = out_wrong_d2a[0]

                        eval_conditions_b2a = [
                            ("intact_a", z_intact_a_N, "none"),
                            ("matching_rglru_b_into_a", z_match_b2a, "B"),
                            ("same_template_wrong_c_into_a", z_wrong_c2a, "C"),
                            ("same_template_wrong_d_into_a", z_wrong_d2a, "D"),
                        ]

                        # Secondary reference controls at N=0 and N=2048 endpoints
                        if horizon in (0, 2048):
                            state_cross_a2b = swap_stores(recipient=state_b_N, donor=state_cross_N, channels="rglru")
                            out_cross_a2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_cross_a2b, step_by_step=False, return_logits=True, logits_to_keep=1)
                            z_cross_a2b = out_cross_a2b[0]

                            state_noise_a2b = add_intervention_matched_noise(recipient=state_b_N, donor=state_a_N, channel="rglru", seed=cur_seed + 10)
                            out_noise_a2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_noise_a2b, step_by_step=False, return_logits=True, logits_to_keep=1)
                            z_noise_a2b = out_noise_a2b[0]

                            out_whole_a2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_a_N.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
                            z_whole_a2b = out_whole_a2b[0]

                            eval_conditions_a2b.extend([
                                ("cross_template_e_into_b", z_cross_a2b, "cross"),
                                ("noise_rglru_a_into_b", z_noise_a2b, "noise"),
                                ("whole_swap_a_into_b", z_whole_a2b, "A"),
                            ])

                            state_cross_b2a = swap_stores(recipient=state_a_N, donor=state_cross_N, channels="rglru")
                            out_cross_b2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_cross_b2a, step_by_step=False, return_logits=True, logits_to_keep=1)
                            z_cross_b2a = out_cross_b2a[0]

                            state_noise_b2a = add_intervention_matched_noise(recipient=state_a_N, donor=state_b_N, channel="rglru", seed=cur_seed + 11)
                            out_noise_b2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_noise_b2a, step_by_step=False, return_logits=True, logits_to_keep=1)
                            z_noise_b2a = out_noise_b2a[0]

                            out_whole_b2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_b_N.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
                            z_whole_b2a = out_whole_b2a[0]

                            eval_conditions_b2a.extend([
                                ("cross_template_e_into_a", z_cross_b2a, "cross"),
                                ("noise_rglru_b_into_a", z_noise_b2a, "noise"),
                                ("whole_swap_b_into_a", z_whole_b2a, "B"),
                            ])

                        # Record Direction A -> B
                        for c_name, z_int, don_label in eval_conditions_a2b:
                            cell_key = (pair.pair_id, reg, arm, horizon, c_name)
                            if cell_key in seen_cells:
                                raise ValueError(f"[E13 Fail-Closed Gate] Duplicate cell detected: {cell_key}")
                            seen_cells.add(cell_key)

                            m = compute_condition_metrics(
                                z_intervened=z_int,
                                z_recipient=z_intact_b_N,
                                u_0=u_0_a2b,
                                norm_0=norm_0_a2b,
                                u_N=u_N_a2b,
                                norm_N=norm_N_a2b,
                                tok_rec_id=tok_b_id,
                                tok_don_id=tok_a_id,
                            )
                            rec = {
                                "pair_id": pair.pair_id,
                                "family_id": pair.family_id,
                                "val_a": pair.val_a,
                                "val_b": pair.val_b,
                                "regime": reg,
                                "arm": arm,
                                "horizon": horizon,
                                "direction": "a_into_b",
                                "recipient": "B",
                                "donor": don_label,
                                "condition": c_name,
                                "c_logit": c_logit_a2b,
                                "c_r": c_r_a2b,
                                "q_r": q_r_a2b,
                                "physical_dist_ab": dist_ab,
                                "physical_dist_ac": dist_ac,
                                **m,
                            }
                            f_trace.write(json.dumps(rec) + "\n")
                            records_written += 1

                        # Record Direction B -> A
                        for c_name, z_int, don_label in eval_conditions_b2a:
                            cell_key = (pair.pair_id, reg, arm, horizon, c_name)
                            if cell_key in seen_cells:
                                raise ValueError(f"[E13 Fail-Closed Gate] Duplicate cell detected: {cell_key}")
                            seen_cells.add(cell_key)

                            m = compute_condition_metrics(
                                z_intervened=z_int,
                                z_recipient=z_intact_a_N,
                                u_0=u_0_b2a,
                                norm_0=norm_0_b2a,
                                u_N=u_N_b2a,
                                norm_N=norm_N_b2a,
                                tok_rec_id=tok_a_id,
                                tok_don_id=tok_b_id,
                            )
                            rec = {
                                "pair_id": pair.pair_id,
                                "family_id": pair.family_id,
                                "val_a": pair.val_a,
                                "val_b": pair.val_b,
                                "regime": reg,
                                "arm": arm,
                                "horizon": horizon,
                                "direction": "b_into_a",
                                "recipient": "A",
                                "donor": don_label,
                                "condition": c_name,
                                "c_logit": c_logit_b2a,
                                "c_r": c_r_b2a,
                                "q_r": q_r_b2a,
                                "physical_dist_ab": dist_ab,
                                "physical_dist_ac": dist_ac,
                                **m,
                            }
                            f_trace.write(json.dumps(rec) + "\n")
                            records_written += 1

            f_trace.flush()
            pair_dur = (datetime.datetime.now() - pair_start).total_seconds()
            print(f"[E13] Pair {p_idx+1:02d}/{len(pairs)} ({pair.pair_id}) complete in {pair_dur:.1f}s ({records_written} total rows)")

    total_elapsed = (datetime.datetime.now() - start_time).total_seconds()

    if pair_ids is None and pair_start is None and pair_end is None:
        expected_records = 11520 if phase == "confirmatory" else 1920
        assert records_written == expected_records, (
            f"[E13 Fail-Closed Gate] Expected exactly {expected_records} records, wrote {records_written}"
        )

    summary_data = {
        "phase": phase,
        "timestamp": timestamp,
        "git_commit": git_sha,
        "is_clean_worktree": is_clean,
        "protocol_code_sha256": protocol_code_sha,
        "panel_hash_sha256": panel_hash,
        "model_provenance": model_provenance,
        "total_pairs": len(pairs),
        "total_records_written": records_written,
        "total_elapsed_seconds": total_elapsed,
        "seconds_per_pair": total_elapsed / len(pairs) if len(pairs) > 0 else 0,
        "config": {
            "base_lag": base_lag,
            "horizons": horizons,
            "regimes": regimes,
            "arms": arms,
            "seed": seed,
            "dtype": dtype_str,
            "device": device,
        },
    }

    with open(summary_file, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2)

    print(f"[E13] Completed {phase.upper()} run in {total_elapsed:.1f}s. Summary written to {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint S13 Controlled Recurrent Dynamics Runner")
    parser.add_argument("--phase", type=str, choices=["scout", "confirmatory"], default="confirmatory")
    parser.add_argument("--model_id", type=str, default="google/recurrentgemma-2b")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_ids", nargs="+", default=None, help="Optional subset of pair IDs to run")
    parser.add_argument("--pair_start", type=int, default=None, help="Start index for sharding")
    parser.add_argument("--pair_end", type=int, default=None, help="End index for sharding")
    parser.add_argument("--resume", action="store_true", help="Resume existing run from output directory")
    parser.add_argument("--dry_run", action="store_true", help="Print dry-run compute profile and exit")
    args = parser.parse_args()

    run_experiment(
        phase=args.phase,
        model_id=args.model_id,
        dtype_str=args.dtype,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        pair_ids=args.pair_ids,
        pair_start=args.pair_start,
        pair_end=args.pair_end,
        resume=args.resume,
        dry_run=args.dry_run,
    )
