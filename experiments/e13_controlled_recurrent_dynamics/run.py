"""Sprint S13: Controlled Task-Irrelevant Recurrent Dynamics Runner.

Tracks the longitudinal trajectory V(N) = P_match(N) - P_wrong_val(N) across horizons
N in {0, 16, 64, 256, 1024, 2048} under 4 future drive regimes starting from a standardized
random 2W baseline at N=0, comparing Intact Recurrence vs RG-LRU Carry Clamped processing.
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

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter, RecurrentGemmaConfig
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
    advance_stream,
    advance_stream_along_horizons,
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


@torch.no_grad()
def run_experiment(
    phase: str = "confirmatory",
    model_id: str = "google/recurrentgemma-2b",
    dtype_str: str = "bfloat16",
    device: str = "cuda",
    output_dir: Optional[str] = None,
    seed: int = 42,
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
    except Exception as e:
        git_sha = "unknown"
        is_clean = False

    # Protocol code hashing
    code_hasher = hashlib.sha256()
    for fname in ["run.py", "analyze.py"]:
        fpath = Path("experiments") / "e13_controlled_recurrent_dynamics" / fname
        if fpath.exists():
            code_hasher.update(fpath.read_bytes())
    for frel in ["tasks/controlled_drive.py", "tasks/specificity_microscope.py", "interventions/surgical_swaps.py"]:
        fpath = Path("src") / "recurrence" / frel
        if fpath.exists():
            code_hasher.update(fpath.read_bytes())
    protocol_code_sha = code_hasher.hexdigest()

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    base_lag = 4096
    horizons = [0, 16, 64, 256, 1024, 2048]
    regimes = ["constant", "random", "natural", "interfering"]
    arms = ["intact_recurrence", "rglru_carry_clamped"]

    print(f"[E13] Initializing {phase.upper()} Controlled Recurrent Dynamics on device={device}, dtype={dtype_str}...")

    if phase == "confirmatory":
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

    all_pairs = build_microscope_pairs()
    if phase == "scout":
        pairs = select_scout_pairs(all_pairs)
        print(f"[E13 Scout] Selected {len(pairs)} scout pairs (1 per template family).")
    else:
        pairs = all_pairs
        print(f"[E13 Confirmatory] Running all {len(pairs)} pairs across 4 template families.")

    start_time = datetime.datetime.now()
    records_written = 0

    with open(trace_file, "w", encoding="utf-8") as f_trace:
        for p_idx, pair in enumerate(pairs):
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

            # 1. Standardized Common Origin Preparation: Random Regime at L_0 = 4096
            cur_seed = seed + p_idx * 100
            random_init_filler = get_filler_tokens_for_regime(
                regime="random",
                length=base_lag,
                seed=cur_seed,
                audited_pool=audited_pool,
                tokenizer=adapter.tokenizer,
                excluded_token_ids=pair_excluded,
            )

            # Unroll initial prompts
            _, state_a_0 = adapter.encode_sequence(toks_prompt_a, step_by_step=False)
            _, state_b_0 = adapter.encode_sequence(toks_prompt_b, step_by_step=False)
            _, state_c_0 = adapter.encode_sequence(toks_prompt_c, step_by_step=False)
            _, state_d_0 = adapter.encode_sequence(toks_prompt_d, step_by_step=False)
            _, state_cross_0 = adapter.encode_sequence(toks_prompt_cross, step_by_step=False)

            # Unroll 4096 random filler tokens chunk-by-chunk to establish N=0 common origin
            chunk_size = 512
            for i in range(0, base_lag, chunk_size):
                end_idx = min(i + chunk_size, base_lag)
                chunk = random_init_filler[i:end_idx]
                _, state_a_0 = adapter.encode_sequence(chunk, initial_snapshot=state_a_0, step_by_step=False)
                _, state_b_0 = adapter.encode_sequence(chunk, initial_snapshot=state_b_0, step_by_step=False)
                _, state_c_0 = adapter.encode_sequence(chunk, initial_snapshot=state_c_0, step_by_step=False)
                _, state_d_0 = adapter.encode_sequence(chunk, initial_snapshot=state_d_0, step_by_step=False)
                _, state_cross_0 = adapter.encode_sequence(chunk, initial_snapshot=state_cross_0, step_by_step=False)

            # Baseline Intact Outputs at N=0
            out_intact_a_0, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_a_0.clone(), step_by_step=False)
            out_intact_b_0, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_b_0.clone(), step_by_step=False)
            z_intact_a_0 = out_intact_a_0[0]
            z_intact_b_0 = out_intact_b_0[0]

            # 2. Compute and Cache Frozen Baseline Axes at N=0
            u_0_a2b, norm_0_a2b = compute_frozen_axis(z_intact_a_0, z_intact_b_0)
            u_0_b2a, norm_0_b2a = compute_frozen_axis(z_intact_b_0, z_intact_a_0)

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
                    # Advance all 5 branches along the single stream across all horizons in a single pass
                    snaps_a = advance_stream_along_horizons(adapter, state_a_0, drive_stream_2048, horizons=horizons, arm=arm)
                    snaps_b = advance_stream_along_horizons(adapter, state_b_0, drive_stream_2048, horizons=horizons, arm=arm)
                    snaps_c = advance_stream_along_horizons(adapter, state_c_0, drive_stream_2048, horizons=horizons, arm=arm)
                    snaps_d = advance_stream_along_horizons(adapter, state_d_0, drive_stream_2048, horizons=horizons, arm=arm)
                    snaps_cross = advance_stream_along_horizons(adapter, state_cross_0, drive_stream_2048, horizons=horizons, arm=arm)

                    for horizon in horizons:
                        state_a_N = snaps_a[horizon]
                        state_b_N = snaps_b[horizon]
                        state_c_N = snaps_c[horizon]
                        state_d_N = snaps_d[horizon]
                        state_cross_N = snaps_cross[horizon]

                        # Baseline Intact Outputs at horizon N
                        out_intact_a_N, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_a_N.clone(), step_by_step=False)
                        out_intact_b_N, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_b_N.clone(), step_by_step=False)
                        z_intact_a_N = out_intact_a_N[0]
                        z_intact_b_N = out_intact_b_N[0]

                        # Contemporaneous axes at horizon N
                        u_N_a2b, norm_N_a2b = compute_frozen_axis(z_intact_a_N, z_intact_b_N)
                        u_N_b2a, norm_N_b2a = compute_frozen_axis(z_intact_b_N, z_intact_a_N)

                        # Physical RG-LRU divergence
                        dist_ab = float(sum(torch.norm(state_a_N.rglru[l].float() - state_b_N.rglru[l].float()).item() for l in state_a_N.rglru))
                        dist_ac = float(sum(torch.norm(state_a_N.rglru[l].float() - state_c_N.rglru[l].float()).item() for l in state_a_N.rglru))

                        # Build Intervention States
                        # Direction A -> B (Recipient B, Donor A/C/D)
                        state_match_a2b = swap_stores(recipient=state_b_N, donor=state_a_N, channels="rglru")
                        out_match_a2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_match_a2b, step_by_step=False)
                        z_match_a2b = out_match_a2b[0]

                        state_wrong_c2b = swap_stores(recipient=state_b_N, donor=state_c_N, channels="rglru")
                        out_wrong_c2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_wrong_c2b, step_by_step=False)
                        z_wrong_c2b = out_wrong_c2b[0]

                        state_wrong_d2b = swap_stores(recipient=state_b_N, donor=state_d_N, channels="rglru")
                        out_wrong_d2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_wrong_d2b, step_by_step=False)
                        z_wrong_d2b = out_wrong_d2b[0]

                        # Direction B -> A (Recipient A, Donor B/C/D)
                        state_match_b2a = swap_stores(recipient=state_a_N, donor=state_b_N, channels="rglru")
                        out_match_b2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_match_b2a, step_by_step=False)
                        z_match_b2a = out_match_b2a[0]

                        state_wrong_c2a = swap_stores(recipient=state_a_N, donor=state_c_N, channels="rglru")
                        out_wrong_c2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_wrong_c2a, step_by_step=False)
                        z_wrong_c2a = out_wrong_c2a[0]

                        state_wrong_d2a = swap_stores(recipient=state_a_N, donor=state_d_N, channels="rglru")
                        out_wrong_d2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_wrong_d2a, step_by_step=False)
                        z_wrong_d2a = out_wrong_d2a[0]

                        eval_conditions_a2b = [
                            ("intact_b", z_intact_b_N, "none"),
                            ("matching_rglru_a_into_b", z_match_a2b, "A"),
                            ("same_template_wrong_c_into_b", z_wrong_c2b, "C"),
                            ("same_template_wrong_d_into_b", z_wrong_d2b, "D"),
                        ]

                        eval_conditions_b2a = [
                            ("intact_a", z_intact_a_N, "none"),
                            ("matching_rglru_b_into_a", z_match_b2a, "B"),
                            ("same_template_wrong_c_into_a", z_wrong_c2a, "C"),
                            ("same_template_wrong_d_into_a", z_wrong_d2a, "D"),
                        ]

                        # Add secondary reference controls at N=0 and N=2048 endpoints
                        if horizon in (0, 2048):
                            state_cross_a2b = swap_stores(recipient=state_b_N, donor=state_cross_N, channels="rglru")
                            out_cross_a2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_cross_a2b, step_by_step=False)
                            z_cross_a2b = out_cross_a2b[0]

                            state_noise_a2b = add_intervention_matched_noise(recipient=state_b_N, donor=state_a_N, channel="rglru", seed=cur_seed + 10)
                            out_noise_a2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_noise_a2b, step_by_step=False)
                            z_noise_a2b = out_noise_a2b[0]

                            out_whole_a2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_a_N.clone(), step_by_step=False)
                            z_whole_a2b = out_whole_a2b[0]

                            eval_conditions_a2b.extend([
                                ("cross_template_e_into_b", z_cross_a2b, "cross"),
                                ("noise_rglru_a_into_b", z_noise_a2b, "noise"),
                                ("whole_swap_a_into_b", z_whole_a2b, "A"),
                            ])

                            state_cross_b2a = swap_stores(recipient=state_a_N, donor=state_cross_N, channels="rglru")
                            out_cross_b2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_cross_b2a, step_by_step=False)
                            z_cross_b2a = out_cross_b2a[0]

                            state_noise_b2a = add_intervention_matched_noise(recipient=state_a_N, donor=state_b_N, channel="rglru", seed=cur_seed + 11)
                            out_noise_b2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_noise_b2a, step_by_step=False)
                            z_noise_b2a = out_noise_b2a[0]

                            out_whole_b2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=state_b_N.clone(), step_by_step=False)
                            z_whole_b2a = out_whole_b2a[0]

                            eval_conditions_b2a.extend([
                                ("cross_template_e_into_a", z_cross_b2a, "cross"),
                                ("noise_rglru_b_into_a", z_noise_b2a, "noise"),
                                ("whole_swap_b_into_a", z_whole_b2a, "B"),
                            ])

                        # Record Direction A -> B
                        for c_name, z_int, don_label in eval_conditions_a2b:
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
                                "physical_dist_ab": dist_ab,
                                "physical_dist_ac": dist_ac,
                                **m,
                            }
                            f_trace.write(json.dumps(rec) + "\n")
                            records_written += 1

                        # Record Direction B -> A
                        for c_name, z_int, don_label in eval_conditions_b2a:
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
                                "physical_dist_ab": dist_ab,
                                "physical_dist_ac": dist_ac,
                                **m,
                            }
                            f_trace.write(json.dumps(rec) + "\n")
                            records_written += 1

            f_trace.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pair_dur = (datetime.datetime.now() - pair_start).total_seconds()
            print(f"[E13] Pair {p_idx+1:02d}/{len(pairs)} ({pair.pair_id}) complete in {pair_dur:.1f}s ({records_written} total rows)")

    total_elapsed = (datetime.datetime.now() - start_time).total_seconds()

    summary_data = {
        "phase": phase,
        "timestamp": timestamp,
        "elapsed_seconds": total_elapsed,
        "total_records": records_written,
        "num_pairs": len(pairs),
        "num_families": len(MICROSCOPE_FAMILIES),
        "base_lag": base_lag,
        "horizons": horizons,
        "regimes": regimes,
        "arms": arms,
        "token_clock_audit": "PASSED (T_theta^(0)(S) = S)",
        "panel_hash": panel_hash,
        "protocol_code_sha256": protocol_code_sha,
        "git_provenance": {
            "commit_sha": git_sha,
            "is_clean": is_clean,
        },
        "model_provenance": model_provenance,
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.startswith("cuda") else "CPU",
        },
    }

    with open(summary_file, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2)

    print(f"[E13] Complete! Recorded {records_written} dynamics condition records in {total_elapsed:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint S13 Controlled Recurrent Dynamics Runner")
    parser.add_argument("--phase", choices=["scout", "confirmatory"], default="scout")
    parser.add_argument("--model_id", type=str, default="google/recurrentgemma-2b")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        phase=args.phase,
        model_id=args.model_id,
        dtype_str=args.dtype,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
    )
