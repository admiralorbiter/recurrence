"""Sprint S12c: Specificity Microscope Runner.

Evaluates value-specific vs same-template vs cross-template recurrent state steering
at 2W = 4096 tokens on google/recurrentgemma-2b (bfloat16, CUDA).
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter, RecurrentGemmaConfig
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
from recurrence.interventions.surgical_swaps import swap_stores, add_intervention_matched_noise


def compute_directional_metrics(
    z_intact_rec: torch.Tensor,
    z_intact_don: torch.Tensor,
    z_intervened: torch.Tensor,
    tok_rec_id: int,
    tok_don_id: int,
) -> Dict[str, float]:
    """Compute normalized directional displacement and cloze metrics."""
    diff_d = (z_intact_don - z_intact_rec).flatten().float()
    diff_g = (z_intervened - z_intact_rec).flatten().float()
    norm_d = float(torch.norm(diff_d).item())
    if norm_d < 1e-6:
        return {
            "directional_displacement": 0.0,
            "logit_directional_projection": 0.0,
            "donor_cloze_margin": 0.0,
            "donor_is_top1": False,
        }

    unit_d = diff_d / norm_d
    dir_disp = float(torch.sum(diff_g * unit_d).item())
    proj = float(dir_disp / norm_d)

    logit_rec = z_intervened[tok_rec_id].item()
    logit_don = z_intervened[tok_don_id].item()
    cloze_margin = float(logit_don - logit_rec)

    return {
        "directional_displacement": dir_disp,
        "logit_directional_projection": proj,
        "donor_cloze_margin": cloze_margin,
        "donor_is_top1": bool(torch.argmax(z_intervened).item() == tok_don_id),
    }


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
        run_name = f"run_e12_{phase}_{timestamp}"
        out_path = Path("results") / "e12_specificity_microscope" / run_name
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    trace_file = out_path / "microscope_trace.jsonl"
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
        fpath = Path("experiments") / "e12_specificity_microscope" / fname
        if fpath.exists():
            code_hasher.update(fpath.read_bytes())
    for frel in ["tasks/specificity_microscope.py", "loop/surgical_swaps.py"]:
        fpath = Path("src") / "recurrence" / frel
        if fpath.exists():
            code_hasher.update(fpath.read_bytes())
    protocol_code_sha = code_hasher.hexdigest()

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    target_lag = 4096
    regimes = ["constant", "interfering", "natural", "random"]

    print(f"[E12c] Initializing {phase.upper()} Specificity Microscope on device={device}, dtype={dtype_str}...")

    if phase == "confirmatory":
        assert model_id == "google/recurrentgemma-2b", f"Confirmatory requires google/recurrentgemma-2b, got {model_id}"
        assert dtype_str == "bfloat16", f"Confirmatory requires bfloat16, got {dtype_str}"
        assert device.startswith("cuda"), f"Confirmatory requires cuda, got {device}"
        assert torch.cuda.is_available(), "CUDA is not available"

    tokenizer = None
    if model_id is None or model_id == "reference_random":
        print("[E12c] Running lightweight reference model scout.")
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
        print(f"[E12c] Loading pretrained model '{model_id}' (FAIL-CLOSED)...")
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
            raise RuntimeError(f"[E12c Fail-Closed Gate] Pretrained model load failed: {e}") from e

    # Audit microscope panel
    is_valid, panel_hash, panel_audit = audit_microscope_panel(adapter.tokenizer)
    print(f"[E12c] Specificity Microscope Panel Audit: {panel_audit['status']} (SHA256: {panel_hash})")
    if phase == "confirmatory" and not is_valid:
        raise ValueError(f"[E12c Fail-Closed Gate] Panel audit failed: {panel_audit['status']}")

    # Audited pool
    audited_pool = None
    if adapter.tokenizer is not None:
        audited_pool, _ = build_audited_vocabulary_pool(adapter.tokenizer)

    pairs = build_microscope_pairs()
    print(f"[E12c] Built {len(pairs)} canonical pairs across 4 template families.")

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

            for reg in regimes:
                cur_seed = seed + p_idx * 100
                filler_tokens = get_filler_tokens_for_regime(
                    regime=reg,
                    length=target_lag,
                    seed=cur_seed,
                    audited_pool=audited_pool,
                    tokenizer=adapter.tokenizer,
                    excluded_token_ids=pair_excluded,
                )

                # Initialize states
                state_a = adapter.init_state()
                state_b = adapter.init_state()
                state_c = adapter.init_state()
                state_d = adapter.init_state()
                state_cross = adapter.init_state()

                # Process initial prompts
                state_a, _ = adapter.step_chunk(state_a, toks_prompt_a)
                state_b, _ = adapter.step_chunk(state_b, toks_prompt_b)
                state_c, _ = adapter.step_chunk(state_c, toks_prompt_c)
                state_d, _ = adapter.step_chunk(state_d, toks_prompt_d)
                state_cross, _ = adapter.step_chunk(state_cross, toks_prompt_cross)

                # Process filler chunk-by-chunk to target lag
                chunk_size = 512
                for i in range(0, target_lag, chunk_size):
                    end_idx = min(i + chunk_size, target_lag)
                    chunk = filler_tokens[i:end_idx]
                    state_a, _ = adapter.step_chunk(state_a, chunk)
                    state_b, _ = adapter.step_chunk(state_b, chunk)
                    state_c, _ = adapter.step_chunk(state_c, chunk)
                    state_d, _ = adapter.step_chunk(state_d, chunk)
                    state_cross, _ = adapter.step_chunk(state_cross, chunk)

                # 1. Baseline Intact Outputs
                _, out_intact_a = adapter.step_chunk(state_a.clone(), toks_query)
                _, out_intact_b = adapter.step_chunk(state_b.clone(), toks_query)
                z_intact_a = out_intact_a.logits[0, -1]
                z_intact_b = out_intact_b.logits[0, -1]

                # 2. Direction A -> B Interventions (Recipient B, Donor A/C/D/Cross)
                state_match_a_into_b = swap_stores(recipient=state_b, donor=state_a, channels="rglru")
                _, out_match_a_into_b = adapter.step_chunk(state_match_a_into_b, toks_query)
                z_match_a_into_b = out_match_a_into_b.logits[0, -1]

                state_wrong_c_into_b = swap_stores(recipient=state_b, donor=state_c, channels="rglru")
                _, out_wrong_c_into_b = adapter.step_chunk(state_wrong_c_into_b, toks_query)
                z_wrong_c_into_b = out_wrong_c_into_b.logits[0, -1]

                state_wrong_d_into_b = swap_stores(recipient=state_b, donor=state_d, channels="rglru")
                _, out_wrong_d_into_b = adapter.step_chunk(state_wrong_d_into_b, toks_query)
                z_wrong_d_into_b = out_wrong_d_into_b.logits[0, -1]

                state_cross_into_b = swap_stores(recipient=state_b, donor=state_cross, channels="rglru")
                _, out_cross_into_b = adapter.step_chunk(state_cross_into_b, toks_query)
                z_cross_into_b = out_cross_into_b.logits[0, -1]

                state_noise_a_into_b = add_intervention_matched_noise(
                    recipient=state_b, donor=state_a, channels="rglru", seed=cur_seed + 10
                )
                _, out_noise_a_into_b = adapter.step_chunk(state_noise_a_into_b, toks_query)
                z_noise_a_into_b = out_noise_a_into_b.logits[0, -1]

                _, out_whole_a_into_b = adapter.step_chunk(state_a.clone(), toks_query)
                z_whole_a_into_b = out_whole_a_into_b.logits[0, -1]

                # 3. Direction B -> A Interventions (Recipient A, Donor B/C/D/Cross)
                state_match_b_into_a = swap_stores(recipient=state_a, donor=state_b, channels="rglru")
                _, out_match_b_into_a = adapter.step_chunk(state_match_b_into_a, toks_query)
                z_match_b_into_a = out_match_b_into_a.logits[0, -1]

                state_wrong_c_into_a = swap_stores(recipient=state_a, donor=state_c, channels="rglru")
                _, out_wrong_c_into_a = adapter.step_chunk(state_wrong_c_into_a, toks_query)
                z_wrong_c_into_a = out_wrong_c_into_a.logits[0, -1]

                state_wrong_d_into_a = swap_stores(recipient=state_a, donor=state_d, channels="rglru")
                _, out_wrong_d_into_a = adapter.step_chunk(state_wrong_d_into_a, toks_query)
                z_wrong_d_into_a = out_wrong_d_into_a.logits[0, -1]

                state_cross_into_a = swap_stores(recipient=state_a, donor=state_cross, channels="rglru")
                _, out_cross_into_a = adapter.step_chunk(state_cross_into_a, toks_query)
                z_cross_into_a = out_cross_into_a.logits[0, -1]

                state_noise_b_into_a = add_intervention_matched_noise(
                    recipient=state_a, donor=state_b, channels="rglru", seed=cur_seed + 11
                )
                _, out_noise_b_into_a = adapter.step_chunk(state_noise_b_into_a, toks_query)
                z_noise_b_into_a = out_noise_b_into_a.logits[0, -1]

                _, out_whole_b_into_a = adapter.step_chunk(state_b.clone(), toks_query)
                z_whole_b_into_a = out_whole_b_into_a.logits[0, -1]

                # Record Direction A -> B conditions
                eval_conditions_a_into_b = [
                    ("intact_b", z_intact_b, False),
                    ("whole_swap_a_into_b", z_whole_a_into_b, False),
                    ("matching_rglru_a_into_b", z_match_a_into_b, True),
                    ("same_template_wrong_c_into_b", z_wrong_c_into_b, False),
                    ("same_template_wrong_d_into_b", z_wrong_d_into_b, False),
                    ("cross_template_e_into_b", z_cross_into_b, False),
                    ("noise_rglru_a_into_b", z_noise_a_into_b, False),
                ]

                for c_name, z_int, is_match in eval_conditions_a_into_b:
                    m = compute_directional_metrics(
                        z_intact_rec=z_intact_b,
                        z_intact_don=z_intact_a,
                        z_intervened=z_int,
                        tok_rec_id=tok_b_id,
                        tok_don_id=tok_a_id,
                    )
                    rec = {
                        "pair_id": pair.pair_id,
                        "family_id": pair.family_id,
                        "val_a": pair.val_a,
                        "val_b": pair.val_b,
                        "regime": reg,
                        "lag": target_lag,
                        "direction": "a_into_b",
                        "recipient": "B",
                        "donor": "A" if is_match else ("C" if "wrong_c" in c_name else ("D" if "wrong_d" in c_name else "cross")),
                        "condition": c_name,
                        **m,
                    }
                    f_trace.write(json.dumps(rec) + "\n")
                    records_written += 1

                # Record Direction B -> A conditions
                eval_conditions_b_into_a = [
                    ("intact_a", z_intact_a, False),
                    ("whole_swap_b_into_a", z_whole_b_into_a, False),
                    ("matching_rglru_b_into_a", z_match_b_into_a, True),
                    ("same_template_wrong_c_into_a", z_wrong_c_into_a, False),
                    ("same_template_wrong_d_into_a", z_wrong_d_into_a, False),
                    ("cross_template_e_into_a", z_cross_into_a, False),
                    ("noise_rglru_b_into_a", z_noise_b_into_a, False),
                ]

                for c_name, z_int, is_match in eval_conditions_b_into_a:
                    m = compute_directional_metrics(
                        z_intact_rec=z_intact_a,
                        z_intact_don=z_intact_b,
                        z_intervened=z_int,
                        tok_rec_id=tok_a_id,
                        tok_don_id=tok_b_id,
                    )
                    rec = {
                        "pair_id": pair.pair_id,
                        "family_id": pair.family_id,
                        "val_a": pair.val_a,
                        "val_b": pair.val_b,
                        "regime": reg,
                        "lag": target_lag,
                        "direction": "b_into_a",
                        "recipient": "A",
                        "donor": "B" if is_match else ("C" if "wrong_c" in c_name else ("D" if "wrong_d" in c_name else "cross")),
                        "condition": c_name,
                        **m,
                    }
                    f_trace.write(json.dumps(rec) + "\n")
                    records_written += 1

            f_trace.flush()
            pair_dur = (datetime.datetime.now() - pair_start).total_seconds()
            print(f"[E12c] Pair {p_idx+1:02d}/24 ({pair.pair_id}) complete across 4 regimes in {pair_dur:.1f}s ({records_written} total rows)")

    total_elapsed = (datetime.datetime.now() - start_time).total_seconds()

    summary_data = {
        "phase": phase,
        "timestamp": timestamp,
        "elapsed_seconds": total_elapsed,
        "total_records": records_written,
        "num_pairs": len(pairs),
        "num_families": len(MICROSCOPE_FAMILIES),
        "regimes": regimes,
        "target_lag": target_lag,
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

    print(f"[E12c] Complete! Recorded {records_written} microscope swap condition results in {total_elapsed:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sprint S12c Specificity Microscope Runner")
    parser.add_argument("--phase", choices=["scout", "confirmatory"], default="confirmatory")
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
