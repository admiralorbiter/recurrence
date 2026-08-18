"""Experiment E11 Runner: Multi-Store Surgical State Swaps (Sprint S12).

Executes causal factorial channel interventions at key lag checkpoints
(e.g., L=8, L=W+1=2049, L=2W=4096) on pretrained RecurrentGemma-2B
to establish causal store attribution.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import List, Optional

import torch
from transformers import AutoTokenizer, RecurrentGemmaForCausalLM, RecurrentGemmaConfig

from recurrence.loop.surgical_swap_harness import evaluate_surgical_swaps
from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import (
    CANONICAL_STIMULI_PAIRS,
    build_audited_vocabulary_pool,
)


def run_experiment(
    phase: str = "scout",
    regimes: Optional[List[str]] = None,
    num_pairs: int = 4,
    target_lags: Optional[List[int]] = None,
    model_id: Optional[str] = None,
    dtype: str = "bfloat16",
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 42,
) -> Path:
    """Execute E11 Multi-Store Surgical State Swaps Experiment."""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    selected_regimes = regimes or ["constant", "interfering", "natural", "random"]
    lags_to_eval = target_lags or [8, 2049, 4096]

    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path(f"results/e11_surgical_swaps/run_e11_{phase}_{timestamp}")
    out_path.mkdir(parents=True, exist_ok=True)

    if device is not None:
        target_device = torch.device(device)
    else:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.bfloat16

    print(f"[E11] Initializing {phase.upper()} Phase on device={target_device}, dtype={torch_dtype}...")

    # Load model
    target_model_id = model_id or ("google/recurrentgemma-2b" if phase == "confirmatory" or target_device.type == "cuda" else None)

    if target_model_id:
        print(f"[E11] Loading required pretrained model '{target_model_id}' (FAIL-CLOSED)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(target_model_id)
            model = RecurrentGemmaForCausalLM.from_pretrained(target_model_id, torch_dtype=torch_dtype)
            adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=target_device, dtype=torch_dtype)
            model_provenance = {
                "model_id": target_model_id,
                "is_reference_model": False,
                "model_class": model.__class__.__name__,
                "commit_hash": getattr(model.config, "_commit_hash", getattr(model.config, "revision", "main")),
                "vocab_size": len(tokenizer),
                "num_hidden_layers": model.config.num_hidden_layers,
                "hidden_size": model.config.hidden_size,
                "conv1d_width": getattr(model.config, "conv1d_width", 4),
                "attention_window_size": getattr(model.config, "attention_window_size", getattr(model.config, "sliding_window", 2048)),
            }
        except Exception as e:
            err_msg = f"FATAL: Failed to load required model '{target_model_id}': {e}"
            print(f"[E11] {err_msg}")
            raise RuntimeError(err_msg) from e
    else:
        print("[E11] Running reference-model engineering scout (random weights).")
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
        adapter = RecurrentGemmaAdapter(config=config, device=target_device, dtype=torch.float32)
        model_provenance = {
            "model_id": "reference_random_recurrentgemma",
            "is_reference_model": True,
            "vocab_size": config.vocab_size,
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "conv1d_width": config.conv1d_width,
            "attention_window_size": getattr(config, "attention_window_size", getattr(config, "sliding_window", 16)),
        }
        lags_to_eval = [2, 8, 16]

    audited_pool, pool_hash = build_audited_vocabulary_pool(tokenizer)
    pairs_to_run = CANONICAL_STIMULI_PAIRS[:num_pairs]

    trace_file = out_path / "swap_trace.jsonl"
    all_records = []

    with open(trace_file, "w", encoding="utf-8") as f_trace:
        for pair_idx, pair in enumerate(pairs_to_run):
            for regime in selected_regimes:
                cur_seed = seed + pair_idx * 100
                records = evaluate_surgical_swaps(
                    adapter=adapter,
                    pair=pair,
                    regime=regime,
                    target_lags=lags_to_eval,
                    seed=cur_seed,
                    tokenizer=tokenizer,
                    audited_pool=audited_pool,
                )
                for rec in records:
                    row = {
                        "pair_id": rec.pair_id,
                        "regime": rec.regime,
                        "lag": rec.lag,
                        "condition": rec.condition,
                        "target_donor": rec.target_donor,
                        "ll_target_a": rec.ll_target_a,
                        "ll_target_b": rec.ll_target_b,
                        "cloze_margin": rec.cloze_margin,
                        "target_choice": rec.target_choice,
                        "raw_graft_effect": rec.raw_graft_effect,
                        "absolute_displacement": rec.absolute_displacement,
                        "donor_recipient_norm": rec.donor_recipient_norm,
                        "logit_directional_projection": rec.logit_directional_projection,
                        "is_eligible_for_attribution": rec.is_eligible_for_attribution,
                        "causal_attribution_index": rec.causal_attribution_index,
                    }
                    f_trace.write(json.dumps(row) + "\n")
                    all_records.append(row)

    # Run mediational forward propagation experiments
    from recurrence.loop.surgical_swap_harness import evaluate_mediational_propagation
    med_file = out_path / "mediational_propagation.jsonl"
    all_med_records = []
    with open(med_file, "w", encoding="utf-8") as f_med:
        for pair_idx, pair in enumerate(pairs_to_run):
            for regime in selected_regimes:
                cur_seed = seed + pair_idx * 100
                is_ref = model_provenance.get("is_reference_model", False)
                med_res = evaluate_mediational_propagation(
                    adapter=adapter,
                    pair=pair,
                    regime=regime,
                    initial_lag=8 if not is_ref else 2,
                    future_tokens=512 if not is_ref else 8,
                    seed=cur_seed,
                    tokenizer=tokenizer,
                    audited_pool=audited_pool,
                )
                f_med.write(json.dumps(med_res) + "\n")
                all_med_records.append(med_res)

    elapsed = time.time() - start_time
    print(f"[E11] Complete! Recorded {len(all_records)} swap condition results and {len(all_med_records)} mediational unrolls in {elapsed:.2f}s.")

    import transformers
    summary = {
        "phase": phase,
        "timestamp": timestamp,
        "elapsed_seconds": round(elapsed, 2),
        "total_swap_records": len(all_records),
        "total_mediational_records": len(all_med_records),
        "num_pairs": len(pairs_to_run),
        "regimes": selected_regimes,
        "target_lags": lags_to_eval,
        "environment": {
            "torch_version": getattr(torch, "__version__", "unknown"),
            "transformers_version": getattr(transformers, "__version__", "unknown"),
            "device": str(target_device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and "cuda" in str(target_device) else "CPU",
            "dtype": str(torch_dtype),
        },
        "model_provenance": model_provenance,
    }

    with open(out_path / "summary.json", "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E11 Multi-Store Surgical Swaps Experiment")
    parser.add_argument("--phase", type=str, default="scout", choices=["scout", "confirmatory"])
    parser.add_argument("--regimes", type=str, default="constant,interfering,natural,random")
    parser.add_argument("--num_pairs", type=int, default=4)
    parser.add_argument("--target_lags", type=str, default="8,2049,4096")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    regimes_list = [r.strip() for r in args.regimes.split(",")]
    lags_list = [int(l.strip()) for l in args.target_lags.split(",")]

    run_experiment(
        phase=args.phase,
        regimes=regimes_list,
        num_pairs=args.num_pairs,
        target_lags=lags_list,
        model_id=args.model_id,
        dtype=args.dtype,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
