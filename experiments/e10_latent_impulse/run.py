"""Experiment E10: Latent Impulse Response & Store Localization Runner (Sprint S11).

Executes matched-trajectory impulse responses across 4 input-dependent filler regimes
over a dynamically computed architectural lag grid.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, RecurrentGemmaConfig, RecurrentGemmaForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import CANONICAL_STIMULI_PAIRS, ImpulseStimulusPair
from recurrence.loop.latent_impulse_harness import (
    generate_dynamic_lag_grid,
    evaluate_impulse_trajectory,
)


def run_experiment(
    phase: str = "scout",
    regimes: Optional[List[str]] = None,
    num_pairs: int = 4,
    model_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cpu",
    seed: int = 42,
) -> Path:
    selected_regimes = regimes or ["constant", "random", "natural", "interfering"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    if output_dir is None:
        run_name = f"run_e10_{phase}_{timestamp}"
        out_path = Path("results") / "e10_latent_impulse" / run_name
    else:
        out_path = Path(output_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)
    snapshot_dir = out_path / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[E10] Initializing {phase.upper()} Phase on device={device}...")

    tokenizer = None
    adapter = None

    if model_id is not None:
        try:
            print(f"[E10] Loading pretrained model '{model_id}'...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = RecurrentGemmaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
            adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=device)
        except Exception as e:
            print(f"[E10] Warning: Could not load '{model_id}' ({e}). Falling back to reference config.")

    if adapter is None:
        print("[E10] Using reference RecurrentGemma configuration for execution.")
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

    lag_grid = generate_dynamic_lag_grid(adapter.config)
    print(f"[E10] Dynamic Lag Grid (len={len(lag_grid)}): {lag_grid}")

    pairs_to_run = CANONICAL_STIMULI_PAIRS[:num_pairs]
    print(f"[E10] Running {len(pairs_to_run)} stimulus pairs across {len(selected_regimes)} regimes...")

    trace_file = out_path / "state_trace.jsonl"
    total_records = 0
    start_time = time.time()

    summary_records = []

    with open(trace_file, "w", encoding="utf-8") as f_trace:
        for pair_idx, pair in enumerate(pairs_to_run):
            for regime in selected_regimes:
                print(f"  -> Pair [{pair.pair_id}] | Regime [{regime}]")
                records = evaluate_impulse_trajectory(
                    adapter=adapter,
                    pair=pair,
                    regime=regime,
                    lag_grid=lag_grid,
                    seed=seed + pair_idx,
                    tokenizer=tokenizer,
                )

                for rec in records:
                    row = {
                        "pair_id": pair.pair_id,
                        "regime": regime,
                        "lag": rec.lag,
                        "conv_directly_resident": rec.conv_directly_resident,
                        "kv_directly_resident": rec.kv_directly_resident,
                        "mean_rglru_d_rel": round(rec.mean_rglru_d_rel, 6),
                        "mean_conv_d_rel": round(rec.mean_conv_d_rel, 6),
                        "mean_kv_d_rel": round(rec.mean_kv_d_rel, 6),
                        "mean_rglru_retention": round(rec.mean_rglru_retention, 6),
                        "mean_conv_retention": round(rec.mean_conv_retention, 6),
                        "mean_kv_retention": round(rec.mean_kv_retention, 6),
                        "jensen_shannon_div": round(rec.jensen_shannon_div, 6),
                        "top1_disagreement": rec.top1_disagreement,
                        "twoway_2afc_margin": round(rec.twoway_2afc_margin, 6),
                        "twoway_2afc_accuracy": round(rec.twoway_2afc_accuracy, 4),
                        "sham_mean_d_rel": round(rec.sham_mean_d_rel, 8),
                        "sham_jensen_shannon_div": round(rec.sham_jensen_shannon_div, 8),
                    }
                    f_trace.write(json.dumps(row) + "\n")
                    summary_records.append(row)
                    total_records += 1

    elapsed = time.time() - start_time
    print(f"[E10] Complete! Wrote {total_records} checkpoint rows in {elapsed:.2f}s.")

    summary = {
        "phase": phase,
        "timestamp": timestamp,
        "elapsed_seconds": round(elapsed, 2),
        "total_records": total_records,
        "num_pairs": len(pairs_to_run),
        "regimes": selected_regimes,
        "lag_grid": lag_grid,
        "model_config": {
            "num_hidden_layers": adapter.config.num_hidden_layers,
            "hidden_size": adapter.config.hidden_size,
            "conv1d_width": getattr(adapter.config, "conv1d_width", 4),
            "sliding_window": getattr(adapter.config, "sliding_window", 2048),
        },
        "mean_sham_floor_d_rel": sum(r["sham_mean_d_rel"] for r in summary_records) / max(len(summary_records), 1),
    }

    with open(out_path / "summary.json", "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E10 Latent Impulse Response Experiment")
    parser.add_argument("--phase", type=str, default="scout", choices=["scout", "confirmatory"])
    parser.add_argument("--regimes", type=str, default="constant,random,natural,interfering")
    parser.add_argument("--num_pairs", type=int, default=4)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    regimes_list = [r.strip() for r in args.regimes.split(",")]
    run_experiment(
        phase=args.phase,
        regimes=regimes_list,
        num_pairs=args.num_pairs,
        model_id=args.model_id,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
