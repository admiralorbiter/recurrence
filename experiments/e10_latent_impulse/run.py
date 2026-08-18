"""Experiment E10: Latent Impulse Response & Store Localization Runner (Sprint S11 Hardened).

Executes matched-trajectory impulse responses across 4 input-dependent filler regimes
over a dynamically computed architectural lag grid with fail-closed provenance and per-layer persistence.
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
from recurrence.tasks.impulse_stimuli import (
    CANONICAL_STIMULI_PAIRS,
    ImpulseStimulusPair,
    build_audited_vocabulary_pool,
)
from recurrence.loop.latent_impulse_harness import (
    generate_dynamic_lag_grid,
    evaluate_impulse_trajectory,
)


def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    """Parse string dtype into torch dtype."""
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Choose from: {list(mapping.keys())}")
    return mapping[dtype_str]


def run_experiment(
    phase: str = "scout",
    regimes: Optional[List[str]] = None,
    num_pairs: int = 4,
    num_seeds: int = 1,
    model_id: Optional[str] = None,
    dtype: str = "bfloat16",
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 42,
) -> Path:
    selected_regimes = regimes or ["constant", "random", "natural", "interfering"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_torch_dtype(dtype)

    if output_dir is None:
        run_name = f"run_e10_{phase}_{timestamp}"
        out_path = Path("results") / "e10_latent_impulse" / run_name
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    snapshot_dir = out_path / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[E10] Initializing {phase.upper()} Phase on device={target_device}, dtype={dtype}...")

    tokenizer = None
    adapter = None
    is_reference_model = False
    model_provenance: Dict[str, Any] = {}

    if model_id is not None or phase == "confirmatory":
        target_model_id = model_id or "google/recurrentgemma-2b"
        print(f"[E10] Loading required pretrained model '{target_model_id}' (FAIL-CLOSED)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(target_model_id)
            model = RecurrentGemmaForCausalLM.from_pretrained(target_model_id, torch_dtype=torch_dtype)
            adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=target_device, dtype=torch_dtype)
            model_provenance = {
                "model_id": target_model_id,
                "is_reference_model": False,
                "model_class": model.__class__.__name__,
                "vocab_size": len(tokenizer),
                "num_hidden_layers": model.config.num_hidden_layers,
                "hidden_size": model.config.hidden_size,
                "conv1d_width": getattr(model.config, "conv1d_width", 4),
                "attention_window_size": getattr(model.config, "attention_window_size", getattr(model.config, "sliding_window", 2048)),
            }
        except Exception as e:
            err_msg = (
                f"FATAL: Failed to load required model '{target_model_id}': {e}\n"
                f"For gated repositories like google/recurrentgemma-2b, verify access at https://huggingface.co/google/recurrentgemma-2b."
            )
            print(f"[E10] {err_msg}")
            raise RuntimeError(err_msg) from e
    else:
        # Explicit lightweight reference model scout
        print("[E10] Running reference-model engineering scout (random weights).")
        is_reference_model = True
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

    # Audit vocabulary pool
    audited_pool, pool_hash = build_audited_vocabulary_pool(tokenizer)
    print(f"[E10] Audited Neutral Vocabulary Pool Size: {len(audited_pool)} (SHA256: {pool_hash})")

    lag_grid = generate_dynamic_lag_grid(adapter.config)
    print(f"[E10] Dynamic Lag Grid (len={len(lag_grid)}): {lag_grid}")

    pairs_to_run = CANONICAL_STIMULI_PAIRS[:num_pairs]
    print(f"[E10] Running {len(pairs_to_run)} stimulus pairs x {len(selected_regimes)} regimes x {num_seeds} seeds...")

    trace_file = out_path / "state_trace.jsonl"
    layer_trace_file = out_path / "layer_trace.jsonl"
    
    total_records = 0
    total_layer_records = 0
    start_time = time.time()
    summary_records = []

    with open(trace_file, "w", encoding="utf-8") as f_trace, open(layer_trace_file, "w", encoding="utf-8") as f_layer:
        for pair_idx, pair in enumerate(pairs_to_run):
            for regime in selected_regimes:
                for s_idx in range(num_seeds):
                    cur_seed = seed + pair_idx * 100 + s_idx
                    records, layer_records = evaluate_impulse_trajectory(
                        adapter=adapter,
                        pair=pair,
                        regime=regime,
                        lag_grid=lag_grid,
                        seed=cur_seed,
                        tokenizer=tokenizer,
                        audited_pool=audited_pool,
                    )

                    for rec in records:
                        row = {
                            "pair_id": pair.pair_id,
                            "regime": regime,
                            "seed_idx": s_idx,
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

                    for l_rec in layer_records:
                        l_row = {
                            "pair_id": l_rec.pair_id,
                            "regime": l_rec.regime,
                            "seed_idx": s_idx,
                            "lag": l_rec.lag,
                            "channel": l_rec.channel,
                            "layer_idx": l_rec.layer_idx,
                            "rmsdiff": round(l_rec.rmsdiff, 6),
                            "scale_relative_dist": round(l_rec.scale_relative_dist, 6),
                            "cossim": round(l_rec.cossim, 6),
                            "frobenius": round(l_rec.frobenius, 6),
                            "retention_ratio": round(l_rec.retention_ratio, 6),
                        }
                        f_layer.write(json.dumps(l_row) + "\n")
                        total_layer_records += 1

    elapsed = time.time() - start_time
    print(f"[E10] Complete! Wrote {total_records} summary rows and {total_layer_records} layer rows in {elapsed:.2f}s.")

    summary = {
        "phase": phase,
        "timestamp": timestamp,
        "elapsed_seconds": round(elapsed, 2),
        "total_summary_records": total_records,
        "total_layer_records": total_layer_records,
        "num_pairs": len(pairs_to_run),
        "num_seeds": num_seeds,
        "regimes": selected_regimes,
        "lag_grid": lag_grid,
        "audited_vocab_pool_hash": pool_hash,
        "audited_vocab_pool_size": len(audited_pool),
        "environment": {
            "torch_version": torch.__version__,
            "transformers_version": "5.15.0",
            "device": str(target_device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and "cuda" in str(target_device) else "CPU",
            "dtype": str(torch_dtype),
        },
        "model_provenance": model_provenance,
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
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    regimes_list = [r.strip() for r in args.regimes.split(",")]
    run_experiment(
        phase=args.phase,
        regimes=regimes_list,
        num_pairs=args.num_pairs,
        num_seeds=args.num_seeds,
        model_id=args.model_id,
        dtype=args.dtype,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
