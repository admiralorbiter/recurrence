"""Experiment E11 Runner: Multi-Store Surgical State Swaps (Sprint S12).

Executes surgical channel interventions across target lag checkpoints (L=8, L=2049, L=4096)
under scout and fail-closed confirmatory protocols on RecurrentGemma.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from recurrence.loop.surgical_swap_harness import (
    evaluate_surgical_swaps,
    evaluate_mediational_propagation,
    get_balanced_donor_pairs,
)
from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import (
    CANONICAL_STIMULI_PAIRS,
    build_audited_vocabulary_pool,
)


S12B_CONFIRMATORY_PROTOCOL: Dict[str, Any] = {
    "protocol_version": "S12b-Confirmatory-v1.0",
    "required_model_id": "google/recurrentgemma-2b",
    "required_dtype": "bfloat16",
    "required_device": "cuda",
    "required_num_pairs": 20,
    "required_regimes": ["constant", "interfering", "natural", "random"],
    "required_target_lags": [8, 2049, 4096],
    "required_mediation_horizons": [512, 2048],
    "eligibility_threshold": 0.5,
    "bootstrap_replicates": 10000,
    "cluster_unit": "pair_id",
    "conditioning": "frozen filler panel / deterministic seed assignment",
}


def get_git_provenance() -> Dict[str, Any]:
    """Capture Git commit SHA and cleanliness of worktree."""
    try:
        commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status_out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        is_clean = (len(status_out) == 0)
        return {
            "commit_sha": commit_sha,
            "is_clean": is_clean,
            "dirty_files": status_out.splitlines() if status_out else [],
        }
    except Exception as e:
        return {"commit_sha": "unknown", "is_clean": False, "error": str(e)}


def compute_protocol_code_hash() -> str:
    """Compute SHA-256 hash over the active S12 intervention and analysis code files."""
    hasher = hashlib.sha256()
    paths = [
        Path(__file__),
        Path(__file__).parent.parent.parent / "src" / "recurrence" / "loop" / "surgical_swap_harness.py",
        Path(__file__).parent.parent.parent / "src" / "recurrence" / "interventions" / "surgical_swaps.py",
        Path(__file__).parent / "analyze.py",
    ]
    for p in sorted(paths):
        if p.exists():
            hasher.update(p.read_bytes())
    return hasher.hexdigest()


def compute_donor_map_hash(pairs: List[Any]) -> Tuple[Dict[str, Dict[str, str]], str]:
    """Compute balanced cyclic derangements and return deterministic mapping with SHA-256 hash."""
    mapping = {}
    for p in pairs:
        unrel, perm = get_balanced_donor_pairs(pairs, p)
        mapping[p.pair_id] = {
            "unrelated_donor_pair_id": unrel.pair_id,
            "permuted_donor_pair_id": perm.pair_id,
        }
    map_str = json.dumps(mapping, sort_keys=True)
    map_hash = hashlib.sha256(map_str.encode("utf-8")).hexdigest()
    return mapping, map_hash


def run_experiment(
    phase: str = "scout",
    model_id: str = "google/recurrentgemma-2b",
    dtype_str: str = "bfloat16",
    device: str = "cuda",
    num_pairs: Optional[int] = None,
    regimes: Optional[List[str]] = None,
    target_lags: Optional[List[int]] = None,
    seed: int = 42,
    output_dir: str = "results/e11_surgical_swaps",
) -> Path:
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_e11_{phase}_{timestamp}"
    out_path = Path(output_dir) / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[E11] Initializing {phase.upper()} Phase on device={device}, dtype={dtype_str}...")

    # Strict Fail-Closed Environment Validation for Confirmatory Phase
    if phase == "confirmatory":
        assert model_id == "google/recurrentgemma-2b", (
            f"[Fail-Closed Gate] Confirmatory run requires google/recurrentgemma-2b, got '{model_id}'"
        )
        assert dtype_str == "bfloat16", (
            f"[Fail-Closed Gate] Confirmatory run requires bfloat16, got '{dtype_str}'"
        )
        assert device.startswith("cuda"), (
            f"[Fail-Closed Gate] Confirmatory run requires CUDA, got '{device}'"
        )
        assert torch.cuda.is_available(), (
            "[Fail-Closed Gate] CUDA must be available for confirmatory run"
        )

    # Configure precision
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # 1. Load Model (Fail-Closed)
    adapter = None
    tokenizer = None
    model_provenance = {}

    if model_id == "reference_model":
        if phase == "confirmatory":
            raise ValueError("[E11 Fail-Closed Gate] Confirmatory S12b cannot run on reference model!")
        print("[E11] Initializing lightweight Reference Model for local dry-run...")
        from transformers import RecurrentGemmaConfig
        config = RecurrentGemmaConfig(
            num_hidden_layers=4,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            lru_width=64,
            conv1d_width=4,
            sliding_window=8,
            vocab_size=200,
        )
        torch.manual_seed(seed)
        adapter = RecurrentGemmaAdapter(config=config, device=device, dtype=torch.float32)
        model_provenance = {
            "model_id": "reference_model",
            "is_reference_model": True,
            "hidden_size": 64,
            "conv1d_width": 4,
            "attention_window_size": 8,
            "model_revision": "local_mock",
        }
    else:
        print(f"[E11] Loading required pretrained model '{model_id}' (FAIL-CLOSED)...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )
            adapter = RecurrentGemmaAdapter(model=model, device=device, dtype=dtype)
            model_commit = getattr(model.config, "_commit_hash", None) or getattr(tokenizer, "_commit_hash", "resolved_hub_head")
            model_provenance = {
                "model_id": model_id,
                "is_reference_model": False,
                "model_revision": model_commit,
                "model_class": model.__class__.__name__,
                "vocab_size": getattr(model.config, "vocab_size", None),
                "num_hidden_layers": getattr(model.config, "num_hidden_layers", 26),
                "hidden_size": getattr(model.config, "hidden_size", 2560),
                "conv1d_width": getattr(model.config, "conv1d_width", 4),
                "attention_window_size": getattr(model.config, "sliding_window", 2048),
            }
        except Exception as e:
            raise RuntimeError(f"[E11 Fail-Closed Gate] Pretrained model load failed: {e}") from e

    # 2. Select Stimulus Panel & Regimes
    if phase == "confirmatory":
        # FAIL-CLOSED CONFIRMATORY PROTOCOL VALIDATION
        pairs_to_run = CANONICAL_STIMULI_PAIRS
        selected_regimes = S12B_CONFIRMATORY_PROTOCOL["required_regimes"]
        lags_to_eval = S12B_CONFIRMATORY_PROTOCOL["required_target_lags"]
        mediation_horizons = S12B_CONFIRMATORY_PROTOCOL["required_mediation_horizons"]

        assert len(pairs_to_run) == S12B_CONFIRMATORY_PROTOCOL["required_num_pairs"], (
            f"Confirmatory run must use exactly 20 stimulus pairs, got {len(pairs_to_run)}"
        )
        assert set(selected_regimes) == set(S12B_CONFIRMATORY_PROTOCOL["required_regimes"]), (
            "Confirmatory run must evaluate all 4 regimes: constant, interfering, natural, random"
        )
        assert lags_to_eval == S12B_CONFIRMATORY_PROTOCOL["required_target_lags"], (
            "Confirmatory run must evaluate lags [8, 2049, 4096]"
        )
    else:
        # Exploratory Scout
        n_pairs = num_pairs if num_pairs is not None else 4
        pairs_to_run = CANONICAL_STIMULI_PAIRS[:n_pairs]
        selected_regimes = regimes if regimes is not None else ["constant", "interfering", "natural", "random"]
        lags_to_eval = target_lags if target_lags is not None else ([0, 2, 8] if model_provenance.get("is_reference_model") else [8, 2049, 4096])
        mediation_horizons = [512] if model_provenance.get("is_reference_model") else [512]

    donor_mapping, donor_map_hash = compute_donor_map_hash(pairs_to_run)
    git_prov = get_git_provenance()
    protocol_code_sha = compute_protocol_code_hash()

    # Audited pool
    audited_pool = None
    audited_pool_hash = None
    if tokenizer is not None:
        audited_pool, _ = build_audited_vocabulary_pool(tokenizer)
        audited_pool_hash = hashlib.sha256(json.dumps(audited_pool).encode("utf-8")).hexdigest()

    # 3. Execute Surgical Swaps Factorial
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
                    all_pairs=pairs_to_run,
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
                        "signed_graft_effect": rec.signed_graft_effect,
                        "directional_displacement": rec.directional_displacement,
                        "donor_recipient_norm": rec.donor_recipient_norm,
                        "logit_directional_projection": rec.logit_directional_projection,
                        "is_eligible_for_attribution": rec.is_eligible_for_attribution,
                        "causal_attribution_index": rec.causal_attribution_index,
                    }
                    f_trace.write(json.dumps(row) + "\n")
                    all_records.append(row)

    # 4. Execute Mediational Forward Dynamic Propagation
    med_file = out_path / "mediational_propagation.jsonl"
    all_med_records = []
    with open(med_file, "w", encoding="utf-8") as f_med:
        for pair_idx, pair in enumerate(pairs_to_run):
            for regime in selected_regimes:
                for horizon in mediation_horizons:
                    cur_seed = seed + pair_idx * 100
                    is_ref = model_provenance.get("is_reference_model", False)
                    med_res = evaluate_mediational_propagation(
                        adapter=adapter,
                        pair=pair,
                        regime=regime,
                        initial_lag=8 if not is_ref else 2,
                        future_tokens=horizon if not is_ref else 8,
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
        "mediation_horizons": mediation_horizons,
        "donor_mapping": donor_mapping,
        "donor_mapping_sha256": donor_map_hash,
        "audited_pool_sha256": audited_pool_hash,
        "protocol_code_sha256": protocol_code_sha,
        "git_provenance": git_prov,
        "protocol": S12B_CONFIRMATORY_PROTOCOL if phase == "confirmatory" else {"phase": "scout"},
        "model_provenance": model_provenance,
        "environment": {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    }

    with open(out_path / "summary.json", "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run E11 Multi-Store Surgical State Swaps (Sprint S12)")
    parser.add_argument("--phase", type=str, default="scout", choices=["scout", "confirmatory"])
    parser.add_argument("--model_id", type=str, default="google/recurrentgemma-2b")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_pairs", type=int, default=None)
    parser.add_argument("--regimes", type=str, default=None)
    parser.add_argument("--target_lags", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/e11_surgical_swaps")

    args = parser.parse_args()

    regimes_list = [r.strip() for r in args.regimes.split(",")] if args.regimes else None
    lags_list = [int(l.strip()) for l in args.target_lags.split(",")] if args.target_lags else None

    run_experiment(
        phase=args.phase,
        model_id=args.model_id,
        dtype_str=args.dtype,
        device=args.device,
        num_pairs=args.num_pairs,
        regimes=regimes_list,
        target_lags=lags_list,
        seed=args.seed,
        output_dir=args.output_dir,
    )
