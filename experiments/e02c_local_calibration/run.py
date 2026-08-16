"""Experiment E02c: Horizon 0 v2.4 Local 2D Calibration & Held-Out Coordinate Validation.

Performs targeted local 2D (H, D) parameter grid searches to bracket missing calibration
points and runs large held-out validation blocks (N=64) to confirm stability before
the full confirmatory metacognitive observer battery.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from recurrence.backends.ollama import OllamaBackend
from recurrence.core.manifest import RunManifest
from recurrence.core.logging import ExperimentLogger
from recurrence.core.schemas import make_2afc_direct_value_schema
from recurrence.tasks.adaptive_metacognition import (
    AdaptiveMetacognition2AFCTask,
)
from recurrence.analysis.psychophysics import (
    compute_sdt_indices,
    compute_sdt_bootstrap_ci,
    compute_wilson_score_interval,
    compute_type2_sdt_metrics,
    evaluate_calibration_gate,
)


def generate_e02c_markdown_report(
    manifest: Dict[str, Any],
    cell_metrics: Dict[str, Dict[str, Any]],
    mode: str,
) -> str:
    """Generate scientific memo and tabular psychometric report for E02c."""
    target_model = manifest.get("target_model")
    lines = [
        f"# Experiment E02c: Horizon 0 v2.4 Local Calibration & Validation Report",
        f"",
        f"**Run ID:** `{manifest.get('run_id')}`  ",
        f"**Target Model:** `{target_model}` (`{manifest.get('model_digest', 'N/A')[:12]}...`)  ",
        f"**Date:** {manifest.get('start_time', datetime.now(timezone.utc).isoformat())}  ",
        f"**Mode:** `{mode}`  ",
        f"**Response Format:** Matched 2-Alternative Forced Choice (Direct-Value Candidate Strings)  ",
        f"**Total Trials:** {manifest.get('total_trials', 0)}  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Calibration Gate Status",
        f"",
        f"| Coordinate `(H, D)` | Trials | Correct | Accuracy | 95% Wilson CI | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Gate Status | Meta-$d'$ Status | AUROC2 | Brier |",
        f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cell_key, m in cell_metrics.items():
        tot = m["total_trials"]
        cor = m["correct_trials"]
        acc = f"{m['accuracy']:.1%}"
        ci = f"[{m['ci_95_lower']:.1%}, {m['ci_95_upper']:.1%}]"
        
        dp_val = m.get("sdt_d_prime")
        dp_l = m.get("sdt_d_prime_ci_lower")
        dp_u = m.get("sdt_d_prime_ci_upper")
        dp_str = f"{dp_val:+.2f} [{dp_l:+.2f}, {dp_u:+.2f}]" if (dp_val is not None and dp_l is not None) else "N/A"

        sc_val = m.get("sdt_criterion_c")
        sc_l = m.get("sdt_criterion_c_ci_lower")
        sc_u = m.get("sdt_criterion_c_ci_upper")
        sc_str = f"{sc_val:+.2f} [{sc_l:+.2f}, {sc_u:+.2f}]" if (sc_val is not None and sc_l is not None) else "N/A"

        gate_res = m.get("calibration_gate", {})
        gate_str = "**PASS**" if gate_res.get("gate_passed") else "FAIL"

        t2 = m.get("type2_metrics", {})
        m_stat = t2.get("meta_d_status", "N/A")
        auroc2_val = t2.get("auroc2")
        auroc2_str = f"{auroc2_val:.3f}" if auroc2_val is not None else "N/A"
        brier_val = t2.get("brier_score")
        brier_str = f"{brier_val:.3f}" if brier_val is not None else "N/A"

        lines.append(
            f"| `{cell_key}` | {tot} | {cor} | **{acc}** | {ci} | {dp_str} | {sc_str} | {gate_str} | `{m_stat}` | {auroc2_str} | {brier_str} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Detailed Diagnostics by Coordinate Cell",
        f"",
    ])

    for cell_key, m in cell_metrics.items():
        gate = m.get("calibration_gate", {})
        t2 = m.get("type2_metrics", {})
        lines.extend([
            f"### Cell: `{cell_key}` (H={m['hop_depth']}, D={m['distractor_count']})",
            f"- **Trials:** {m['total_trials']} | **Accuracy:** {m['accuracy']:.1%} ({m['correct_trials']}/{m['total_trials']})",
            f"- **Type-1 Sensitivity ($d'$):** {m.get('sdt_d_prime', 0.0):+.2f} (95% CI: [{m.get('sdt_d_prime_ci_lower', 0.0):+.2f}, {m.get('sdt_d_prime_ci_upper', 0.0):+.2f}])",
            f"- **Type-1 Criterion ($c$):** {m.get('sdt_criterion_c', 0.0):+.2f} (95% CI: [{m.get('sdt_criterion_c_ci_lower', 0.0):+.2f}, {m.get('sdt_criterion_c_ci_upper', 0.0):+.2f}])",
            f"- **Option 1 Primacy Selection Rate:** {m.get('option_a_selection_rate', 0.0):.1%}",
            f"- **Calibration Gate Evaluation:**",
            f"  - Sensitivity Gate ($d' \\in [0.9, 1.4]$): {'PASS' if gate.get('d_prime_pass') else 'FAIL'}",
            f"  - Response Bias Gate ($|c| \\le 0.50$): {'PASS' if gate.get('criterion_pass') else 'FAIL'}",
            f"  - Accuracy Gate ($60\\% \\le Acc \\le 80\\%$): {'PASS' if gate.get('accuracy_pass') else 'FAIL'}",
            f"  - Schema Compliance Gate ($\\ge 95\\%$): {'PASS' if gate.get('compliance_pass') else 'FAIL'}",
            f"- **Type-2 Metacognitive Metrics:**",
            f"  - Status: `{t2.get('meta_d_status')}`",
            f"  - Mean Confidence: {t2.get('mean_confidence', 0.0):.1%}",
            f"  - Confidence Separation (Correct - Incorrect): {t2.get('confidence_separation', 0.0):+.1%}" if t2.get("confidence_separation") is not None else "  - Confidence Separation: N/A",
            f"  - AUROC2: {t2.get('auroc2', 'N/A')}",
            f"  - Brier Score: {t2.get('brier_score', 'N/A')}",
            f"",
        ])

    return "\n".join(lines)


def run_e02c_calibration(
    model_name: str = "qwen2.5:14b",
    mode: str = "local_search",
    grid: Optional[List[Tuple[int, int]]] = None,
    hop_depth: int = 1,
    distractor_count: int = 16,
    trials_per_cell: int = 24,
    seed: int = 42,
    temperature: float = 0.0,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute local 2D calibration search or held-out coordinate validation."""
    run_id = f"run_e02c_{mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir = output_dir or Path(f"artifacts/e02c_local_calibration/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine cells to test
    if mode == "validate":
        cells = [(hop_depth, distractor_count)]
        n_trials = trials_per_cell
    else:  # local_search
        if grid:
            cells = grid
        elif "14b" in model_name:
            cells = [(2, 32), (2, 64), (3, 4), (3, 8), (3, 16)]
        elif "llama" in model_name:
            cells = [(1, 16), (1, 32), (2, 4), (2, 8), (2, 16)]
        else:  # 3b default
            cells = [(1, 8), (1, 16), (1, 32), (2, 4), (2, 8)]
        n_trials = trials_per_cell

    print(f"\n{'='*60}")
    print(f"EXPERIMENT E02c: LOCAL 2D CALIBRATION & VALIDATION")
    print(f"Run ID: {run_id} | Model: {model_name} | Mode: {mode} | Cells: {cells} | N/Cell: {n_trials}")
    print(f"{'='*60}\n")

    backend = OllamaBackend(model_name=model_name, temperature=temperature, seed=seed)
    model_digest = backend.get_digest()

    task = AdaptiveMetacognition2AFCTask(
        task_family="multi_hop",
        ask_confidence=True,
        response_mode="direct_value",
    )

    items = task.generate_multi_hop_grid(
        grid_cells=cells,
        count_per_cell=n_trials,
        base_seed=seed,
        ask_confidence=True,
    )

    trial_records: List[Dict[str, Any]] = []
    cell_recs_map: Dict[str, List[Dict[str, Any]]] = {f"H{h}_D{d}": [] for h, d in cells}

    for item in items:
        h = item.metadata["hop_depth"]
        d = item.metadata["distractor_count"]
        cell_key = f"H{h}_D{d}"

        oa = item.metadata["option_map"]["A"]
        ob = item.metadata["option_map"]["B"]
        schema = make_2afc_direct_value_schema(oa, ob, ask_confidence=True)

        messages = [{"role": "user", "content": item.prompt}]
        t0 = time.perf_counter()
        raw_text, meta = backend.chat(
            messages=messages,
            temperature=temperature,
            seed=seed,
            format=schema,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        score = task.score_response(item, raw_text)

        rec = {
            "run_id": run_id,
            "cell_key": cell_key,
            "item_id": item.item_id,
            "hop_depth": h,
            "distractor_count": d,
            "ground_truth": item.ground_truth,
            "parsed_answer": score["parsed_answer"],
            "correct": score["correct"],
            "probability": score["probability"],
            "schema_valid": score["schema_valid"],
            "answer_parse_valid": score["answer_parse_valid"],
            "latency_ms": latency_ms,
            "prompt_eval_count": meta.get("prompt_eval_count"),
            "eval_count": meta.get("eval_count"),
            "prompt": item.prompt,
            "raw_response": raw_text,
        }
        trial_records.append(rec)
        cell_recs_map[cell_key].append(rec)

    # Compute metrics per cell
    cell_metrics: Dict[str, Dict[str, Any]] = {}
    for cell_key, recs in cell_recs_map.items():
        n_total = len(recs)
        correct_count = sum(1 for r in recs if r.get("correct", False))
        acc = float(correct_count / n_total) if n_total > 0 else 0.0
        ci_l, ci_u = compute_wilson_score_interval(correct_count, n_total)

        a_count = sum(1 for r in recs if str(r.get("parsed_answer", "")).upper() == "A")
        p_a = float(a_count / n_total) if n_total > 0 else 0.5

        sdt = compute_sdt_indices(recs, signal_target="A")
        sdt_ci = compute_sdt_bootstrap_ci(recs, signal_target="A", n_bootstraps=500)
        compliance = float(sum(1 for r in recs if r.get("schema_valid", False)) / n_total) if n_total > 0 else 0.0

        h_val, d_val = recs[0]["hop_depth"], recs[0]["distractor_count"]

        m_dict = {
            "cell_key": cell_key,
            "hop_depth": h_val,
            "distractor_count": d_val,
            "total_trials": n_total,
            "correct_trials": correct_count,
            "accuracy": acc,
            "ci_95_lower": ci_l,
            "ci_95_upper": ci_u,
            "option_a_selection_rate": p_a,
            "sdt_d_prime": sdt.get("d_prime"),
            "sdt_d_prime_ci_lower": sdt_ci.get("d_prime_ci_lower"),
            "sdt_d_prime_ci_upper": sdt_ci.get("d_prime_ci_upper"),
            "sdt_criterion_c": sdt.get("criterion_c"),
            "sdt_criterion_c_ci_lower": sdt_ci.get("criterion_c_ci_lower"),
            "sdt_criterion_c_ci_upper": sdt_ci.get("criterion_c_ci_upper"),
            "schema_compliance_rate": compliance,
        }

        # Gate & Type-2
        m_dict["calibration_gate"] = evaluate_calibration_gate(m_dict)
        m_dict["type2_metrics"] = compute_type2_sdt_metrics(recs)
        cell_metrics[cell_key] = m_dict

    manifest_dict = {
        "run_id": run_id,
        "target_model": model_name,
        "model_digest": model_digest,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "cells_evaluated": cells,
        "trials_per_cell": n_trials,
        "total_trials": len(trial_records),
        "temperature": temperature,
        "seed": seed,
    }

    summary_dict = {
        "manifest": manifest_dict,
        "cell_metrics": cell_metrics,
    }

    report_md = generate_e02c_markdown_report(
        manifest=manifest_dict,
        cell_metrics=cell_metrics,
        mode=mode,
    )

    df_trials = pd.DataFrame(trial_records)
    dirs_to_write = [out_dir]
    canonical_res_dir = Path(f"results/e02c_local_calibration/{run_id}")
    canonical_res_dir.mkdir(parents=True, exist_ok=True)
    if canonical_res_dir != out_dir:
        dirs_to_write.append(canonical_res_dir)

    for target_d in dirs_to_write:
        with open(target_d / "trials.jsonl", "w", encoding="utf-8") as f:
            for r in trial_records:
                f.write(json.dumps(r) + "\n")
        df_trials.to_parquet(target_d / "trials.parquet", index=False)
        with open(target_d / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)
        with open(target_d / "report.md", "w", encoding="utf-8") as f:
            f.write(report_md)
        with open(target_d / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, indent=2)

    print(f"\n{'='*60}")
    print(f"E02c COMPLETE | Artifacts: {out_dir}")
    print(f"{'='*60}\n")

    return summary_dict


def main():
    parser = argparse.ArgumentParser(description="Run Experiment E02c Local 2D Calibration & Validation")
    parser.add_argument("--model", type=str, default="qwen2.5:14b", help="Target model")
    parser.add_argument("--mode", type=str, default="local_search", choices=["local_search", "validate"], help="Execution mode")
    parser.add_argument("--hop-depth", type=int, default=1, help="Hop depth for validation mode")
    parser.add_argument("--distractor-count", type=int, default=16, help="Distractor count for validation mode")
    parser.add_argument("--trials-per-cell", type=int, default=24, help="Trials per grid cell")
    parser.add_argument("--seed", type=int, default=100, help="Random seed (use fresh seed for validation)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir) if args.output_dir else None
    run_e02c_calibration(
        model_name=args.model,
        mode=args.mode,
        hop_depth=args.hop_depth,
        distractor_count=args.distractor_count,
        trials_per_cell=args.trials_per_cell,
        seed=args.seed,
        temperature=args.temperature,
        output_dir=out_path,
    )


if __name__ == "__main__":
    main()
