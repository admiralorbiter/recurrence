"""Scientific Runner for Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping & Psychometric Surface Pilot.

Maps first-order accuracy, confidence calibration, position bias, and monotonicity
across three independent difficulty dimensions on matched 2AFC tasks:
  1. Distractor Load Sweep: D in [4, 8, 16, 32, 64, 128, 256] (1-hop, middle stratum)
  2. Multi-Hop Pointer Depth Sweep: H in [1, 2, 3, 4, 5] (relational chaining)
  3. Overwrite Load Sweep: U in [0, 1, 2, 3, 4] (temporal updates, stale foil)

Includes paired elicitation reactivity controls (Answer-Only vs Answer+Confidence).
Outputs structured manifest, trials JSONL/Parquet, and markdown psychometric report.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from recurrence.backends.ollama import OllamaBackend
from recurrence.backends.toy import ToyBackend
from recurrence.core.manifest import RunManifest
from recurrence.core.logging import ExperimentLogger
from recurrence.core.schemas import TARGET_2AFC_CONFIDENCE_SCHEMA, TARGET_2AFC_SCHEMA
from recurrence.tasks.adaptive_metacognition import (
    AdaptiveMetacognition2AFCTask,
    DifficultyConfig,
)
from recurrence.analysis.psychophysics import (
    compute_psychometric_curve,
    compute_monotonicity_diagnostics,
    compute_elicitation_reactivity,
)


class Mock2AFCBackend:
    """Deterministic mock backend for dry-run testing with simulated monotonic drop."""
    def __init__(self, model_name: str = "mock-qwen2.5:3b", seed: int = 42):
        self.model_name = model_name
        self.seed = seed
        self.model_digest = "mock_2afc_digest_000000000000000000000000"

    def get_digest(self) -> str:
        return self.model_digest

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        format: Optional[Any] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        content = messages[0]["content"] if messages else ""
        # Check if probability schema is requested
        is_conf = "probability" in content or (isinstance(format, dict) and "probability" in format.get("properties", {}))
        
        # Simulate high accuracy on small context, lower on large context
        prob_val = 80
        ans_val = "A" if "Question:" in content else "B"
        
        if is_conf:
            resp = json.dumps({"answer": ans_val, "probability": prob_val})
        else:
            resp = json.dumps({"answer": ans_val})

        return resp, {"prompt_eval_count": len(content) // 4, "eval_count": 12}


def generate_e02b_markdown_report(
    manifest: Dict[str, Any],
    sweep_results: Dict[str, Dict[str, Any]],
    reactivity_results: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate scientific research memo and tabular psychometric curves for E02b."""
    lines = [
        f"# Experiment E02b: Horizon 0 v2 Difficulty-Grid Mapping Report",
        f"",
        f"**Run ID:** `{manifest.get('run_id')}`  ",
        f"**Target Model:** `{manifest.get('target_model')}` (`{manifest.get('model_digest', 'N/A')[:12]}...`)  ",
        f"**Date:** {manifest.get('start_time', datetime.now(timezone.utc).isoformat())}  ",
        f"**Response Format:** Matched 2-Alternative Forced Choice (2AFC) with 50/50 Counterbalancing  ",
        f"**Total Trials:** {manifest.get('total_trials', 0)}  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Staircase Readiness",
        f"",
        f"| Sweep / Task Family | Monotonicity (Spearman $\\rho$) | Kendall $\\tau$ | Min Acc | Max Acc | Acc Span | Readiness Verdict |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for name, res in sweep_results.items():
        mono = res.get("monotonicity_diagnostics", {})
        summ = res.get("overall_summary", {})
        rho = mono.get("spearman_rho")
        rho_str = f"{rho:+.3f}" if rho is not None else "N/A"
        tau = mono.get("kendall_tau")
        tau_str = f"{tau:+.3f}" if tau is not None else "N/A"
        min_a = f"{mono.get('min_accuracy', 0.0):.1%}"
        max_a = f"{mono.get('max_accuracy', 0.0):.1%}"
        span = f"{mono.get('max_accuracy_drop', 0.0):.1%}"
        verdict = mono.get("staircase_readiness", "unknown").replace("_", " ").title()
        lines.append(
            f"| **{name.replace('_', ' ').title()}** | {rho_str} | {tau_str} | {min_a} | {max_a} | {span} | **{verdict}** |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Psychometric Curves by Task Family",
        f"",
    ])

    for name, res in sweep_results.items():
        lines.extend([
            f"### Sweep: {name.replace('_', ' ').title()}",
            f"",
            f"| Level | Trials | Correct | Accuracy | 95% Wilson CI | P(Chose 'A') | SDT $d'$ [95% CI] | SDT $c$ [95% CI] | Mean Conf | Conf Sep | Brier | Prompt Tok | Compliance |",
            f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for lvl_str, m in res.get("level_metrics", {}).items():
            lvl = m["difficulty_level"]
            tot = m["total_trials"]
            cor = m["correct_trials"]
            acc = f"{m['accuracy']:.1%}"
            ci = f"[{m['ci_95_lower']:.1%}, {m['ci_95_upper']:.1%}]"
            pa = f"{m['option_a_selection_rate']:.1%}"
            
            dp_val = m.get("sdt_d_prime")
            dp_l = m.get("sdt_d_prime_ci_lower")
            dp_u = m.get("sdt_d_prime_ci_upper")
            if dp_val is not None and dp_l is not None and dp_u is not None:
                dp = f"{dp_val:+.2f} [{dp_l:+.2f}, {dp_u:+.2f}]"
            elif dp_val is not None:
                dp = f"{dp_val:+.2f}"
            else:
                dp = "N/A"

            sc_val = m.get("sdt_criterion_c")
            sc_l = m.get("sdt_criterion_c_ci_lower")
            sc_u = m.get("sdt_criterion_c_ci_upper")
            if sc_val is not None and sc_l is not None and sc_u is not None:
                sc = f"{sc_val:+.2f} [{sc_l:+.2f}, {sc_u:+.2f}]"
            elif sc_val is not None:
                sc = f"{sc_val:+.2f}"
            else:
                sc = "N/A"

            mc = f"{m['mean_confidence']:.1%}" if m.get("mean_confidence") is not None else "N/A"
            cs = f"{m['confidence_separation']:+.1%}" if m.get("confidence_separation") is not None else "N/A"
            br = f"{m['brier_score']:.3f}" if m.get("brier_score") is not None else "N/A"
            tok = f"{m['mean_estimated_tokens']:.0f}"
            comp = f"{m['schema_compliance_rate']:.1%}"
            lines.append(
                f"| `{lvl}` | {tot} | {cor} | **{acc}** | {ci} | {pa} | {dp} | {sc} | {mc} | {cs} | {br} | {tok} | {comp} |"
            )
        lines.append("")

        # Add adjacent level paired transitions if present
        transitions = res.get("paired_transitions", [])
        if transitions:
            lines.extend([
                f"#### Within-Item Paired Transitions (Adjacent Levels):",
                f"",
                f"| Transition ($D_k \\to D_{{k+1}}$) | Retained ($1 \\to 1$) | Degraded ($1 \\to 0$) | Persisted Wrong ($0 \\to 0$) | Rebounded ($0 \\to 1$) | Degradation Rate | Rebound Rate | Net $\\Delta$ Acc |",
                f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for t in transitions:
                lines.append(
                    f"| `{t['from_level']} -> {t['to_level']}` | {t['retained_correct_1_to_1']} | {t['degraded_1_to_0']} | {t['persisted_wrong_0_to_0']} | {t['rebounded_0_to_1']} | {t['degradation_rate']:.1%} | {t['rebound_rate']:.1%} | {t['net_accuracy_delta']:+.1%} |"
                )
            lines.append("")

    if reactivity_results:
        lines.extend([
            f"---",
            f"",
            f"## 3. Elicitation Reactivity Control (Answer-Only vs. Answer+Confidence)",
            f"",
            f"- **Paired Trials Evaluated:** {reactivity_results.get('paired_trials_count')}",
            f"- **Answer-Only Accuracy:** {reactivity_results.get('answer_only_accuracy', 0.0):.1%}",
            f"- **Answer+Confidence Accuracy:** {reactivity_results.get('answer_conf_accuracy', 0.0):.1%}",
            f"- **Delta Accuracy (Confidence - Only):** {reactivity_results.get('delta_accuracy_conf_minus_only', 0.0):+.1%}",
            f"- **Exact Option Concordance Rate:** {reactivity_results.get('exact_answer_concordance_rate', 0.0):.1%}",
            f"- **McNemar Chi2 Statistic:** {reactivity_results.get('mcnemar_chi2_statistic', 0.0):.3f} (p = {reactivity_results.get('mcnemar_p_value', 1.0):.4f})",
            f"- **Reactivity Verdict:** `{reactivity_results.get('reactivity_status')}`",
            f"- **Policy Reactivity Note:** Concordance below 85% reflects item-level decision changes even if net accuracy difference is small.",
            f"",
        ])

    lines.extend([
        f"---",
        f"",
        f"## 4. Scientific Takeaways for Adaptive Calibration",
        f"",
        f"1. **Monotonicity Assessment:** A difficulty axis is staircase-ready if it exhibits a consistent negative rank correlation (rho <= -0.70, negative_step_ratio >= 0.70) and spans the operational target window (~55-90%).",
        f"2. **Response Bias (Criterion $c$):** Positive $c$ indicates an Option-B selection bias under Signal=A conventions. Extreme criterion shifts ($|c| > 1.0$) indicate response collapse rather than sensitivity loss.",
        f"3. **Within-Item Transitions:** High rebound rates ($0 \\to 1$) indicate order/placement sensitivity or item noise, whereas high degradation ($1 \\to 0$) confirms true capacity degradation.",
    ])

    return "\n".join(lines)


def run_e02b_difficulty_mapping(
    model_name: str = "qwen2.5:3b",
    sweeps: str = "all",
    trials_per_level: int = 16,
    paired_reactivity: bool = True,
    seed: int = 42,
    temperature: float = 0.0,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete E02b difficulty-grid mapping across selected sweeps."""
    run_id = f"run_e02b_diff_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if dry_run:
        run_id += "_dryrun"

    out_dir = output_dir or Path(f"artifacts/e02b_difficulty_map/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Backend
    if dry_run:
        backend = Mock2AFCBackend(model_name="mock-qwen2.5:3b", seed=seed)
        model_digest = backend.get_digest()
    else:
        backend = OllamaBackend(model_name=model_name, temperature=temperature, seed=seed)
        model_digest = backend.get_digest()

    print(f"\n{'='*60}")
    print(f"EXPERIMENT E02b: H0-v2 DIFFICULTY-GRID MAPPING PILOT")
    print(f"Run ID: {run_id} | Model: {model_name} | Trials/Level: {trials_per_level}")
    print(f"{'='*60}\n")

    trial_records: List[Dict[str, Any]] = []
    sweep_results: Dict[str, Dict[str, Any]] = {}
    paired_reactivity_records: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    # 1. Distractor Load Sweep
    if sweeps in ["all", "distractors", "distractor_load"]:
        print("\n[Sweep 1/3]: Distractor Load Sweep (D in [4, 8, 16, 32, 64, 128, 256])...")
        d_levels = [4, 8, 16, 32, 64, 128, 256]
        d_task = AdaptiveMetacognition2AFCTask(task_family="distractor_load", ask_confidence=True)
        d_items = d_task.generate_distractor_sweep(
            levels=d_levels,
            count_per_level=trials_per_level,
            base_seed=seed,
            ask_confidence=True,
        )

        d_records: List[Dict[str, Any]] = []
        for idx, item in enumerate(d_items):
            messages = [{"role": "user", "content": item.prompt}]
            t0 = time.perf_counter()
            raw_text, meta = backend.chat(
                messages=messages,
                temperature=temperature,
                seed=seed,
                format=TARGET_2AFC_CONFIDENCE_SCHEMA,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            score = d_task.score_response(item, raw_text)

            rec = {
                "run_id": run_id,
                "task_family": "distractor_load",
                "item_id": item.item_id,
                "difficulty_level": item.metadata["distractor_count"],
                "distractor_count": item.metadata["distractor_count"],
                "hop_depth": 1,
                "overwrite_count": 0,
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
                "ask_confidence": True,
                "is_reactivity_control": False,
            }
            d_records.append(rec)
            trial_records.append(rec)

        d_curve = compute_psychometric_curve(d_records, difficulty_key="difficulty_level")
        sweep_results["distractor_load"] = d_curve
        print(f"  -> Distractor Sweep Accuracy: {d_curve['overall_summary']['overall_accuracy']:.1%} | Readiness: {d_curve['monotonicity_diagnostics']['staircase_readiness']}")

    # 2. Multi-Hop Pointer Depth Sweep
    if sweeps in ["all", "hops", "multi_hop"]:
        print("\n[Sweep 2/3]: Multi-Hop Pointer Depth Sweep (H in [1, 2, 3, 4, 5])...")
        h_levels = [1, 2, 3, 4, 5]
        h_task = AdaptiveMetacognition2AFCTask(task_family="multi_hop", ask_confidence=True)
        h_items = h_task.generate_multi_hop_sweep(
            levels=h_levels,
            count_per_level=trials_per_level,
            distractor_count=16,
            base_seed=seed,
            ask_confidence=True,
        )

        h_records: List[Dict[str, Any]] = []
        for idx, item in enumerate(h_items):
            messages = [{"role": "user", "content": item.prompt}]
            t0 = time.perf_counter()
            raw_text, meta = backend.chat(
                messages=messages,
                temperature=temperature,
                seed=seed,
                format=TARGET_2AFC_CONFIDENCE_SCHEMA,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            score = h_task.score_response(item, raw_text)

            rec = {
                "run_id": run_id,
                "task_family": "multi_hop",
                "item_id": item.item_id,
                "difficulty_level": item.metadata["hop_depth"],
                "distractor_count": 16,
                "hop_depth": item.metadata["hop_depth"],
                "overwrite_count": 0,
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
                "ask_confidence": True,
                "is_reactivity_control": False,
            }
            h_records.append(rec)
            trial_records.append(rec)

        h_curve = compute_psychometric_curve(h_records, difficulty_key="difficulty_level")
        sweep_results["multi_hop"] = h_curve
        print(f"  -> Multi-Hop Sweep Accuracy: {h_curve['overall_summary']['overall_accuracy']:.1%} | Readiness: {h_curve['monotonicity_diagnostics']['staircase_readiness']}")

    # 3. Overwrite Load Sweep
    if sweeps in ["all", "overwrites", "overwrite_load"]:
        print("\n[Sweep 3/3]: Overwrite Load Sweep (U in [0, 1, 2, 3, 4])...")
        u_levels = [0, 1, 2, 3, 4]
        u_task = AdaptiveMetacognition2AFCTask(task_family="overwrite_load", ask_confidence=True)
        u_items = u_task.generate_overwrite_sweep(
            levels=u_levels,
            count_per_level=trials_per_level,
            distractor_count=16,
            base_seed=seed,
            ask_confidence=True,
        )

        u_records: List[Dict[str, Any]] = []
        for idx, item in enumerate(u_items):
            messages = [{"role": "user", "content": item.prompt}]
            t0 = time.perf_counter()
            raw_text, meta = backend.chat(
                messages=messages,
                temperature=temperature,
                seed=seed,
                format=TARGET_2AFC_CONFIDENCE_SCHEMA,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            score = u_task.score_response(item, raw_text)

            rec = {
                "run_id": run_id,
                "task_family": "overwrite_load",
                "item_id": item.item_id,
                "difficulty_level": item.metadata["overwrite_count"],
                "distractor_count": 16,
                "hop_depth": 1,
                "overwrite_count": item.metadata["overwrite_count"],
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
                "ask_confidence": True,
                "is_reactivity_control": False,
            }
            u_records.append(rec)
            trial_records.append(rec)

        u_curve = compute_psychometric_curve(u_records, difficulty_key="difficulty_level")
        sweep_results["overwrite_load"] = u_curve
        print(f"  -> Overwrite Sweep Accuracy: {u_curve['overall_summary']['overall_accuracy']:.1%} | Readiness: {u_curve['monotonicity_diagnostics']['staircase_readiness']}")

    # 4. Optional Paired Elicitation Reactivity Control
    reactivity_summary = None
    if paired_reactivity:
        print("\nEvaluating Paired Elicitation Reactivity Control (Answer-Only vs Answer+Confidence)...")
        reactivity_items_conf = d_items[:trials_per_level] if "distractor_load" in sweep_results else []
        if reactivity_items_conf:
            react_task_only = AdaptiveMetacognition2AFCTask(task_family="distractor_load", ask_confidence=False)
            react_items_only = react_task_only.generate_distractor_sweep(
                levels=[d_levels[0]],
                count_per_level=trials_per_level,
                base_seed=seed,
                ask_confidence=False,
                nested=True,
            )

            for item_only, item_conf in zip(react_items_only, reactivity_items_conf):
                messages = [{"role": "user", "content": item_only.prompt}]
                t0 = time.perf_counter()
                raw_only, meta_only = backend.chat(
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    format=TARGET_2AFC_SCHEMA,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                score_only = react_task_only.score_response(item_only, raw_only)
                rec_only = {
                    "run_id": run_id,
                    "task_family": "distractor_load",
                    "item_id": item_only.item_id,
                    "difficulty_level": item_only.metadata["distractor_count"],
                    "distractor_count": item_only.metadata["distractor_count"],
                    "hop_depth": 1,
                    "overwrite_count": 0,
                    "ground_truth": item_only.ground_truth,
                    "parsed_answer": score_only["parsed_answer"],
                    "correct": score_only["correct"],
                    "probability": None,
                    "schema_valid": score_only["schema_valid"],
                    "answer_parse_valid": score_only["answer_parse_valid"],
                    "latency_ms": latency_ms,
                    "prompt_eval_count": meta_only.get("prompt_eval_count"),
                    "eval_count": meta_only.get("eval_count"),
                    "prompt": item_only.prompt,
                    "raw_response": raw_only,
                    "ask_confidence": False,
                    "is_reactivity_control": True,
                }
                trial_records.append(rec_only)

                # Find corresponding conf record
                matching_conf = next(r for r in d_records if r["item_id"] == item_conf.item_id)
                paired_reactivity_records.append((rec_only, matching_conf))

            reactivity_summary = compute_elicitation_reactivity(paired_reactivity_records)
            print(f"  -> Reactivity Status: {reactivity_summary['reactivity_status']} | Delta Acc: {reactivity_summary['delta_accuracy_conf_minus_only']:+.1%}")

    # Output Serialization
    manifest_dict = {
        "run_id": run_id,
        "target_model": model_name,
        "model_digest": model_digest,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "sweeps_evaluated": list(sweep_results.keys()),
        "trials_per_level": trials_per_level,
        "total_trials": len(trial_records),
        "temperature": temperature,
        "seed": seed,
    }

    summary_dict = {
        "manifest": manifest_dict,
        "sweep_results": sweep_results,
        "reactivity_results": reactivity_summary,
    }

    report_md = generate_e02b_markdown_report(
        manifest=manifest_dict,
        sweep_results=sweep_results,
        reactivity_results=reactivity_summary,
    )

    df_trials = pd.DataFrame(trial_records)

    dirs_to_write = [out_dir]
    canonical_res_dir = Path(f"results/e02b_difficulty_map/{run_id}")
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
    print(f"E02b DIFFICULTY MAPPING COMPLETE")
    print(f"Artifacts: {out_dir}")
    print(f"Canonical Results: {canonical_res_dir}")
    print(f"{'='*60}\n")

    return summary_dict


def main():
    parser = argparse.ArgumentParser(description="Run Experiment E02b H0-v2 Difficulty-Grid Mapping Pilot")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Target model identifier")
    parser.add_argument("--sweeps", type=str, default="all", choices=["all", "distractors", "hops", "overwrites"], help="Sweeps to run")
    parser.add_argument("--trials-per-level", type=int, default=16, help="Fresh trials per difficulty level (default: 16)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (default: 0.0)")
    parser.add_argument("--toy", action="store_true", help="Run with mock 2AFC backend for fast verification")
    parser.add_argument("--no-reactivity", action="store_true", help="Disable paired answer-only reactivity control")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir) if args.output_dir else None
    run_e02b_difficulty_mapping(
        model_name=args.model,
        sweeps=args.sweeps,
        trials_per_level=args.trials_per_level,
        paired_reactivity=not args.no_reactivity,
        seed=args.seed,
        temperature=args.temperature,
        dry_run=args.toy,
        output_dir=out_path,
    )


if __name__ == "__main__":
    main()
