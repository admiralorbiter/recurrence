"""Experiment E07: State x Memory Conflict & Causal State Intervention Benchmark (Sprint S08).

Evaluates whether StructuredSelfState acts as an independently manipulable causal control surface
under State x Memory conflict, reset with memory preserved, surgical single-slot inversion,
and clone/reconvergence testbeds.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from recurrence.backends.ollama import OllamaBackend
from recurrence.tasks.intervention import (
    StateInterventionGenerator,
    MatchedTwinEpisodePair,
    CloneReconvergenceSpec,
)
from recurrence.loop.intervention_experiment import (
    InterventionHarness,
    InterventionTrialResult,
)
from recurrence.analysis.intervention_metrics import (
    analyze_state_intervention_results,
    StateInterventionAnalysisSummary,
)


class MockInterventionBackend:
    """Mock backend for dry-run verification of Sprint S08."""

    def __init__(self, model_name: str = "qwen2.5:3b") -> None:
        self.model_name = model_name

    def get_digest(self) -> str:
        return "mock_digest_dryrun_e07"

    def step(self, prompt: str, format: Optional[Dict[str, Any]] = None) -> tuple[str, str, Dict[str, Any]]:
        text = json.dumps({"answer": "A"})
        metadata = {
            "prompt_eval_count": len(prompt) // 4,
            "eval_count": len(text) // 4,
            "total_duration_ms": 5.0,
        }
        return text, "hash_mock_e07", metadata


def generate_e07_markdown_report(
    manifest: Dict[str, Any],
    analysis: StateInterventionAnalysisSummary,
    twin_manifests: List[Dict[str, Any]],
    raw_df: pd.DataFrame,
) -> str:
    """Generate publication-ready Markdown report for Experiment E07."""
    lines = [
        f"# Experiment E07: State $\\times$ Memory Conflict & Causal Intervention Report (Sprint S08)",
        f"",
        f"**Run ID:** `{manifest['run_id']}`  ",
        f"**Model:** `{manifest['target_model']}` (`{manifest['model_digest'][:12]}...`)  ",
        f"**Phase:** `{manifest['phase'].upper()}` (Seed: `{manifest['seed']}`)  ",
        f"**Date:** {manifest['start_time']}  ",
        f"**Scope:** {manifest['total_twin_pairs']} Matched Twin Episode Pairs | {manifest['total_trials']} Total Paired Intervention Trials  ",
        f"**Primary Question:** *Holding the model's explicit memory and final question constant, does changing only the explicit StructuredSelfState causally redirect downstream behavior?*  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Causal Steering Estimands",
        f"",
        f"| Causal Estimand | Description | Point Estimate | 95% Bootstrap CI | Permutation $p$ (Method) | Scientific Inference |",
        f"| :--- | :--- | :---: | :---: | :---: | :--- |",
    ]

    for name, est in analysis.causal_estimands.items():
        if name == "Delta_allegiance":
            sig_desc = "**Statistically Distinguishable Conflict Preference**" if est.is_statistically_distinguishable else "**No Resolved Conflict Preference (Null)**"
        elif name == "Delta_state_given_memory":
            sig_desc = "**State Has Causal Leverage**" if est.is_statistically_distinguishable and est.point_estimate > 0 else "**Transcript-Equivalent Null**"
        elif name == "Delta_memory_given_state":
            sig_desc = "**Memory Has Causal Leverage**" if est.is_statistically_distinguishable and est.point_estimate > 0 else "**Memory Invariant / Null**"
        elif name == "Reset_Dependence":
            sig_desc = "**State Carries Critical Leverage**" if est.is_statistically_distinguishable and est.point_estimate > 0 else "**Direct Memory Fully Compensates**"
        else:
            sig_desc = "**Statistically Significant**" if est.is_statistically_distinguishable else "**Null / Invariant**"

        lines.append(
            f"| **`{name}`** | {est.description} | **{est.point_estimate:+.1%}** | [{est.ci_lower_95:+.1%}, {est.ci_upper_95:+.1%}] | {est.permutation_p_value:.4f} (`{est.permutation_method}`) | {sig_desc} |"
        )

    cp = analysis.conflict_partition
    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. State $\\times$ Memory Conflict 3-Way Partition & Directional Breakdown",
        f"",
        f"- **Total Conflict Trials Evaluated:** {cp.total_conflict_trials}",
        f"- **Follows State Value Rate ($SAR$):** **{cp.follows_state_rate:.1%}**",
        f"- **Follows Memory Value Rate ($MAR$):** **{cp.follows_memory_rate:.1%}**",
        f"- **Chooses Neither / Foil Option Rate:** **{cp.chooses_neither_rate:.1%}**",
        f"- **Conditional State Preference ($P(\\text{{State}} \\mid \\text{{State or Memory}})$):** **{cp.conditional_state_preference:.1%}**",
        f"- **Primary Conflict Contrast ($\\Delta_{{\\text{{allegiance}}}} = SAR - MAR$):** **{cp.delta_allegiance:+.1%}**",
        f"",
        f"### Directional Conflict Breakdown:",
        f"- **Direction 1 ($M_A + S_B$):** State Allegiance = **{cp.directional_MA_SB_state_rate:.1%}** | Memory Allegiance = **{cp.directional_MA_SB_memory_rate:.1%}**",
        f"- **Direction 2 ($M_B + S_A$):** State Allegiance = **{cp.directional_MB_SA_state_rate:.1%}** | Memory Allegiance = **{cp.directional_MB_SA_memory_rate:.1%}**",
        f"",
        f"---",
        f"",
        f"## 3. Multi-Condition Intervention Matrix Breakdown",
        f"",
        f"| Condition | Presentation Order | Trials | State Allegiance | Memory Allegiance | Target Acc (Congruent) | Control Acc | Goal Acc | Prompt Tokens | Latency |",
        f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for key, stats in analysis.condition_stats.items():
        cond_label = {
            "congruent_A": "Congruent Baseline A ($M_A + S_A$)",
            "congruent_B": "Congruent Baseline B ($M_B + S_B$)",
            "conflict_MA_SB": "State/Memory Conflict ($M_A + S_B$)",
            "conflict_MB_SA": "State/Memory Conflict ($M_B + S_A$)",
            "reset_MA_Sempty": "Reset with Memory Preserved ($M_A + S_0$)",
            "surgical_MA_SAprime": "Surgical Slot Inversion ($M_A + S_A'$)",
            "state_only_SA": "State-Only Calibration ($S_A$)",
            "state_only_SB": "State-Only Calibration ($S_B$)",
            "state_only_Sempty": "State-Only Calibration ($S_0$)",
            "memory_only_MA": "Memory-Only Calibration ($M_A$)",
            "memory_only_MB": "Memory-Only Calibration ($M_B$)",
            "clone_fork_A_congruent": "Clone Fork A (Congruent)",
            "clone_fork_A_cross_swap_SB": "Clone Fork A (Cross-Swap $S_B$)",
            "clone_fork_B_congruent": "Clone Fork B (Congruent)",
            "reconverged_branch_A": "Reconverged Branch A ($E_{\\text{sync}}$)",
            "reconverged_branch_B": "Reconverged Branch B ($E_{\\text{sync}}$)",
        }.get(stats.condition, stats.condition)

        tgt_acc_str = f"{stats.target_accuracy_congruent:.1%}" if "congruent" in stats.condition or "calibration" in stats.condition else "—"
        ctrl_acc_str = f"{stats.control_accuracy:.1%}"
        goal_acc_str = f"{stats.goal_accuracy:.1%}"

        lines.append(
            f"| **{cond_label}** | `{stats.presentation_order}` | {stats.total_trials} | **{stats.state_allegiance_rate:.1%}** | **{stats.memory_allegiance_rate:.1%}** | {tgt_acc_str} | {ctrl_acc_str} | {goal_acc_str} | {stats.mean_prompt_tokens:.1f} tok | {stats.mean_latency_ms:.1f} ms |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Surgical Single-Slot Edit & Local Causal Precision",
        "",
        f"- **Target Slot Intervention Uptake (P(Target = Injected)):** **{analysis.local_precision.target_intervention_uptake:.1%}**",
        f"- **Control Slot Preservation (P(Control = Gold)):** **{analysis.local_precision.control_slot_preservation:.1%}**",
        f"- **Joint Local Causal Precision (P(Target Uptake and Control Preserved)):** **{analysis.local_precision.joint_local_causal_precision:.1%}**",
        "",
        "---",
        "",
        "## 5. Presentation Order Sensitivity & Infrastructure Invariants",
        "",
        f"- **State Allegiance (Memory -> State Order):** **{analysis.order_effects.get('state_allegiance_memory_first', 0.0):.1%}**",
        f"- **State Allegiance (State -> Memory Order):** **{analysis.order_effects.get('state_allegiance_state_first', 0.0):.1%}**",
        f"- **Order Sensitivity Gap:** **{analysis.order_effects.get('order_sensitivity_gap', 0.0):+.1%}**",
        f"- **Reconvergence Behavioral Concordance Rate:** **{analysis.reconvergence_concordance_rate:.1%}**",
        "",
        "---",
        "",
        "## 6. Scientific Gate Decision for Sprint S08",
        "",
        "1. **Causal State Steering vs Transcript Equivalence:** Does explicit state intervention reliably steer the model's output away from historical memory?",
        "2. **Reset Dependence:** Does removing state with memory intact impair performance, proving explicit state provides non-redundant operational utility?",
        "3. **Local Surgical Precision:** Does single-slot editing steer the targeted behavior without causing collateral representation drift?",
    ])

    return "\n".join(lines)


def run_e07_experiment(
    model_name: str = "qwen2.5:3b",
    seed: int = 42,
    phase: str = "exploratory",
    temperature: float = 0.0,
    twin_pairs_count: Optional[int] = None,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete E07 benchmark suite across matched twin pairs and intervention conditions."""
    run_id = f"run_e07_interv_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{phase}"
    if dry_run:
        run_id += "_dryrun"

    out_dir = output_dir or Path(f"artifacts/e07_state_interventions/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_results_dir = Path(f"results/e07_state_interventions/{run_id}")
    canonical_results_dir.mkdir(parents=True, exist_ok=True)

    n_twin_pairs = twin_pairs_count or (16 if phase == "confirmatory" else 4)

    print("=" * 70)
    print(f"EXPERIMENT E07: STATE x MEMORY CONFLICT & CAUSAL INTERVENTIONS (S08)")
    print(f"Run ID: {run_id} | Model: {model_name} | Phase: {phase.upper()} | Seed: {seed} | Dry Run: {dry_run}")
    print(f"Matched Twin Pairs: {n_twin_pairs} (Total Episodes: {n_twin_pairs * 2})")
    print("=" * 70)

    if dry_run:
        backend = MockInterventionBackend(model_name=model_name)
    else:
        backend = OllamaBackend(
            model_name=model_name,
            temperature=temperature,
            seed=seed,
        )

    digest = backend.get_digest()
    generator = StateInterventionGenerator(seed=seed)
    harness = InterventionHarness(backend=backend)

    all_trials: List[InterventionTrialResult] = []
    twin_manifests: List[Dict[str, Any]] = []

    for twin_idx in range(n_twin_pairs):
        print(f"\n--- EXECUTING TWIN PAIR {twin_idx + 1}/{n_twin_pairs} ---")
        twin_pair = generator.generate_twin_pair(
            twin_idx=twin_idx,
            prefix_ticks=4,
            seed=seed,
        )

        twin_trials = harness.execute_twin_pair(twin_pair=twin_pair)
        all_trials.extend(twin_trials)

        # Clone / Reconvergence Testbed
        clone_spec = generator.generate_clone_reconvergence_spec(twin_idx=twin_idx, seed=seed)
        clone_trials = harness.execute_clone_reconvergence(clone_spec=clone_spec)
        all_trials.extend(clone_trials)

        twin_manifests.append({
            "pair_id": twin_pair.pair_id,
            "twin_index": twin_idx,
            "k_target": twin_pair.k_target,
            "k_control": twin_pair.k_control,
            "val_target_A": twin_pair.val_target_A,
            "val_target_B": twin_pair.val_target_B,
            "val_control": twin_pair.val_control,
            "trials_recorded": len(twin_trials) + len(clone_trials),
        })
        print(f"  -> Twin Pair {twin_pair.pair_id}: {len(twin_trials) + len(clone_trials)} trials recorded.")

    print(f"\nTotal Trials Recorded: {len(all_trials)}")

    analysis = analyze_state_intervention_results(
        trials=all_trials,
        num_bootstrap=2000,
        seed=seed,
    )

    manifest = {
        "run_id": run_id,
        "target_model": model_name,
        "model_digest": digest,
        "phase": phase,
        "seed": seed,
        "temperature": temperature,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "total_twin_pairs": len(twin_manifests),
        "total_trials": len(all_trials),
    }

    df_trials = pd.DataFrame([asdict(t) for t in all_trials])

    report_md = generate_e07_markdown_report(
        manifest=manifest,
        analysis=analysis,
        twin_manifests=twin_manifests,
        raw_df=df_trials,
    )

    for target_dir in [out_dir, canonical_results_dir]:
        with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        summary_payload = {
            "manifest": manifest,
            "analysis": {
                "total_twin_pairs": analysis.total_twin_pairs,
                "total_trials": analysis.total_trials,
                "condition_stats": {k: asdict(v) for k, v in analysis.condition_stats.items()},
                "causal_estimands": {k: asdict(v) for k, v in analysis.causal_estimands.items()},
                "conflict_partition": asdict(analysis.conflict_partition),
                "local_precision": asdict(analysis.local_precision),
                "reconvergence_concordance_rate": analysis.reconvergence_concordance_rate,
                "order_effects": analysis.order_effects,
            },
            "twins": twin_manifests,
        }
        with open(target_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        with open(target_dir / "trials.jsonl", "w", encoding="utf-8") as f:
            for t in all_trials:
                f.write(json.dumps(asdict(t)) + "\n")

        df_trials.to_parquet(target_dir / "trials.parquet", index=False)

        with open(target_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(report_md)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT E07 BENCHMARK COMPLETE")
    print(f"Artifacts written to: {out_dir}")
    print(f"Canonical Results written to: {canonical_results_dir}")
    print("=" * 70 + "\n")

    return {
        "manifest": manifest,
        "analysis": analysis,
        "report_md": report_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E07: State x Memory Conflict & Causal State Intervention Benchmark (Sprint S08)")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--phase", type=str, default="exploratory", choices=["exploratory", "confirmatory"], help="Experiment phase")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--twin-pairs", type=int, default=None, help="Number of matched twin episode pairs (default 4 exploratory, 16 confirmatory)")
    parser.add_argument("--dry-run", action="store_true", help="Run with mock backend")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()

    run_e07_experiment(
        model_name=args.model,
        seed=args.seed,
        phase=args.phase,
        temperature=args.temperature,
        twin_pairs_count=args.twin_pairs,
        dry_run=args.dry_run,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
