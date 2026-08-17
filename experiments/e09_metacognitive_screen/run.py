"""Experiment E09: Metacognitive Continuity & Item-Paired Post-Choice Error Prediction Screen (Sprint S09b)."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from recurrence.backends.ollama import OllamaBackend
from recurrence.tasks.ownership import (
    OwnershipTaskGenerator,
    OwnershipEpisode,
)
from recurrence.loop.ownership_experiment import (
    OwnershipHarness,
    OwnershipTrialResult,
)
from recurrence.analysis.ownership_metrics import (
    analyze_ownership_results,
    S09AnalysisSummary,
)
from experiments.e08_source_ownership.run import MockOwnershipBackend


def generate_e09_markdown_report(
    manifest: Dict[str, Any],
    analysis: S09AnalysisSummary,
    ep_manifests: List[Dict[str, Any]],
    raw_df: pd.DataFrame,
) -> str:
    """Generate publication-ready Markdown report for Experiment E09."""
    lines = [
        f"# Experiment E09: Metacognitive Continuity & Item-Paired Error Prediction Report (Sprint S09b)",
        f"",
        f"**Run ID:** `{manifest['run_id']}`  ",
        f"**Model:** `{manifest['target_model']}` (`{manifest['model_digest'][:12]}...`)  ",
        f"**Phase:** `{manifest['phase'].upper()}` (Seed: `{manifest['seed']}`)  ",
        f"**Date:** {manifest['start_time']}  ",
        f"**Scope:** {manifest['total_episodes']} Multi-Source Episodes | {manifest['total_trials']} Total Metacognitive Probes  ",
        f"**Provenance & Epistemic Status:** Raw Trial Freeze: `4bc4b6b` | Post-Confirmatory Analysis: `7e65b52` | Version: `v1.1_post_confirmatory_repaired`  ",
        f"**Primary Question:** *Under matched visible public information, does self-referential framing provide a post-choice error-prediction advantage over an auditing observer predicting the exact same target decisions?*  ",
        f"",
        f"---",
        f"",
        f"## 1. Metacognitive Calibration Breakdown (Brier Score & AUROC Resolution)",
        f"",
        f"| Evaluator | Memory Format | Trials | Accuracy | Mean Confidence | Brier Score (Lower is Better) | AUROC Error Prediction |",
        f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cond_name, mc in analysis.metacognitive_conditions.items():
        eval_label = "Primary Agent (Self / alpha)" if mc.evaluator == "self" else "Auditing Observer (gamma)"
        fmt_label = "Transcript-Only" if mc.memory_format == "transcript_only" else "Scaffolded Persistence"
        lines.append(
            f"| **{eval_label}** | `{fmt_label}` | {mc.total_trials} | {mc.mean_accuracy:.1%} | {mc.mean_confidence_pct:.1f}% | **{mc.brier_score:.4f}** | **{mc.auroc_error_prediction:.3f}** |"
        )

    mi = analysis.metacognitive_interaction
    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Item-Paired Metacognitive Estimands (Predicting Identical Target Decisions)",
        f"",
        f"| Item-Paired Estimand | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |",
        f"| :--- | :---: | :---: | :---: | :--- |",
        f"| **`Delta_AUROC_Transcript`** | **{mi.delta_auroc_transcript.point_estimate:+.3f}** | [{mi.delta_auroc_transcript.ci_lower_95:+.3f}, {mi.delta_auroc_transcript.ci_upper_95:+.3f}] | {mi.delta_auroc_transcript.permutation_p_value:.4f} (`{mi.delta_auroc_transcript.permutation_method}`) | {'**Significant Self-Framing Advantage**' if mi.delta_auroc_transcript.is_statistically_distinguishable and mi.delta_auroc_transcript.point_estimate > 0 else '**Null / Invariant**'} |",
        f"| **`Delta_AUROC_Scaffolded`** | **{mi.delta_auroc_scaffolded.point_estimate:+.3f}** | [{mi.delta_auroc_scaffolded.ci_lower_95:+.3f}, {mi.delta_auroc_scaffolded.ci_upper_95:+.3f}] | {mi.delta_auroc_scaffolded.permutation_p_value:.4f} (`{mi.delta_auroc_scaffolded.permutation_method}`) | {'**Significant Self-Framing Advantage**' if mi.delta_auroc_scaffolded.is_statistically_distinguishable and mi.delta_auroc_scaffolded.point_estimate > 0 else '**Null / Invariant**'} |",
        f"| **`Delta_Brier_Transcript`** | **{mi.delta_brier_transcript.point_estimate:+.4f}** | [{mi.delta_brier_transcript.ci_lower_95:+.4f}, {mi.delta_brier_transcript.ci_upper_95:+.4f}] | {mi.delta_brier_transcript.permutation_p_value:.4f} (`{mi.delta_brier_transcript.permutation_method}`) | {'**Self Calibrated Better**' if mi.delta_brier_transcript.is_statistically_distinguishable and mi.delta_brier_transcript.point_estimate > 0 else '**Null / Invariant**'} |",
        f"| **`Delta_Brier_Scaffolded`** | **{mi.delta_brier_scaffolded.point_estimate:+.4f}** | [{mi.delta_brier_scaffolded.ci_lower_95:+.4f}, {mi.delta_brier_scaffolded.ci_upper_95:+.4f}] | {mi.delta_brier_scaffolded.permutation_p_value:.4f} (`{mi.delta_brier_scaffolded.permutation_method}`) | {'**Self Calibrated Better**' if mi.delta_brier_scaffolded.is_statistically_distinguishable and mi.delta_brier_scaffolded.point_estimate > 0 else '**Null / Invariant**'} |",
        f"| **`Scaffolding_Metacognitive_Interaction`** | **{mi.scaffolding_metacognitive_interaction.point_estimate:+.3f}** | [{mi.scaffolding_metacognitive_interaction.ci_lower_95:+.3f}, {mi.scaffolding_metacognitive_interaction.ci_upper_95:+.3f}] | {mi.scaffolding_metacognitive_interaction.permutation_p_value:.4f} (`{mi.scaffolding_metacognitive_interaction.permutation_method}`) | {'**Scaffolded Persistence Alters Self-Observer Calibration**' if mi.scaffolding_metacognitive_interaction.is_statistically_distinguishable else '**Scaffolding-Invariant Metacognition**'} |",
        f"",
        f"---",
        f"",
        f"## 3. Scientific Gate Synthesis for Horizon 1 Closeout",
        f"",
        f"1. **Pre-Feedback Correctness Prediction:** Evaluates whether subjective confidence discriminates impending errors prior to external feedback.",
        f"2. **Item-Paired Framing Control:** Both Self and Observer evaluate the exact same first-order decisions made by `agent_alpha` under matched public evidence.",
        f"3. **Persistence Scaffolding Interaction:** Measures whether explicit Level-1 state shifts the metacognitive gap between internal self-framing and external observer evaluation.",
    ])

    return "\n".join(lines)


def run_e09_experiment(
    model_name: str = "qwen2.5:3b",
    seed: int = 42,
    phase: str = "exploratory",
    temperature: float = 0.0,
    episodes_count: Optional[int] = None,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete E09 benchmark suite across metacognitive screen conditions."""
    run_id = f"run_e09_meta_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{phase}"
    if dry_run:
        run_id += "_dryrun"

    out_dir = output_dir or Path(f"artifacts/e09_metacognitive_screen/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_results_dir = Path(f"results/e09_metacognitive_screen/{run_id}")
    canonical_results_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = episodes_count or (16 if phase == "confirmatory" else 4)

    print("=" * 70)
    print(f"EXPERIMENT E09: METACOGNITIVE CONTINUITY SCREEN (S09b)")
    print(f"Run ID: {run_id} | Model: {model_name} | Phase: {phase.upper()} | Seed: {seed} | Dry Run: {dry_run}")
    print(f"Episodes: {n_episodes}")
    print("=" * 70)

    if dry_run:
        backend = MockOwnershipBackend(model_name=model_name)
    else:
        backend = OllamaBackend(
            model_name=model_name,
            temperature=temperature,
            seed=seed,
        )

    digest = backend.get_digest()
    generator = OwnershipTaskGenerator(seed=seed)
    harness = OwnershipHarness(backend=backend)

    all_trials: List[OwnershipTrialResult] = []
    ep_manifests: List[Dict[str, Any]] = []

    for ep_idx in range(n_episodes):
        print(f"\n--- EXECUTING METACOGNITIVE EPISODE {ep_idx + 1}/{n_episodes} ---")
        episode = generator.generate_episode(twin_idx=ep_idx, seed=seed)
        ep_trials = harness.execute_e09_metacognitive_screen(episode=episode)
        all_trials.extend(ep_trials)

        ep_manifests.append({
            "episode_id": episode.episode_id,
            "twin_index": ep_idx,
            "trials_recorded": len(ep_trials),
        })
        print(f"  -> Episode {episode.episode_id}: {len(ep_trials)} metacognitive trials recorded.")

    print(f"\nTotal Metacognitive Trials Recorded: {len(all_trials)}")

    analysis = analyze_ownership_results(trials=all_trials, num_bootstrap=2000, seed=seed)

    manifest = {
        "run_id": run_id,
        "target_model": model_name,
        "model_digest": digest,
        "phase": phase,
        "seed": seed,
        "temperature": temperature,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "total_episodes": len(ep_manifests),
        "total_trials": len(all_trials),
    }

    df_trials = pd.DataFrame([asdict(t) for t in all_trials])
    report_md = generate_e09_markdown_report(
        manifest=manifest,
        analysis=analysis,
        ep_manifests=ep_manifests,
        raw_df=df_trials,
    )

    for target_dir in [out_dir, canonical_results_dir]:
        with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        summary_payload = {
            "manifest": manifest,
            "analysis": {
                "metacognitive_conditions": {k: asdict(v) for k, v in analysis.metacognitive_conditions.items()},
                "metacognitive_interaction": {
                    "delta_auroc_transcript": asdict(analysis.metacognitive_interaction.delta_auroc_transcript),
                    "delta_auroc_scaffolded": asdict(analysis.metacognitive_interaction.delta_auroc_scaffolded),
                    "delta_brier_transcript": asdict(analysis.metacognitive_interaction.delta_brier_transcript),
                    "delta_brier_scaffolded": asdict(analysis.metacognitive_interaction.delta_brier_scaffolded),
                    "scaffolding_metacognitive_interaction": asdict(analysis.metacognitive_interaction.scaffolding_metacognitive_interaction),
                },
            },
            "episodes": ep_manifests,
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
    print(f"EXPERIMENT E09 BENCHMARK COMPLETE")
    print(f"Artifacts written to: {out_dir}")
    print(f"Canonical Results written to: {canonical_results_dir}")
    print("=" * 70 + "\n")

    return {
        "manifest": manifest,
        "analysis": analysis,
        "report_md": report_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E09: Metacognitive Screen (Sprint S09b)")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--phase", type=str, default="exploratory", choices=["exploratory", "confirmatory"], help="Experiment phase")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--episodes", type=int, default=None, help="Number of multi-source episodes")
    parser.add_argument("--dry-run", action="store_true", help="Run with mock backend")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()

    run_e09_experiment(
        model_name=args.model,
        seed=args.seed,
        phase=args.phase,
        temperature=args.temperature,
        episodes_count=args.episodes,
        dry_run=args.dry_run,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
