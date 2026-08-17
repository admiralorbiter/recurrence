"""Experiment E09: Metacognitive Continuity & Future-Failure Prediction Screen (Sprint S09b)."""

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
        f"# Experiment E09: Metacognitive Continuity & Future-Failure Screen Report (Sprint S09b)",
        f"",
        f"**Run ID:** `{manifest['run_id']}`  ",
        f"**Model:** `{manifest['target_model']}` (`{manifest['model_digest'][:12]}...`)  ",
        f"**Phase:** `{manifest['phase'].upper()}` (Seed: `{manifest['seed']}`)  ",
        f"**Date:** {manifest['start_time']}  ",
        f"**Scope:** {manifest['total_episodes']} Multi-Source Episodes | {manifest['total_trials']} Total Metacognitive Probes  ",
        f"**Primary Question:** *Does scaffolded persistence improve an agent's metacognitive calibration and future-failure resolution over an external observer?*  ",
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

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Self vs Observer Metacognitive Advantage ($\\Delta_{{\\text{{meta}}}}$)",
        f"",
        f"- **Self vs Observer Metacognitive Advantage (Transcript-Only):** **{analysis.self_vs_observer_advantage_transcript:+.3f} AUROC**",
        f"- **Self vs Observer Metacognitive Advantage (Scaffolded State):** **{analysis.self_vs_observer_advantage_scaffolded:+.3f} AUROC**",
        f"",
        f"---",
        f"",
        f"## 3. Scientific Gate Synthesis for Horizon 1 Closeout",
        f"",
        f"1. **Metacognitive Calibration:** Does the agent accurately calibrate confidence against its actual empirical error distribution?",
        f"2. **Future-Failure Resolution:** Can subjective confidence discriminate impending attribution errors prior to feedback?",
        f"3. **Privileged Self-Access:** Does internal self-evaluation provide an error-predictive advantage over an external observer inspecting identical scaffolded representations?",
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
                "self_vs_observer_advantage_transcript": analysis.self_vs_observer_advantage_transcript,
                "self_vs_observer_advantage_scaffolded": analysis.self_vs_observer_advantage_scaffolded,
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
