"""Reanalyze E07 runs using the enhanced S08 analysis module."""

from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd

from recurrence.analysis.intervention_metrics import (
    analyze_state_intervention_results,
    StateInterventionAnalysisSummary,
)
from recurrence.loop.intervention_experiment import InterventionTrialResult
from experiments.e07_state_interventions.run import generate_e07_markdown_report


def reanalyze_run(run_dir: Path) -> None:
    print(f"Reanalyzing {run_dir}...")
    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
        old_summary = json.load(f)
    twin_manifests = old_summary.get("twins", [])

    trials = []
    with open(run_dir / "trials.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                trials.append(InterventionTrialResult(**d))

    analysis = analyze_state_intervention_results(
        trials=trials,
        num_bootstrap=2000,
        seed=manifest.get("seed", 42),
    )

    df_trials = pd.DataFrame([asdict(t) for t in trials])

    report_md = generate_e07_markdown_report(
        manifest=manifest,
        analysis=analysis,
        twin_manifests=twin_manifests,
        raw_df=df_trials,
    )

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

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Done reanalyzing {run_dir}")


def main() -> None:
    results_base = Path("results/e07_state_interventions")
    for run_dir in sorted(results_base.glob("run_e07_*")):
        if run_dir.is_dir():
            reanalyze_run(run_dir)


if __name__ == "__main__":
    main()
