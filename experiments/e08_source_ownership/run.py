"""Experiment E08: Source Attribution, Self/Other Memory Ownership, and Agency Boundaries (Sprint S09a)."""

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


class MockOwnershipBackend:
    """Mock backend for dry-run verification of Sprint S09."""

    def __init__(self, model_name: str = "qwen2.5:3b") -> None:
        self.model_name = model_name

    def get_digest(self) -> str:
        return "mock_digest_dryrun_e08"

    def step(self, prompt: str, format: Optional[Dict[str, Any]] = None) -> tuple[str, str, Dict[str, Any]]:
        if format and "confidence_percentage" in format.get("properties", {}):
            text = json.dumps({"confidence_percentage": 80, "reasoning": "mock_confidence"})
        elif format and len(format.get("properties", {}).get("answer", {}).get("enum", [])) == 5:
            text = json.dumps({"answer": "A"})
        else:
            text = json.dumps({"answer": "A"})

        metadata = {
            "prompt_eval_count": len(prompt) // 4,
            "eval_count": len(text) // 4,
            "total_duration_ms": 5.0,
        }
        return text, "hash_mock_e08", metadata


def generate_e08_markdown_report(
    manifest: Dict[str, Any],
    analysis: S09AnalysisSummary,
    ep_manifests: List[Dict[str, Any]],
    raw_df: pd.DataFrame,
) -> str:
    """Generate publication-ready Markdown report for Experiment E08."""
    lines = [
        f"# Experiment E08: Source Attribution, Self/Other Ownership & Agency Boundaries Report (Sprint S09a)",
        f"",
        f"**Run ID:** `{manifest['run_id']}`  ",
        f"**Model:** `{manifest['target_model']}` (`{manifest['model_digest'][:12]}...`)  ",
        f"**Phase:** `{manifest['phase'].upper()}` (Seed: `{manifest['seed']}`)  ",
        f"**Date:** {manifest['start_time']}  ",
        f"**Scope:** {manifest['total_episodes']} Multi-Source Episodes | {manifest['total_trials']} Total Ownership Intervention Trials  ",
        f"**Primary Question:** *Does the model reliably track epistemic source origin, maintain self-other agency boundaries, and resist pressure-induced narrative revision?*  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Source Attribution Breakdown (5AFC)",
        f"",
        f"- **Overall Source Attribution Accuracy (5AFC):** **{analysis.attribution_breakdown.overall_accuracy:.1%}**",
        f"- **Self Attribution Accuracy (`agent_alpha`):** **{analysis.attribution_breakdown.self_accuracy:.1%}**",
        f"- **Environment Attribution Accuracy (`telemetry_sensor`):** **{analysis.attribution_breakdown.environment_accuracy:.1%}**",
        f"- **Experimenter Attribution Accuracy (`human_controller`):** **{analysis.attribution_breakdown.experimenter_accuracy:.1%}**",
        f"- **Peer Agent Attribution Accuracy (`agent_beta`):** **{analysis.attribution_breakdown.peer_agent_accuracy:.1%}**",
        f"- **Observer Attribution Accuracy (`auditor_gamma`):** **{analysis.attribution_breakdown.observer_accuracy:.1%}**",
        f"- **Self-Other Confusion Rate ($SOCR$):** **{analysis.attribution_breakdown.self_other_confusion_rate:.1%}**",
        f"",
        f"---",
        f"",
        f"## 2. Self vs Peer Conflict (Operative Belief & Agency)",
        f"",
        f"- **Self Operative Value Adoption Rate:** **{analysis.self_peer_belief_self_rate:.1%}**",
        f"- **Peer Value Adoption Rate:** **{analysis.self_peer_belief_peer_rate:.1%}**",
        f"- **Self-Allegiance Contrast under Conflict ($\\Delta_{{\\text{{self-peer}}}}$):** **{analysis.self_peer_allegiance_contrast:+.1%}**",
        f"",
        f"---",
        f"",
        f"## 3. Cue-Conflict Factorial ($2 \\times 2$ Explicit Tag $\\times$ Narrative Actor)",
        f"",
        f"- **Congruent Tag + Narrative Accuracy:** **{analysis.cue_conflict.tag_congruent_accuracy:.1%}**",
        f"- **Tag Leverage Under Conflict ($P(\\text{{Answer}} = \\text{{Tag}})$):** **{analysis.cue_conflict.tag_leverage:.1%}**",
        f"- **Narrative Leverage Under Conflict ($P(\\text{{Answer}} = \\text{{Narrative}})$):** **{analysis.cue_conflict.narrative_leverage:.1%}**",
        f"- **Tag vs Narrative Contrast:** **{analysis.cue_conflict.tag_narrative_contrast:+.1%}**",
        f"",
        f"---",
        f"",
        f"## 4. Channel Factorial ($2 \\times 2$ Transcript Tags $\\times$ State Ledger)",
        f"",
        f"- **Tags Present + Ledger Present:** **{analysis.channel_factorial.tags_present_ledger_present:.1%}**",
        f"- **Tags Present + Ledger Stripped:** **{analysis.channel_factorial.tags_present_ledger_absent:.1%}**",
        f"- **Tags Stripped + Ledger Present:** **{analysis.channel_factorial.tags_absent_ledger_present:.1%}**",
        f"- **Tags Stripped + Ledger Stripped:** **{analysis.channel_factorial.tags_absent_ledger_absent:.1%}**",
        f"- **Transcript Tag Marginal Effect:** **{analysis.channel_factorial.transcript_tag_marginal_effect:+.1%}**",
        f"- **Source Ledger Marginal Effect:** **{analysis.channel_factorial.source_ledger_marginal_effect:+.1%}**",
        f"",
        f"---",
        f"",
        f"## 5. Self-Referential Framing & Pressure Revision Susceptibility",
        f"",
        f"- **Self-Referential Framing Accuracy (*\"What did you execute?\"*):** **{analysis.framing_self_referential_acc:.1%}**",
        f"- **3rd-Person Framing Accuracy (*\"What did agent_alpha execute?\"*):** **{analysis.framing_3rd_person_acc:.1%}**",
        f"- **Framing Discrepancy Gap:** **{analysis.framing_discrepancy_gap:.1%}**",
        f"- **Ownership Revision Susceptibility ($ORS$ after False Audit Challenge):** **{analysis.ownership_revision_susceptibility:.1%}**",
        f"",
        f"---",
        f"",
        f"## 6. Scientific Gate Synthesis for Sprint S09a",
        f"",
        f"1. **Epistemic Origin Resolution:** Can the model distinguish facts asserted by self vs peer vs external sources under source-neutral semantic framing?",
        f"2. **Self-Other Boundary:** Does the model protect its own state decisions against peer assertions, and does it resist falsely adopting peer actions as its own?",
        f"3. **Provenance Channel Ownership:** Does source tracking rely on episodic metadata tags, explicit state ledgers, or narrative context?",
    ]
    return "\n".join(lines)


def run_e08_experiment(
    model_name: str = "qwen2.5:3b",
    seed: int = 42,
    phase: str = "exploratory",
    temperature: float = 0.0,
    episodes_count: Optional[int] = None,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete E08 benchmark suite across multi-source episodes."""
    run_id = f"run_e08_owner_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{phase}"
    if dry_run:
        run_id += "_dryrun"

    out_dir = output_dir or Path(f"artifacts/e08_source_ownership/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_results_dir = Path(f"results/e08_source_ownership/{run_id}")
    canonical_results_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = episodes_count or (16 if phase == "confirmatory" else 4)

    print("=" * 70)
    print(f"EXPERIMENT E08: SOURCE ATTRIBUTION & OWNERSHIP BOUNDARIES (S09a)")
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
        print(f"\n--- EXECUTING EPISODE {ep_idx + 1}/{n_episodes} ---")
        episode = generator.generate_episode(twin_idx=ep_idx, seed=seed)
        ep_trials = harness.execute_e08_episode(episode=episode)
        all_trials.extend(ep_trials)

        ep_manifests.append({
            "episode_id": episode.episode_id,
            "twin_index": ep_idx,
            "trials_recorded": len(ep_trials),
        })
        print(f"  -> Episode {episode.episode_id}: {len(ep_trials)} trials recorded.")

    print(f"\nTotal Trials Recorded: {len(all_trials)}")

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
    report_md = generate_e08_markdown_report(
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
                "attribution_breakdown": asdict(analysis.attribution_breakdown),
                "cue_conflict": asdict(analysis.cue_conflict),
                "channel_factorial": asdict(analysis.channel_factorial),
                "self_peer_belief_self_rate": analysis.self_peer_belief_self_rate,
                "self_peer_belief_peer_rate": analysis.self_peer_belief_peer_rate,
                "self_peer_allegiance_contrast": analysis.self_peer_allegiance_contrast,
                "framing_self_referential_acc": analysis.framing_self_referential_acc,
                "framing_3rd_person_acc": analysis.framing_3rd_person_acc,
                "framing_discrepancy_gap": analysis.framing_discrepancy_gap,
                "ownership_revision_susceptibility": analysis.ownership_revision_susceptibility,
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
    print(f"EXPERIMENT E08 BENCHMARK COMPLETE")
    print(f"Artifacts written to: {out_dir}")
    print(f"Canonical Results written to: {canonical_results_dir}")
    print("=" * 70 + "\n")

    return {
        "manifest": manifest,
        "analysis": analysis,
        "report_md": report_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E08: Source Attribution & Agency Boundaries (Sprint S09a)")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--phase", type=str, default="exploratory", choices=["exploratory", "confirmatory"], help="Experiment phase")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--episodes", type=int, default=None, help="Number of multi-source episodes")
    parser.add_argument("--dry-run", action="store_true", help="Run with mock backend")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()

    run_e08_experiment(
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
