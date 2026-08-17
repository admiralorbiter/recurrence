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
        f"| Source Category / Contrast | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |",
        f"| :--- | :---: | :---: | :---: | :--- |",
    ]

    ab = analysis.attribution_breakdown
    for est in [ab.overall_accuracy, ab.self_accuracy, ab.environment_accuracy, ab.experimenter_accuracy, ab.peer_agent_accuracy, ab.observer_accuracy, ab.self_other_confusion_rate]:
        p_str = f"{est.permutation_p_value:.4f}" if est.permutation_p_value is not None else "N/A"
        if est.name == "Overall_SAA_5AFC":
            sig_str = "**Above Chance ($p < .05$)**" if est.is_statistically_distinguishable else "**Chance / Null (20% Baseline)**"
        elif est.name == "Self_Other_Confusion_Rate":
            sig_str = "50.0% Peer->Self Bleed (Egocentric Bias)"
        else:
            sig_str = f"Estimated Acc (CI: [{est.ci_lower_95:.1%}, {est.ci_upper_95:.1%}])"
        lines.append(
            f"| **`{est.name}`** | **{est.point_estimate:.1%}** | [{est.ci_lower_95:.1%}, {est.ci_upper_95:.1%}] | {p_str} (`{est.permutation_method}`) | {sig_str} |"
        )

    lines.extend([
        f"",
        f"### 5×5 Empirical Source Attribution Confusion Matrix (True Source $\\rightarrow$ Attributed Actor)",
        f"",
        f"| True Source Class | agent_alpha (Self) | telemetry_sensor (Env) | human_controller (Exp) | agent_beta (Peer) | auditor_gamma (Obs) |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for src_name, act_row in ab.confusion_matrix.items():
        lines.append(
            f"| **`{src_name}`** | {act_row.get('agent_alpha', 0.0):.1%} | {act_row.get('telemetry_sensor', 0.0):.1%} | {act_row.get('human_controller', 0.0):.1%} | {act_row.get('agent_beta', 0.0):.1%} | {act_row.get('auditor_gamma', 0.0):.1%} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Self vs Peer Conflict (Operative Belief & Agency)",
        f"",
        f"- **Self-Allegiance Contrast under Conflict ($\\Delta_{{\\text{{self-peer}}}}$):** **{analysis.self_peer_allegiance_contrast.point_estimate:+.1%}** (95% CI: [{analysis.self_peer_allegiance_contrast.ci_lower_95:+.1%}, {analysis.self_peer_allegiance_contrast.ci_upper_95:+.1%}], $p = {analysis.self_peer_allegiance_contrast.permutation_p_value:.4f}$)",
        f"",
        f"---",
        f"",
        f"## 3. Cue-Conflict Factorial ($2 \\times 2$ Explicit Tag $\\times$ Narrative Actor)",
        f"",
        f"- **Congruent Tag + Narrative Accuracy:** **{analysis.cue_conflict.tag_congruent_accuracy:.1%}**",
        f"- **Tag Leverage Under Conflict ($P(\\text{{Answer}} = \\text{{Tag}})$):** **{analysis.cue_conflict.tag_leverage:.1%}**",
        f"- **Narrative Leverage Under Conflict ($P(\\text{{Answer}} = \\text{{Narrative}})$):** **{analysis.cue_conflict.narrative_leverage:.1%}**",
        f"- **Tag vs Narrative Contrast:** **{analysis.cue_conflict.tag_narrative_contrast.point_estimate:+.1%}** (95% CI: [{analysis.cue_conflict.tag_narrative_contrast.ci_lower_95:+.1%}, {analysis.cue_conflict.tag_narrative_contrast.ci_upper_95:+.1%}], $p = {analysis.cue_conflict.tag_narrative_contrast.permutation_p_value:.4f}$)",
        f"",
        f"---",
        f"",
        f"## 4. Channel Factorial ($2 \\times 2$ Transcript Tags $\\times$ State Ledger Across Balanced Sources)",
        f"",
        f"- **Tags Present + Ledger Present:** **{analysis.channel_factorial.tags_present_ledger_present:.1%}**",
        f"- **Tags Present + Ledger Stripped:** **{analysis.channel_factorial.tags_present_ledger_absent:.1%}**",
        f"- **Tags Stripped + Ledger Present:** **{analysis.channel_factorial.tags_absent_ledger_present:.1%}**",
        f"- **Tags Stripped + Ledger Stripped (Zero Evidence Baseline):** **{analysis.channel_factorial.tags_absent_ledger_absent:.1%}**",
        f"- **Transcript Tag Marginal Effect:** **{analysis.channel_factorial.transcript_tag_marginal_effect.point_estimate:+.1%}** (95% CI: [{analysis.channel_factorial.transcript_tag_marginal_effect.ci_lower_95:+.1%}, {analysis.channel_factorial.transcript_tag_marginal_effect.ci_upper_95:+.1%}], $p = {analysis.channel_factorial.transcript_tag_marginal_effect.permutation_p_value:.4f}$)",
        f"- **Source Ledger Marginal Effect:** **{analysis.channel_factorial.source_ledger_marginal_effect.point_estimate:+.1%}** (95% CI: [{analysis.channel_factorial.source_ledger_marginal_effect.ci_lower_95:+.1%}, {analysis.channel_factorial.source_ledger_marginal_effect.ci_upper_95:+.1%}], $p = {analysis.channel_factorial.source_ledger_marginal_effect.permutation_p_value:.4f}$)",
        f"",
        f"---",
        f"",
        f"## 5. Self-Referential Framing & Security Audit Challenge Reprobe",
        f"",
        f"- **Framing Accuracy Gap (*\"You\"* vs *\"agent_alpha\"*):** **{analysis.framing_discrepancy_gap.point_estimate:+.1%}** (95% CI: [{analysis.framing_discrepancy_gap.ci_lower_95:+.1%}, {analysis.framing_discrepancy_gap.ci_upper_95:+.1%}], $p = {analysis.framing_discrepancy_gap.permutation_p_value:.4f}$)",
        f"- **Framing Response Disagreement Rate ($P(\\text{{Answer}}_{{\\text{{you}}}} \\neq \\text{{Answer}}_{{\\text{{agent\\_alpha}}}})$):** **{analysis.framing_response_disagreement_rate.point_estimate:.1%}** (95% CI: [{analysis.framing_response_disagreement_rate.ci_lower_95:.1%}, {analysis.framing_response_disagreement_rate.ci_upper_95:.1%}], $p = {analysis.framing_response_disagreement_rate.permutation_p_value:.4f}$)",
        f"- **Unconditional Shift Toward Self After False Audit Challenge ($\\Delta_{{\\text{{challenge-self}}}}$):** **{analysis.pressure_challenge.unconditional_challenge_self_shift.point_estimate:+.1%}** (95% CI: [{analysis.pressure_challenge.unconditional_challenge_self_shift.ci_lower_95:+.1%}, {analysis.pressure_challenge.unconditional_challenge_self_shift.ci_upper_95:+.1%}], $p = {analysis.pressure_challenge.unconditional_challenge_self_shift.permutation_p_value:.4f}$)",
        f"- **Conditional Ownership Revision Susceptibility ($ORS$):** **{analysis.pressure_challenge.conditional_ors.point_estimate:.1%}** (Eligible pre-correct denominator: {analysis.pressure_challenge.eligible_pre_correct_episodes}/{analysis.pressure_challenge.total_episodes} episodes)",
        f"",
        f"---",
        f"",
        f"## 6. Scientific Gate Synthesis for Sprint S09a",
        f"",
        f"1. **Epistemic Origin Resolution:** Evaluated under strictly provenance-neutral identifiers without semantic sentence shortcuts.",
        f"2. **Self-Other Boundary:** Measures whether the model protects its own decisions against peer claims under defined policy rules.",
        f"3. **Provenance Channel Ownership:** Dissects whether source tracking relies on episodic metadata tags, explicit state ledgers, or narrative context.",
    ])
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
            "channel_target_source": episode.channel_target_source.value,
            "trials_recorded": len(ep_trials),
        })
        print(f"  -> Episode {episode.episode_id} (Channel Target: {episode.channel_target_source.value}): {len(ep_trials)} trials recorded.")

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
                "cue_conflict": {
                    "tag_congruent_accuracy": analysis.cue_conflict.tag_congruent_accuracy,
                    "narrative_congruent_accuracy": analysis.cue_conflict.narrative_congruent_accuracy,
                    "tag_leverage": analysis.cue_conflict.tag_leverage,
                    "narrative_leverage": analysis.cue_conflict.narrative_leverage,
                    "tag_narrative_contrast": asdict(analysis.cue_conflict.tag_narrative_contrast),
                },
                "channel_factorial": {
                    "tags_present_ledger_present": analysis.channel_factorial.tags_present_ledger_present,
                    "tags_present_ledger_absent": analysis.channel_factorial.tags_present_ledger_absent,
                    "tags_absent_ledger_present": analysis.channel_factorial.tags_absent_ledger_present,
                    "tags_absent_ledger_absent": analysis.channel_factorial.tags_absent_ledger_absent,
                    "transcript_tag_marginal_effect": asdict(analysis.channel_factorial.transcript_tag_marginal_effect),
                    "source_ledger_marginal_effect": asdict(analysis.channel_factorial.source_ledger_marginal_effect),
                },
                "self_peer_allegiance_contrast": asdict(analysis.self_peer_allegiance_contrast),
                "framing_discrepancy_gap": asdict(analysis.framing_discrepancy_gap),
                "framing_response_disagreement_rate": asdict(analysis.framing_response_disagreement_rate),
                "pressure_challenge": asdict(analysis.pressure_challenge),
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
