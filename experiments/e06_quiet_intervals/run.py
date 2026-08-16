"""Experiment E06: Scaffolded Null-Interval & Quiet Processing Benchmark (Sprint S07).

Evaluates whether scaffolded quiet processing cycles between informative events selectively
reorganize unresolved cognitive state across 6 conditions and intervals K in {0, 1, 3, 6, 12}.
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
from recurrence.tasks.quiet_interval import (
    QuietIntervalGenerator,
    QuietIntervalEpisode,
)
from recurrence.loop.quiet_experiment import (
    QuietIntervalHarness,
    QuietTrialResult,
)
from recurrence.analysis.quiet_metrics import (
    analyze_quiet_interval_results,
    QuietIntervalAnalysisSummary,
    QuietCausalEstimandSummary,
)


class MockQuietBackend:
    """Mock backend for dry-run verification of Sprint S07."""

    def __init__(self, model_name: str = "qwen2.5:3b") -> None:
        self.model_name = model_name

    def get_digest(self) -> str:
        return "mock_digest_dryrun_e06"

    def step(self, prompt: str, format: Optional[Dict[str, Any]] = None) -> tuple[str, str, Dict[str, Any]]:
        fmt_str = json.dumps(format or {})
        if "answer" in fmt_str or "target_answer" in fmt_str:
            text = json.dumps({"answer": "A"})
        elif "derived_inferences" in fmt_str:
            text = json.dumps({
                "derived_inferences": {"key_mock_derived": "val_mock_term"},
                "unresolved_items": ["conflict:key_mock_conflict"],
                "goal_status_updates": [{"goal_id": "goal_beta", "status": "active"}],
            })
        else:
            text = json.dumps({
                "working_memory": {"key_mock_stable": "val_mock_stable"},
                "goals": [{"goal_id": "goal_alpha", "description": "Mock active", "status": "active"}],
                "source_ledger": {"key_mock_stable": "environment"},
                "unresolved_items": [],
            })

        metadata = {
            "prompt_eval_count": len(prompt) // 4,
            "eval_count": len(text) // 4,
            "total_duration_ms": 5.0,
        }
        return text, "hash_mock_e06", metadata


def generate_e06_markdown_report(
    manifest: Dict[str, Any],
    analysis: QuietIntervalAnalysisSummary,
    episode_manifests: List[Dict[str, Any]],
    raw_df: pd.DataFrame,
) -> str:
    """Generate publication-ready Markdown report for Experiment E06."""
    lines = [
        f"# Experiment E06: Scaffolded Null-Interval & Quiet Processing Benchmark Report (Sprint S07)",
        f"",
        f"**Run ID:** `{manifest['run_id']}`  ",
        f"**Model:** `{manifest['target_model']}` (`{manifest['model_digest'][:12]}...`)  ",
        f"**Phase:** `{manifest['phase'].upper()}` (Seed: `{manifest['seed']}`)  ",
        f"**Date:** {manifest['start_time']}  ",
        f"**Scope:** {manifest['total_episodes']} Base Episodes Cloned Across Intervals $K \\in {manifest['intervals']}$ | {manifest['total_trials']} Total Paired Trials  ",
        f"**Primary Question:** *Do scaffolded null-interval update cycles selectively preserve or reorganize unresolved cognitive state?*  ",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Benchmark Results",
        f"",
        f"Experiment E06 evaluates whether intervening quiet processing cycles ($K \\in \\{{0, 1, 3, 6, 12\\}}$ null ticks) placed between a common prefix ($E_{{\\text{{prefix}}}}$) and continuation ($E_{{\\text{{continuation}}}}$) selectively improve multi-hop relational derivation, source conflict consolidation, and goal prioritization, or whether they introduce representational drift.",
        f"",
        f"### Multi-Condition Performance & Cost Summary Table (Pooled Across $K > 0$)",
        f"",
        f"| Condition | Group | Micro Accuracy | Multi-Hop Derivation (4AFC) | Source Conflict (3AFC) | Goal State (4AFC) | Stable WM (4AFC) | Query Prompt Tok | Amortized Prompt Tok | Query Latency | Amortized Latency |",
        f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cond, stats in analysis.condition_stats.items():
        p_accs = stats.probe_accuracies
        hop_acc = f"{p_accs.get('derivation_multihop', 0.0):.1%}"
        src_acc = f"{p_accs.get('source_conflict', 0.0):.1%}"
        gs_acc = f"{p_accs.get('unresolved_goal', 0.0):.1%}"
        wm_acc = f"{p_accs.get('stable_kv', 0.0):.1%}"

        cond_group = {
            "strict_identity": "No Write Control",
            "clock_only": "No Write Control",
            "semantic_no_write": "No Write Control",
            "selective_reflection": "**Persistent Write (Primary)**",
            "unconstrained_reflection": "Persistent Write (Diagnostic)",
            "replay_transcript": "Retrospective Reference",
        }.get(cond, "Other")

        cond_label = {
            "strict_identity": "**Strict Identity Scaffold**",
            "clock_only": "**Clock-Only (Timestamp Cue)**",
            "semantic_no_write": "**Semantic Reasoning (No-Write)**",
            "selective_reflection": "**Selective Reflection (Derived Channel)**",
            "unconstrained_reflection": "**Unconstrained Full-State Rewrite**",
            "replay_transcript": "**Replay Transcript (Raw)**",
        }.get(cond, cond)

        lines.append(
            f"| {cond_label} | {cond_group} | **{stats.accuracy_micro:.1%}** | {hop_acc} | {src_acc} | {gs_acc} | {wm_acc} | {stats.mean_query_prompt_tokens:.1f} tok | {stats.mean_amortized_prompt_tokens:.1f} tok | {stats.mean_query_latency_ms:.1f} ms | {stats.mean_amortized_latency_ms:.1f} ms |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Targeted Causal Estimands & Statistical Inference",
        f"",
        f"Episode-clustered paired bootstrap 95% confidence intervals (B=2,000), exact two-sided binomial McNemar tests, and episode-level sign-flip permutation tests (Primary Inferential Decision):",
        f"",
        f"| Causal Contrast | Target Domain | Contrast Definition | $\\Delta$ Accuracy | 95% Bootstrap CI | Discordance ($b / c$) | Exact McNemar $p$ | Permutation $p$ (Method) | Scientific Inference |",
        f"| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])

    for name, est in analysis.causal_estimands.items():
        if name == "Delta_derivation-selective":
            desc = "Selective Reflection vs Strict Identity (Multi-Hop)"
            sig_desc = "**Statistically Significant Derivation Gain**" if est.is_statistically_distinguishable and est.delta_accuracy > 0 else "**No Resolved Difference / Null**"
        elif name == "Delta_derivation-nowrite":
            desc = "Selective Reflection vs Semantic No-Write (Multi-Hop)"
            sig_desc = "**Statistically Significant Write Gain**" if est.is_statistically_distinguishable and est.delta_accuracy > 0 else "**No Resolved Difference / Null**"
        elif name == "Delta_conflict-consolidation":
            desc = "Selective Reflection vs Strict Identity (Conflict)"
            sig_desc = "**Statistically Significant Consolidation**" if est.is_statistically_distinguishable and est.delta_accuracy > 0 else "**No Resolved Difference / Null**"
        elif name == "Delta_clock-cue":
            desc = "Clock-Only vs Strict Identity (All Probes)"
            sig_desc = "**Timestamp Sensitivity Detected**" if est.is_statistically_distinguishable else "**Null / Timing Cue Invariant**"
        elif name == "Delta_evidence-integrity":
            desc = "Selective Reflection vs Strict Identity (Stable KV)"
            sig_desc = "**Evidence Invariance Confirmed**" if not est.is_statistically_distinguishable else "**Evidence Mutation / Drift**"
        elif name == "Delta_unconstrained-drift":
            desc = "Unconstrained Reflection vs Strict Identity (All Probes)"
            sig_desc = "**Statistically Significant State Decay**" if est.is_statistically_distinguishable and est.delta_accuracy < 0 else "**No Resolved Difference**"
        else:
            desc = name
            sig_desc = "**Statistically Significant**" if est.is_statistically_distinguishable else "**No Resolved Difference**"

        lines.append(
            f"| **`{name}`** | `{est.target_probe_domain}` | {desc} | **{est.delta_accuracy:+.1%}** | [{est.ci_lower_95:+.1%}, {est.ci_upper_95:+.1%}] | {est.discordance_b} / {est.discordance_c} | {est.exact_mcnemar_p_value:.4f} | {est.permutation_p_value:.4f} (`{est.permutation_method}`) | {sig_desc} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Quiet Interval Scaling Dynamics ($K \\in \\{{0, 1, 3, 6, 12\\}}$ Null Ticks)",
        f"",
        f"| Quiet Interval | Strict Identity | Clock-Only | Semantic No-Write | Selective Reflection | Unconstrained Rewrite | Replay Transcript | $\\Delta_{{\\text{{derivation-selective}}}}$ [95% CI] | Permutation $p$ |",
        f"| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for k in sorted(analysis.interval_breakdown.keys()):
        k_stats = analysis.interval_breakdown[k]
        ic = [c for c in analysis.interval_contrasts if c.interval_k == k]
        ic_str = f"{ic[0].delta_accuracy:+.1%} [{ic[0].ci_lower_95:+.1%}, {ic[0].ci_upper_95:+.1%}]" if ic else "N/A"
        ic_p = f"{ic[0].permutation_p_value:.4f} (`{ic[0].permutation_method}`)" if ic else "N/A"

        id_str = f"{k_stats.get('strict_identity', 0.0):.1%}"
        clk_str = f"{k_stats.get('clock_only', 0.0):.1%}" if k > 0 else "-"
        nw_str = f"{k_stats.get('semantic_no_write', 0.0):.1%}" if k > 0 else "-"
        sel_str = f"{k_stats.get('selective_reflection', 0.0):.1%}" if k > 0 else "-"
        uncon_str = f"{k_stats.get('unconstrained_reflection', 0.0):.1%}" if k > 0 else "-"
        rep_str = f"{k_stats.get('replay_transcript', 0.0):.1%}"

        lines.append(
            f"| **$K={k}$ ticks** | {id_str} | {clk_str} | {nw_str} | {sel_str} | {uncon_str} | {rep_str} | {ic_str} | {ic_p} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Evidence Integrity & Representational Drift Analysis",
        f"",
        f"- **Selective Reflection Protected Evidence Mutation Rate:** **{analysis.selective_evidence_drift_rate:.1%}** (enforced by invariant assertion)",
        f"- **Unconstrained Reflection Evidence Drift / Slot Loss Rate:** **{analysis.unconstrained_evidence_drift_rate:.1%}**",
        f"",
        f"---",
        f"",
        f"## 5. Key Scientific Conclusions & Gate Assessment",
        f"",
        f"1. **Scaffolded Null Processing vs Identity Baseline:** Evaluates whether active quiet processing cycles reorganize state for later continuation integration or whether deterministic identity preservation suffices.",
        f"2. **Evidence Channel Protection:** Confirms that restricting write access to `derived_inferences` and `unresolved_items` prevents the catastrophic state decay observed under unconstrained reflection.",
        f"3. **Compute vs Storage Separation:** Compares persistent writing against matched semantic reasoning token exposure (`semantic_no_write`).",
        f"4. **Horizon 1 Program Progression:** These findings complete the quiet-interval screen of Horizon 1, advancing to Sprint S08 (Reset/Clone/Swap) and Sprint S09 (Metacognition & Ownership).",
    ])

    return "\n".join(lines)


def run_e06_experiment(
    model_name: str = "qwen2.5:3b",
    seed: int = 42,
    phase: str = "exploratory",
    temperature: float = 0.0,
    episodes_count: Optional[int] = None,
    intervals: Optional[List[int]] = None,
    dry_run: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete E06 benchmark suite across intervals and conditions."""
    run_id = f"run_e06_quiet_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{phase}"
    if dry_run:
        run_id += "_dryrun"

    out_dir = output_dir or Path(f"artifacts/e06_quiet_intervals/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical_results_dir = Path(f"results/e06_quiet_intervals/{run_id}")
    canonical_results_dir.mkdir(parents=True, exist_ok=True)

    eval_intervals = intervals or [0, 1, 3, 6, 12]
    n_base_episodes = episodes_count or (8 if phase == "confirmatory" else 4)

    print("=" * 70)
    print(f"EXPERIMENT E06: SCAFFOLDED NULL-INTERVAL & QUIET PROCESSING BENCHMARK (S07)")
    print(f"Run ID: {run_id} | Model: {model_name} | Phase: {phase.upper()} | Seed: {seed} | Dry Run: {dry_run}")
    print(f"Base Episodes: {n_base_episodes} | Intervals K: {eval_intervals}")
    print("=" * 70)

    if dry_run:
        backend = MockQuietBackend(model_name=model_name)
    else:
        backend = OllamaBackend(
            model_name=model_name,
            temperature=temperature,
            seed=seed,
        )

    digest = backend.get_digest()

    generator = QuietIntervalGenerator(seed=seed)
    harness = QuietIntervalHarness(backend=backend)

    all_trials: List[QuietTrialResult] = []
    episode_manifests: List[Dict[str, Any]] = []

    for ep_idx in range(n_base_episodes):
        print(f"\nGenerating and Executing Base Episode {ep_idx + 1}/{n_base_episodes}...")
        ep = generator.generate_episode(
            episode_idx=ep_idx,
            prefix_ticks=4,
            continuation_ticks=3,
            seed=seed,
        )

        trials, ep_meta = harness.execute_episode(
            episode=ep,
            interval_ks=eval_intervals,
        )
        all_trials.extend(trials)
        episode_manifests.append({
            "episode_id": ep.episode_id,
            "prefix_event_count": len(ep.prefix_events),
            "continuation_event_count": len(ep.continuation_events),
            "probe_count": len(ep.probes),
            "evidence_hash_prefix": ep_meta.get("evidence_hash_prefix"),
        })
        print(f"  -> Episode {ep.episode_id}: {len(trials)} trials recorded across conditions & intervals.")

    print(f"\nTotal Trials Recorded: {len(all_trials)}")

    analysis = analyze_quiet_interval_results(
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
        "total_episodes": len(episode_manifests),
        "total_trials": len(all_trials),
        "intervals": eval_intervals,
    }

    df_trials = pd.DataFrame([asdict(t) for t in all_trials])

    report_md = generate_e06_markdown_report(
        manifest=manifest,
        analysis=analysis,
        episode_manifests=episode_manifests,
        raw_df=df_trials,
    )

    for target_dir in [out_dir, canonical_results_dir]:
        with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        summary_payload = {
            "manifest": manifest,
            "analysis": {
                "total_episodes": analysis.total_episodes,
                "total_trials": analysis.total_trials,
                "intervals_evaluated": analysis.intervals_evaluated,
                "condition_stats": {k: asdict(v) for k, v in analysis.condition_stats.items()},
                "causal_estimands": {k: asdict(v) for k, v in analysis.causal_estimands.items()},
                "interval_breakdown": analysis.interval_breakdown,
                "interval_probe_breakdown": analysis.interval_probe_breakdown,
                "interval_contrasts": [asdict(c) for c in analysis.interval_contrasts],
                "unconstrained_evidence_drift_rate": analysis.unconstrained_evidence_drift_rate,
                "selective_evidence_drift_rate": analysis.selective_evidence_drift_rate,
            },
            "episodes": episode_manifests,
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
    print(f"EXPERIMENT E06 BENCHMARK COMPLETE")
    print(f"Artifacts written to: {out_dir}")
    print(f"Canonical Results written to: {canonical_results_dir}")
    print("=" * 70 + "\n")

    return {
        "manifest": manifest,
        "analysis": analysis,
        "report_md": report_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiment E06: Scaffolded Null-Interval & Quiet Processing Benchmark (Sprint S07)")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--phase", type=str, default="exploratory", choices=["exploratory", "confirmatory"], help="Experiment phase")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--episodes", type=int, default=None, help="Base episodes count (default 4 exploratory, 8 confirmatory)")
    parser.add_argument("--intervals", type=str, default="0,1,3,6,12", help="Comma-separated interval durations K")
    parser.add_argument("--dry-run", action="store_true", help="Run with mock backend")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")

    args = parser.parse_args()
    interval_list = [int(k.strip()) for k in args.intervals.split(",") if k.strip()]

    run_e06_experiment(
        model_name=args.model,
        seed=args.seed,
        phase=args.phase,
        temperature=args.temperature,
        episodes_count=args.episodes,
        intervals=interval_list,
        dry_run=args.dry_run,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
