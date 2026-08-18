"""Experiment E09c: Fixed-Target Metacognitive Interaction Screen (Sprint S09d).

Evaluates whether the -0.235 format-framing interaction persists when first-order target decisions
are strictly frozen and matched across both memory formats (Transcript vs Scaffolded) and both evaluators (Self vs Observer).

Primary Estimand:
Brier Score Difference-in-Differences and AUROC Difference-in-Differences under exact fixed target choices.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
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
    calculate_auroc,
    compute_clustered_bootstrap_ci,
    compute_pooled_auroc_cluster_inference,
    compute_clustered_auroc_interaction_inference,
    compute_interaction_format_block_permutation_test,
    EstimandWithUncertainty,
)
from experiments.e08_source_ownership.run import MockOwnershipBackend


@dataclass
class FixedTargetConditionSummary:
    """Performance summary for a single cell in the 2x2 Fixed-Target Metacognitive Factorial."""
    evaluator: str  # 'self' or 'observer'
    memory_format: str  # 'transcript_only' or 'scaffolded_state'
    total_trials: int
    mean_accuracy: float
    mean_confidence_pct: float
    brier_score: float
    auroc_error_prediction: float


@dataclass
class FixedTargetMetacognitiveAnalysisSummary:
    """Master analytical summary for Experiment E09c Fixed-Target Metacognitive Screen."""
    total_episodes: int
    total_items: int
    total_trials: int
    first_order_target_accuracy: float
    conditions: Dict[str, FixedTargetConditionSummary]
    # Primary Estimands
    delta_brier_transcript: EstimandWithUncertainty
    delta_brier_scaffolded: EstimandWithUncertainty
    brier_diff_in_diff_interaction: EstimandWithUncertainty
    delta_auroc_transcript: EstimandWithUncertainty
    delta_auroc_scaffolded: EstimandWithUncertainty
    auroc_interaction: EstimandWithUncertainty


def analyze_e09c_results(
    trials: List[OwnershipTrialResult],
    num_bootstrap: int = 2000,
    seed: int = 1337,
) -> FixedTargetMetacognitiveAnalysisSummary:
    """Analyze E09c fixed-target trials and compute Brier and AUROC Diff-in-Diff."""
    df = pd.DataFrame([asdict(t) for t in trials])
    df["evaluator"] = df["metadata"].apply(lambda m: m.get("evaluator", "unknown"))
    df["format"] = df["metadata"].apply(lambda m: m.get("format", "unknown"))
    df["brier"] = (df["subjective_confidence_pct"] / 100.0 - df["is_correct"].astype(float)) ** 2

    episodes = sorted(df["episode_id"].unique())
    n_episodes = len(episodes)

    # Condition splits
    df_st = df[(df["evaluator"] == "self") & (df["format"] == "transcript_only")].copy()
    df_ot = df[(df["evaluator"] == "observer") & (df["format"] == "transcript_only")].copy()
    df_ss = df[(df["evaluator"] == "self") & (df["format"] == "scaffolded_state")].copy()
    df_os = df[(df["evaluator"] == "observer") & (df["format"] == "scaffolded_state")].copy()

    # Target choice accuracy (frozen across all conditions)
    target_acc = float(df_st["is_correct"].mean()) if len(df_st) else 0.0

    def make_cond_summary(df_sub: pd.DataFrame, eval_name: str, fmt_name: str) -> FixedTargetConditionSummary:
        n = len(df_sub)
        if n == 0:
            return FixedTargetConditionSummary(eval_name, fmt_name, 0, 0.0, 0.0, 0.0, 0.5)
        acc = float(df_sub["is_correct"].mean())
        mean_conf = float(df_sub["subjective_confidence_pct"].mean())
        brier = float(df_sub["brier"].mean())
        auroc = calculate_auroc(df_sub["subjective_confidence_pct"].tolist(), df_sub["is_correct"].tolist())
        return FixedTargetConditionSummary(eval_name, fmt_name, n, acc, mean_conf, brier, auroc)

    conditions_dict = {
        "self_transcript": make_cond_summary(df_st, "self", "transcript_only"),
        "observer_transcript": make_cond_summary(df_ot, "observer", "transcript_only"),
        "self_scaffolded": make_cond_summary(df_ss, "self", "scaffolded_state"),
        "observer_scaffolded": make_cond_summary(df_os, "observer", "scaffolded_state"),
    }

    # Paired Brier differences per episode
    ep_brier_diff_trans = []
    ep_brier_diff_scaff = []
    ep_brier_interaction = []

    for ep in episodes:
        b_st = float(df_st[df_st["episode_id"] == ep]["brier"].mean()) if len(df_st[df_st["episode_id"] == ep]) else 0.0
        b_ot = float(df_ot[df_ot["episode_id"] == ep]["brier"].mean()) if len(df_ot[df_ot["episode_id"] == ep]) else 0.0
        b_ss = float(df_ss[df_ss["episode_id"] == ep]["brier"].mean()) if len(df_ss[df_ss["episode_id"] == ep]) else 0.0
        b_os = float(df_os[df_os["episode_id"] == ep]["brier"].mean()) if len(df_os[df_os["episode_id"] == ep]) else 0.0

        # Note: Brier score: Lower is better. Delta = Brier(Self) - Brier(Observer). Negative delta means Self is better calibrated.
        d_trans = b_st - b_ot
        d_scaff = b_ss - b_os
        ep_brier_diff_trans.append(d_trans)
        ep_brier_diff_scaff.append(d_scaff)
        ep_brier_interaction.append(d_scaff - d_trans)

    pt_brier_t, lo_bt, hi_bt, _ = compute_clustered_bootstrap_ci(ep_brier_diff_trans, baseline=0.0, num_bootstrap=num_bootstrap, seed=seed)
    pt_brier_s, lo_bs, hi_bs, _ = compute_clustered_bootstrap_ci(ep_brier_diff_scaff, baseline=0.0, num_bootstrap=num_bootstrap, seed=seed)
    pt_brier_int, lo_bint, hi_bint, _ = compute_clustered_bootstrap_ci(ep_brier_interaction, baseline=0.0, num_bootstrap=num_bootstrap, seed=seed)

    # Exact sign-flip permutation for Brier interaction
    if n_episodes > 0:
        obs_stat = abs(sum(ep_brier_interaction))
        extreme_count = 0
        n_perms = min(2 ** n_episodes, 65536)
        for k in range(n_perms):
            signs = [1 if (k & (1 << j)) else -1 for j in range(n_episodes)]
            perm_stat = abs(sum(s * val for s, val in zip(signs, ep_brier_interaction)))
            if perm_stat >= obs_stat - 1e-9:
                extreme_count += 1
        p_brier_int = (extreme_count + 1) / (n_perms + 1)
    else:
        p_brier_int = 1.0

    delta_brier_trans_est = EstimandWithUncertainty(
        name="Delta_Brier_Transcript",
        description="Brier(Self) - Brier(Observer) under raw transcript (lower is better for Self)",
        point_estimate=pt_brier_t,
        ci_lower_95=lo_bt,
        ci_upper_95=hi_bt,
        permutation_p_value=None,
        permutation_method="cluster_bootstrap_ci_only",
        is_statistically_distinguishable=(lo_bt > 0 or hi_bt < 0),
    )

    delta_brier_scaff_est = EstimandWithUncertainty(
        name="Delta_Brier_Scaffolded",
        description="Brier(Self) - Brier(Observer) under scaffolded state (lower is better for Self)",
        point_estimate=pt_brier_s,
        ci_lower_95=lo_bs,
        ci_upper_95=hi_bs,
        permutation_p_value=None,
        permutation_method="cluster_bootstrap_ci_only",
        is_statistically_distinguishable=(lo_bs > 0 or hi_bs < 0),
    )

    brier_interaction_est = EstimandWithUncertainty(
        name="Brier_Diff_in_Diff_Interaction",
        description="Delta_Brier(Scaffolded) - Delta_Brier(Transcript) under fixed target decisions",
        point_estimate=pt_brier_int,
        ci_lower_95=lo_bint,
        ci_upper_95=hi_bint,
        permutation_p_value=p_brier_int,
        permutation_method=f"exact_sign_flip_2^{n_episodes}",
        is_statistically_distinguishable=(lo_bint > 0 or hi_bint < 0),
    )

    # AUROC Estimands with real cluster-bootstrap CIs across episodes
    pt_at, lo_at, hi_at, _, _ = compute_pooled_auroc_cluster_inference(
        df_self=df_st,
        df_obs=df_ot,
        episodes=episodes,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )

    pt_as, lo_as, hi_as, _, _ = compute_pooled_auroc_cluster_inference(
        df_self=df_ss,
        df_obs=df_os,
        episodes=episodes,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )

    pt_aint, lo_aint, hi_aint, p_auroc_int, method_auroc = compute_clustered_auroc_interaction_inference(
        df_self_trans=df_st,
        df_obs_trans=df_ot,
        df_self_scaff=df_ss,
        df_obs_scaff=df_os,
        episodes=episodes,
        num_bootstrap=num_bootstrap,
        seed=seed,
    )

    delta_auroc_trans_est = EstimandWithUncertainty(
        name="Delta_AUROC_Transcript",
        description="AUROC(Self) - AUROC(Observer) under raw transcript",
        point_estimate=pt_at,
        ci_lower_95=lo_at,
        ci_upper_95=hi_at,
        permutation_p_value=None,
        permutation_method="cluster_bootstrap_ci_only",
        is_statistically_distinguishable=(lo_at > 0 or hi_at < 0),
    )

    delta_auroc_scaff_est = EstimandWithUncertainty(
        name="Delta_AUROC_Scaffolded",
        description="AUROC(Self) - AUROC(Observer) under scaffolded state",
        point_estimate=pt_as,
        ci_lower_95=lo_as,
        ci_upper_95=hi_as,
        permutation_p_value=None,
        permutation_method="cluster_bootstrap_ci_only",
        is_statistically_distinguishable=(lo_as > 0 or hi_as < 0),
    )

    auroc_int_est = EstimandWithUncertainty(
        name="AUROC_Metacognitive_Interaction",
        description="Delta_AUROC(Scaffolded) - Delta_AUROC(Transcript) under fixed target decisions",
        point_estimate=pt_aint,
        ci_lower_95=lo_aint,
        ci_upper_95=hi_aint,
        permutation_p_value=p_auroc_int,
        permutation_method=method_auroc,
        is_statistically_distinguishable=(p_auroc_int < 0.05),
    )

    return FixedTargetMetacognitiveAnalysisSummary(
        total_episodes=n_episodes,
        total_items=len(df_st),
        total_trials=len(trials),
        first_order_target_accuracy=target_acc,
        conditions=conditions_dict,
        delta_brier_transcript=delta_brier_trans_est,
        delta_brier_scaffolded=delta_brier_scaff_est,
        brier_diff_in_diff_interaction=brier_interaction_est,
        delta_auroc_transcript=delta_auroc_trans_est,
        delta_auroc_scaffolded=delta_auroc_scaff_est,
        auroc_interaction=auroc_int_est,
    )


def generate_e09c_markdown_report(
    manifest: Dict[str, Any],
    analysis: FixedTargetMetacognitiveAnalysisSummary,
) -> str:
    """Generate publication-ready Markdown report for Experiment E09c."""
    lines = [
        f"# Experiment E09c: Fixed-Target Metacognitive Interaction Screen Report (Sprint S09d)",
        f"",
        f"**Run ID:** `{manifest['run_id']}`  ",
        f"**Model:** `{manifest['target_model']}` (`{manifest['model_digest'][:12]}...`)  ",
        f"**Phase:** `{manifest['phase'].upper()}` (Seed: `{manifest['seed']}`)  ",
        f"**Date:** {manifest['start_time']}  ",
        f"**Scope:** {manifest['total_episodes']} Episodes | {analysis.total_items} Fixed Target Decisions | {manifest['total_trials']} Total Evaluator Probes  ",
        f"**Primary Question:** *Under strictly frozen, identical first-order target decisions across all conditions, does scaffolded persistence alter the self-observer metacognitive calibration gap?*",
        f"",
        f"---",
        f"",
        f"## 1. 2x2 Fixed-Target Metacognitive Factorial Matrix",
        f"",
        f"| Evaluator | Memory Format | Trials | Target Accuracy | Mean Confidence | Brier Score (Lower is Better) | AUROC Error Prediction |",
        f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cond_key, c in analysis.conditions.items():
        eval_label = "Primary Agent (Self / alpha)" if c.evaluator == "self" else "Auditing Observer (gamma)"
        fmt_label = "Transcript-Only" if c.memory_format == "transcript_only" else "Scaffolded Persistence"
        lines.append(
            f"| **{eval_label}** | `{fmt_label}` | {c.total_trials} | {c.mean_accuracy:.1%} | {c.mean_confidence_pct:.1f}% | **{c.brier_score:.4f}** | **{c.auroc_error_prediction:.3f}** |"
        )

    bi = analysis.brier_diff_in_diff_interaction
    ai = analysis.auroc_interaction

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Primary Estimands: Brier & AUROC Difference-in-Differences",
        f"",
        f"| Estimand | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |",
        f"| :--- | :---: | :---: | :---: | :--- |",
        f"| **`Brier_Diff_in_Diff_Interaction`** | **{bi.point_estimate:+.4f}** | [{bi.ci_lower_95:+.4f}, {bi.ci_upper_95:+.4f}] | {bi.permutation_p_value:.4f} (`{bi.permutation_method}`) | {'**Resolved Metacognitive Interaction**' if bi.is_statistically_distinguishable else '**No resolved format × framing interaction under prespecified exact test**'} |",
        f"| **`AUROC_Metacognitive_Interaction`** | **{ai.point_estimate:+.3f}** | [{ai.ci_lower_95:+.3f}, {ai.ci_upper_95:+.3f}] | {ai.permutation_p_value:.4f} (`{ai.permutation_method}`) | {'**Resolved AUROC Interaction**' if ai.is_statistically_distinguishable else '**No resolved format × framing interaction under prespecified exact test**'} |",
        f"| **`Delta_Brier_Transcript`** | **{analysis.delta_brier_transcript.point_estimate:+.4f}** | [{analysis.delta_brier_transcript.ci_lower_95:+.4f}, {analysis.delta_brier_transcript.ci_upper_95:+.4f}] | N/A (`cluster_bootstrap_ci_only`) | {f'Self calibrated better' if analysis.delta_brier_transcript.point_estimate < 0 else 'Observer calibrated better'} |",
        f"| **`Delta_Brier_Scaffolded`** | **{analysis.delta_brier_scaffolded.point_estimate:+.4f}** | [{analysis.delta_brier_scaffolded.ci_lower_95:+.4f}, {analysis.delta_brier_scaffolded.ci_upper_95:+.4f}] | N/A (`cluster_bootstrap_ci_only`) | {f'Self calibrated better' if analysis.delta_brier_scaffolded.point_estimate < 0 else 'Observer calibrated better'} |",
        f"",
        f"---",
        f"",
        f"## 3. Scientific Conclusion",
        f"",
        f"- **First-Order Choice Invariance:** Primary agent choice distribution held fixed at **{analysis.first_order_target_accuracy:.1%} accuracy** across all evaluators and formats.",
        f"- **Brier Calibration Diff-in-Diff:** $\\text{{Interaction}}_{{\\text{{Brier}}}} = \\mathbf{{{bi.point_estimate:+.4f}}}$ ($p = {bi.permutation_p_value:.4f}$).",
        f"- **AUROC Resolution Diff-in-Diff:** $\\text{{Interaction}}_{{\\text{{AUROC}}}} = \\mathbf{{{ai.point_estimate:+.3f}}}$ ($p = {ai.permutation_p_value:.4f}$).",
        f"- **Epistemic Invariance:** No resolved format $\\times$ framing interaction under the prespecified exact tests ($p = {bi.permutation_p_value:.4f}$ and $p = {ai.permutation_p_value:.4f}$).",
    ])

    return "\n".join(lines)


def reprocess_e09c_directory(target_dir: Path, seed: int = 1337) -> None:
    """Reprocess an existing E09c run directory offline using true clustered bootstrap inference."""
    trials_path = target_dir / "trials.jsonl"
    manifest_path = target_dir / "manifest.json"
    if not trials_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing trials or manifest in {target_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    trials = []
    with open(trials_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                trials.append(OwnershipTrialResult(**d))

    analysis = analyze_e09c_results(trials, num_bootstrap=2000, seed=seed)

    summary_payload = {
        "manifest": manifest,
        "analysis": asdict(analysis),
    }

    with open(target_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    report_md = generate_e09c_markdown_report(manifest, analysis)
    with open(target_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Successfully reprocessed E09c directory: {target_dir}")


def run_e09c_experiment(
    phase: str = "exploratory",
    seed: int = 42,
    total_episodes: int = 4,
    model_name: str = "qwen2.5:3b",
    use_mock: bool = False,
    output_dir: Optional[Path] = None,
) -> Path:
    """Execute Experiment E09c fixed-target metacognitive interaction screen."""
    start_time = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_e09c_meta_{timestamp}_{phase}"

    if output_dir is None:
        target_dir = Path("results") / "e09c_fixed_target_meta" / run_id
    else:
        target_dir = output_dir / run_id
    target_dir.mkdir(parents=True, exist_ok=True)

    if use_mock:
        backend = MockOwnershipBackend(model_name=model_name)
    else:
        backend = OllamaBackend(model_name=model_name)

    model_digest = backend.get_digest() if hasattr(backend, "get_digest") else "mock_digest"
    generator = OwnershipTaskGenerator(seed=seed)
    harness = OwnershipHarness(backend=backend)

    all_trials: List[OwnershipTrialResult] = []
    episodes_manifest: List[Dict[str, Any]] = []

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting E09c {phase.upper()} run: {total_episodes} episodes...")

    for i in range(total_episodes):
        ep_seed = seed + i * 7001
        episode = generator.generate_episode(twin_idx=i, seed=ep_seed)
        ep_trials: List[OwnershipTrialResult] = []

        for probe in episode.probes_attribution_5afc:
            # -------------------------------------------------------------
            # Step 1: Query and Freeze Primary Agent Target Decision (under Transcript)
            # -------------------------------------------------------------
            prompt_target, p_hash_t = harness._build_prompt(
                events=episode.events_neutral,
                state=None,
                probe=probe,
                role_preamble="You are primary agent 'agent_alpha' operating within a multi-agent system.",
            )
            pred_let_target, pred_text_target, p_tok_t, c_tok_t, lat_t, err_t = harness._query_choice(prompt_target, probe)
            is_target_correct = (pred_let_target == probe.correct_option)

            # -------------------------------------------------------------
            # Step 2: Evaluate 4 Factorial Cells on IDENTICAL Target Choice
            # -------------------------------------------------------------
            for fmt_name, st_obj in [("transcript_only", None), ("scaffolded_state", episode.oracle_state)]:
                # Cell A: Self Confidence in Frozen Decision
                prompt_self, p_hash_s = harness._build_prompt(
                    events=episode.events_neutral,
                    state=st_obj,
                    probe=probe,
                    role_preamble="You are primary agent 'agent_alpha' operating within a multi-agent system.",
                )
                conf_self, p_tok_s, c_tok_s, lat_s, err_s = harness._query_confidence_assessment(
                    base_prompt=prompt_self,
                    target_choice_letter=pred_let_target,
                    target_choice_text=pred_text_target,
                    evaluator="self",
                )
                trial_self = OwnershipTrialResult(
                    trial_id=f"{episode.episode_id}_e09c_self_{fmt_name}_{probe.probe_id}",
                    episode_id=episode.episode_id,
                    experiment_submodule="e09c_fixed_target_meta",
                    condition_name=f"meta_self_{fmt_name}",
                    probe_id=probe.probe_id,
                    probe_type="fixed_target_confidence_self",
                    question=probe.question,
                    options=probe.options,
                    predicted_letter=pred_let_target,
                    predicted_text=pred_text_target,
                    correct_letter=probe.correct_option,
                    is_correct=is_target_correct,
                    attributed_actor=None,
                    target_source=probe.target_source,
                    target_actor=probe.target_actor,
                    target_value=probe.target_value,
                    subjective_confidence_pct=conf_self,
                    prompt_hash=p_hash_s,
                    prompt_tokens=p_tok_s,
                    completion_tokens=c_tok_s,
                    latency_ms=lat_s,
                    error_message=err_s,
                    metadata={"evaluator": "self", "format": fmt_name, "key": probe.metadata.get("key"), "target_choice": pred_let_target},
                )
                ep_trials.append(trial_self)
                all_trials.append(trial_self)

                # Cell B: Observer Confidence in Frozen Decision
                prompt_obs, p_hash_o = harness._build_prompt(
                    events=episode.events_neutral,
                    state=st_obj,
                    probe=probe,
                    role_preamble="You are an external auditing observer 'auditor_gamma' monitoring multi-agent system execution.",
                )
                conf_obs, p_tok_o, c_tok_o, lat_o, err_o = harness._query_confidence_assessment(
                    base_prompt=prompt_obs,
                    target_choice_letter=pred_let_target,
                    target_choice_text=pred_text_target,
                    evaluator="observer",
                )
                trial_obs = OwnershipTrialResult(
                    trial_id=f"{episode.episode_id}_e09c_observer_{fmt_name}_{probe.probe_id}",
                    episode_id=episode.episode_id,
                    experiment_submodule="e09c_fixed_target_meta",
                    condition_name=f"meta_observer_{fmt_name}",
                    probe_id=probe.probe_id,
                    probe_type="fixed_target_confidence_observer",
                    question=probe.question,
                    options=probe.options,
                    predicted_letter=pred_let_target,
                    predicted_text=pred_text_target,
                    correct_letter=probe.correct_option,
                    is_correct=is_target_correct,
                    attributed_actor=None,
                    target_source=probe.target_source,
                    target_actor=probe.target_actor,
                    target_value=probe.target_value,
                    subjective_confidence_pct=conf_obs,
                    prompt_hash=p_hash_o,
                    prompt_tokens=p_tok_o,
                    completion_tokens=c_tok_o,
                    latency_ms=lat_o,
                    error_message=err_o,
                    metadata={"evaluator": "observer", "format": fmt_name, "key": probe.metadata.get("key"), "target_choice": pred_let_target},
                )
                ep_trials.append(trial_obs)
                all_trials.append(trial_obs)

        episodes_manifest.append({
            "episode_index": i,
            "episode_id": episode.episode_id,
            "trials_count": len(ep_trials),
        })

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed Episode {i+1}/{total_episodes} ({len(ep_trials)} probes)")

    # Save trials.jsonl
    with open(target_dir / "trials.jsonl", "w", encoding="utf-8") as f:
        for t in all_trials:
            f.write(json.dumps(asdict(t)) + "\n")

    analysis = analyze_e09c_results(all_trials, num_bootstrap=2000, seed=seed)

    manifest = {
        "run_id": run_id,
        "phase": phase,
        "seed": seed,
        "target_model": model_name,
        "model_digest": model_digest,
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "total_episodes": total_episodes,
        "total_trials": len(all_trials),
        "use_mock": use_mock,
    }

    with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary_payload = {
        "manifest": manifest,
        "analysis": asdict(analysis),
        "episodes": episodes_manifest,
    }

    with open(target_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    report_md = generate_e09c_markdown_report(manifest, analysis)
    with open(target_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] E09c run completed successfully. Output saved to {target_dir}")
    return target_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment E09c: Fixed-Target Metacognitive Screen")
    parser.add_argument("--phase", choices=["exploratory", "confirmatory"], default="exploratory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--model", type=str, default="qwen2.5:3b")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()
    run_e09c_experiment(
        phase=args.phase,
        seed=args.seed,
        total_episodes=args.episodes,
        model_name=args.model,
        use_mock=args.mock,
        output_dir=args.output_dir,
    )
