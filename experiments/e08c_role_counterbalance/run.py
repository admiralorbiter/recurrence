"""Experiment E08c: Primary-Role Counterbalance & Instrument Ceiling Control (Sprint S09c).

Disentangles egocentric self-attribution bias from lexical actor token prior ('agent_alpha' vs 'agent_beta')
by evaluating matched twin episodes under:
1. Role Assignment A: agent_alpha = Self / Primary Agent, agent_beta = Peer Agent
2. Role Assignment B: agent_beta = Self / Primary Agent, agent_alpha = Peer Agent

Also measures isolated 5AFC prompt lookup accuracy (instrument ceiling control).
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
    get_actor_map,
    get_actor_display_names,
)
from recurrence.loop.ownership_experiment import (
    OwnershipHarness,
    OwnershipTrialResult,
)
from recurrence.analysis.ownership_metrics import (
    compute_clustered_bootstrap_ci,
    compute_within_episode_source_permutation_test,
    EstimandWithUncertainty,
)
from experiments.e08_source_ownership.run import MockOwnershipBackend


@dataclass
class RoleCounterbalanceAnalysisSummary:
    """Analytical summary for Experiment E08c Primary-Role Counterbalance."""
    total_episode_pairs: int
    total_trials: int
    # Role A (Alpha-Primary) metrics
    role_a_overall_5afc: float
    role_a_self_accuracy: float
    role_a_alpha_attribution_rate: float
    role_a_beta_attribution_rate: float
    # Role B (Beta-Primary) metrics
    role_b_overall_5afc: float
    role_b_self_accuracy: float
    role_b_alpha_attribution_rate: float
    role_b_beta_attribution_rate: float
    # Primary Estimands
    delta_role_reversal_shift: EstimandWithUncertainty
    alpha_lexical_token_bias: EstimandWithUncertainty
    # Isolated positive control ceiling
    isolated_ceiling_overall_accuracy: EstimandWithUncertainty
    isolated_ceiling_per_source: Dict[str, float]
    # Confusion matrices
    confusion_matrix_role_a: Dict[str, Dict[str, float]]
    confusion_matrix_role_b: Dict[str, Dict[str, float]]


def analyze_e08c_results(
    trials: List[OwnershipTrialResult],
    num_bootstrap: int = 2000,
    seed: int = 1337,
) -> RoleCounterbalanceAnalysisSummary:
    """Analyze E08c counterbalance trials and estimate role-reversal shift vs lexical token bias."""
    df = pd.DataFrame([asdict(t) for t in trials])
    df["role_mapping"] = df["metadata"].apply(lambda m: m.get("role_mapping", "alpha_self_beta_peer"))

    df_neutral = df[df["condition_name"] == "neutral_5afc_attribution"].copy()
    df_ceiling = df[df["condition_name"] == "isolated_ceiling_5afc"].copy()

    df_role_a = df_neutral[df_role_mapping_a := (df_neutral["role_mapping"] == "alpha_self_beta_peer")].copy()
    df_role_b = df_neutral[df_role_mapping_b := (df_neutral["role_mapping"] == "beta_self_alpha_peer")].copy()

    episodes_a = sorted(df_role_a["episode_id"].unique())
    episodes_b = sorted(df_role_b["episode_id"].unique())
    base_episodes = [ep.replace("_roleA", "").replace("_roleB", "") for ep in episodes_a]

    # Attribution rates
    a_tot = len(df_role_a)
    b_tot = len(df_role_b)

    role_a_acc = float(df_role_a["is_correct"].mean()) if a_tot else 0.0
    role_b_acc = float(df_role_b["is_correct"].mean()) if b_tot else 0.0

    df_a_self = df_role_a[df_role_a["target_source"] == "self"]
    df_b_self = df_role_b[df_role_b["target_source"] == "self"]

    role_a_self_acc = float(df_a_self["is_correct"].mean()) if len(df_a_self) else 0.0
    role_b_self_acc = float(df_b_self["is_correct"].mean()) if len(df_b_self) else 0.0

    role_a_alpha_rate = float((df_role_a["attributed_actor"] == "agent_alpha").mean()) if a_tot else 0.0
    role_a_beta_rate = float((df_role_a["attributed_actor"] == "agent_beta").mean()) if a_tot else 0.0

    role_b_alpha_rate = float((df_role_b["attributed_actor"] == "agent_alpha").mean()) if b_tot else 0.0
    role_b_beta_rate = float((df_role_b["attributed_actor"] == "agent_beta").mean()) if b_tot else 0.0

    # Paired role reversal difference per episode pair
    # For each twin episode:
    # Role A: P(pick Alpha | Alpha is Self) vs Role B: P(pick Alpha | Alpha is Peer)
    # Role B: P(pick Beta | Beta is Self) vs Role A: P(pick Beta | Beta is Peer)
    shift_vals = []
    alpha_bias_vals = []

    for base_ep in base_episodes:
        ep_a = f"{base_ep}_roleA"
        ep_b = f"{base_ep}_roleB"

        sub_a = df_role_a[df_role_a["episode_id"] == ep_a]
        sub_b = df_role_b[df_role_b["episode_id"] == ep_b]

        if len(sub_a) == 0 or len(sub_b) == 0:
            continue

        # Rate of selecting designated self actor
        # In Role A: self actor is agent_alpha
        # In Role B: self actor is agent_beta
        p_alpha_when_self = (sub_a["attributed_actor"] == "agent_alpha").mean()
        p_alpha_when_peer = (sub_b["attributed_actor"] == "agent_alpha").mean()

        p_beta_when_self = (sub_b["attributed_actor"] == "agent_beta").mean()
        p_beta_when_peer = (sub_a["attributed_actor"] == "agent_beta").mean()

        # Shift: how much more is an actor chosen when designated Self vs when designated Peer
        ep_shift = 0.5 * ((p_alpha_when_self - p_alpha_when_peer) + (p_beta_when_self - p_beta_when_peer))
        shift_vals.append(ep_shift)

        # Lexical bias: preference for agent_alpha over agent_beta across both roles
        p_alpha_all = 0.5 * ((sub_a["attributed_actor"] == "agent_alpha").mean() + (sub_b["attributed_actor"] == "agent_alpha").mean())
        p_beta_all = 0.5 * ((sub_a["attributed_actor"] == "agent_beta").mean() + (sub_b["attributed_actor"] == "agent_beta").mean())
        alpha_bias_vals.append(p_alpha_all - p_beta_all)

    pt_shift = float(np.mean(shift_vals)) if shift_vals else 0.0
    pt_bias = float(np.mean(alpha_bias_vals)) if alpha_bias_vals else 0.0

    rng = np.random.default_rng(seed)
    n_pairs = len(shift_vals)
    boot_shifts = []
    boot_biases = []

    if n_pairs > 1:
        for _ in range(num_bootstrap):
            idx = rng.choice(n_pairs, size=n_pairs, replace=True)
            boot_shifts.append(np.mean([shift_vals[i] for i in idx]))
            boot_biases.append(np.mean([alpha_bias_vals[i] for i in idx]))

        ci_shift_lo, ci_shift_hi = float(np.percentile(boot_shifts, 2.5)), float(np.percentile(boot_shifts, 97.5))
        ci_bias_lo, ci_bias_hi = float(np.percentile(boot_biases, 2.5)), float(np.percentile(boot_biases, 97.5))
    else:
        ci_shift_lo, ci_shift_hi = pt_shift, pt_shift
        ci_bias_lo, ci_bias_hi = pt_bias, pt_bias

    # Sign-flip permutation test for role reversal shift
    if n_pairs > 0:
        obs_stat = abs(sum(shift_vals))
        extreme_count = 0
        n_perms = 2 ** n_pairs
        for k in range(n_perms):
            signs = [1 if (k & (1 << j)) else -1 for j in range(n_pairs)]
            perm_stat = abs(sum(s * val for s, val in zip(signs, shift_vals)))
            if perm_stat >= obs_stat - 1e-9:
                extreme_count += 1
        p_shift = (extreme_count + 1) / (n_perms + 1)
    else:
        p_shift = 1.0

    delta_role_est = EstimandWithUncertainty(
        name="Delta_Role_Reversal_Shift",
        description="P(choose actor | designated Self) - P(choose actor | designated Peer) averaged across Alpha and Beta",
        point_estimate=pt_shift,
        ci_lower_95=ci_shift_lo,
        ci_upper_95=ci_shift_hi,
        permutation_p_value=p_shift,
        permutation_method=f"exact_sign_flip_2^{n_pairs}",
        is_statistically_distinguishable=(ci_shift_lo > 0 or ci_shift_hi < 0),
    )

    alpha_bias_est = EstimandWithUncertainty(
        name="Alpha_Lexical_Token_Bias",
        description="P(choose agent_alpha) - P(choose agent_beta) across both role configurations",
        point_estimate=pt_bias,
        ci_lower_95=ci_bias_lo,
        ci_upper_95=ci_bias_hi,
        permutation_p_value=None,
        permutation_method="cluster_bootstrap_ci_only",
        is_statistically_distinguishable=(ci_bias_lo > 0 or ci_bias_hi < 0),
    )

    # Isolated positive control ceiling
    ceil_acc = float(df_ceiling["is_correct"].mean()) if len(df_ceiling) else 0.0
    ep_ceil_vals = [float(df_ceiling[df_ceiling["episode_id"] == ep]["is_correct"].mean()) for ep in df_ceiling["episode_id"].unique()] if len(df_ceiling) else []
    _, ceil_lo, ceil_hi, _ = compute_clustered_bootstrap_ci(ep_ceil_vals, baseline=0.20, num_bootstrap=num_bootstrap, seed=seed)
    
    ceil_per_source = {}
    for src in ["self", "environment", "experimenter", "peer_agent", "observer"]:
        sub_src = df_ceiling[df_ceiling["target_source"] == src]
        ceil_per_source[src] = float(sub_src["is_correct"].mean()) if len(sub_src) else 0.0

    isolated_ceiling_est = EstimandWithUncertainty(
        name="Isolated_Positive_Control_Ceiling",
        description="5AFC source identification accuracy under direct isolated context without memory load",
        point_estimate=ceil_acc,
        ci_lower_95=ceil_lo,
        ci_upper_95=ceil_hi,
        permutation_p_value=None,
        permutation_method="cluster_bootstrap_ci_only",
        is_statistically_distinguishable=(ceil_lo > 0.20),
    )

    # Confusion matrices
    def make_confusion_matrix(df_sub: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        sources = ["self", "environment", "experimenter", "peer_agent", "observer"]
        actors = ["agent_alpha", "agent_beta", "telemetry_sensor", "human_controller", "auditor_gamma"]
        cm = {}
        for src in sources:
            sub = df_sub[df_sub["target_source"] == src]
            tot = len(sub)
            cm[src] = {}
            for act in actors:
                cm[src][act] = float((sub["attributed_actor"] == act).sum() / tot) if tot > 0 else 0.0
        return cm

    cm_a = make_confusion_matrix(df_role_a)
    cm_b = make_confusion_matrix(df_role_b)

    return RoleCounterbalanceAnalysisSummary(
        total_episode_pairs=n_pairs,
        total_trials=len(trials),
        role_a_overall_5afc=role_a_acc,
        role_a_self_accuracy=role_a_self_acc,
        role_a_alpha_attribution_rate=role_a_alpha_rate,
        role_a_beta_attribution_rate=role_a_beta_rate,
        role_b_overall_5afc=role_b_acc,
        role_b_self_accuracy=role_b_self_acc,
        role_b_alpha_attribution_rate=role_b_alpha_rate,
        role_b_beta_attribution_rate=role_b_beta_rate,
        delta_role_reversal_shift=delta_role_est,
        alpha_lexical_token_bias=alpha_bias_est,
        isolated_ceiling_overall_accuracy=isolated_ceiling_est,
        isolated_ceiling_per_source=ceil_per_source,
        confusion_matrix_role_a=cm_a,
        confusion_matrix_role_b=cm_b,
    )


def generate_e08c_markdown_report(
    manifest: Dict[str, Any],
    analysis: RoleCounterbalanceAnalysisSummary,
) -> str:
    """Generate publication-ready Markdown report for Experiment E08c."""
    lines = [
        f"# Experiment E08c: Primary-Role Counterbalance & Instrument Ceiling Control Report (Sprint S09c)",
        f"",
        f"**Run ID:** `{manifest['run_id']}`  ",
        f"**Model:** `{manifest['target_model']}` (`{manifest['model_digest'][:12]}...`)  ",
        f"**Phase:** `{manifest['phase'].upper()}` (Seed: `{manifest['seed']}`)  ",
        f"**Date:** {manifest['start_time']}  ",
        f"**Scope:** {manifest['total_episodes']} Matched Episode Pairs ({manifest['total_episodes']*2} Episodes) | {manifest['total_trials']} Total Counterbalance Trials  ",
        f"**Primary Question:** *Does the primary-agent attribution attractor follow the prompt-designated Self role or the lexical token 'agent_alpha'? What is the direct prompt instrument ceiling?*",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary & Core Disentanglement Estimands",
        f"",
        f"| Estimand | Point Estimate | 95% Clustered CI | Permutation $p$ (Method) | Scientific Inference |",
        f"| :--- | :---: | :---: | :---: | :--- |",
        f"| **`Delta_Role_Reversal_Shift`** | **{analysis.delta_role_reversal_shift.point_estimate:+.1%}** | [{analysis.delta_role_reversal_shift.ci_lower_95:+.1%}, {analysis.delta_role_reversal_shift.ci_upper_95:+.1%}] | {analysis.delta_role_reversal_shift.permutation_p_value:.4f} (`{analysis.delta_role_reversal_shift.permutation_method}`) | {'**Attractor follows designated Self role**' if analysis.delta_role_reversal_shift.point_estimate > 0 and analysis.delta_role_reversal_shift.is_statistically_distinguishable else '**Attractor is lexical/token-bound**'} |",
        f"| **`Alpha_Lexical_Token_Bias`** | **{analysis.alpha_lexical_token_bias.point_estimate:+.1%}** | [{analysis.alpha_lexical_token_bias.ci_lower_95:+.1%}, {analysis.alpha_lexical_token_bias.ci_upper_95:+.1%}] | N/A (`cluster_bootstrap_ci_only`) | {f'Preference for agent_alpha over agent_beta' if analysis.alpha_lexical_token_bias.point_estimate > 0 else 'Balanced / Beta preference'} |",
        f"| **`Isolated_Positive_Control_Ceiling`** | **{analysis.isolated_ceiling_overall_accuracy.point_estimate:.1%}** | [{analysis.isolated_ceiling_overall_accuracy.ci_lower_95:.1%}, {analysis.isolated_ceiling_overall_accuracy.ci_upper_95:.1%}] | N/A (`cluster_bootstrap_ci_only`) | **Prompt Instrument Ceiling (No Memory Load)** |",
        f"",
        f"---",
        f"",
        f"## 2. Role Configuration Breakdown: Role A (Alpha-Primary) vs Role B (Beta-Primary)",
        f"",
        f"| Metric / Attribution Rate | Role A (Alpha = Self, Beta = Peer) | Role B (Beta = Self, Alpha = Peer) | Contrast / Delta |",
        f"| :--- | :---: | :---: | :---: |",
        f"| **Overall 5AFC Accuracy** | **{analysis.role_a_overall_5afc:.1%}** | **{analysis.role_b_overall_5afc:.1%}** | {analysis.role_a_overall_5afc - analysis.role_b_overall_5afc:+.1%} |",
        f"| **True-Self Accuracy** | **{analysis.role_a_self_accuracy:.1%}** | **{analysis.role_b_self_accuracy:.1%}** | {analysis.role_a_self_accuracy - analysis.role_b_self_accuracy:+.1%} |",
        f"| **Attributed to `agent_alpha`** | **{analysis.role_a_alpha_attribution_rate:.1%}** | **{analysis.role_b_alpha_attribution_rate:.1%}** | {analysis.role_a_alpha_attribution_rate - analysis.role_b_alpha_attribution_rate:+.1%} |",
        f"| **Attributed to `agent_beta`** | **{analysis.role_a_beta_attribution_rate:.1%}** | **{analysis.role_b_beta_attribution_rate:.1%}** | {analysis.role_a_beta_attribution_rate - analysis.role_b_beta_attribution_rate:+.1%} |",
        f"",
        f"---",
        f"",
        f"## 3. Isolated Positive Control Ceiling Breakdown (Per Source)",
        f"",
        f"| Epistemic Source | Direct Isolated 5AFC Accuracy | Theoretical Baseline |",
        f"| :--- | :---: | :---: |",
    ]

    for src, acc in analysis.isolated_ceiling_per_source.items():
        lines.append(f"| **`{src}`** | **{acc:.1%}** | 20.0% (5AFC Chance) |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Empirical Confusion Matrices",
        f"",
        f"### Role A: Alpha-Primary (Alpha = Self, Beta = Peer)",
        f"| True Source | Attributed Alpha (Self) | Attributed Beta (Peer) | Attributed Sensor | Attributed Controller | Attributed Auditor |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for src, m in analysis.confusion_matrix_role_a.items():
        lines.append(f"| **`{src}`** | {m.get('agent_alpha', 0.0):.1%} | {m.get('agent_beta', 0.0):.1%} | {m.get('telemetry_sensor', 0.0):.1%} | {m.get('human_controller', 0.0):.1%} | {m.get('auditor_gamma', 0.0):.1%} |")

    lines.extend([
        f"",
        f"### Role B: Beta-Primary (Beta = Self, Alpha = Peer)",
        f"| True Source | Attributed Beta (Self) | Attributed Alpha (Peer) | Attributed Sensor | Attributed Controller | Attributed Auditor |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for src, m in analysis.confusion_matrix_role_b.items():
        lines.append(f"| **`{src}`** | {m.get('agent_beta', 0.0):.1%} | {m.get('agent_alpha', 0.0):.1%} | {m.get('telemetry_sensor', 0.0):.1%} | {m.get('human_controller', 0.0):.1%} | {m.get('auditor_gamma', 0.0):.1%} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 5. Scientific Conclusion",
        f"",
        f"- **Primary Role Reversal:** $\\Delta_{{\\text{{role}}}} = \\mathbf{{{analysis.delta_role_reversal_shift.point_estimate:+.1%}}}$ (95% CI: [{analysis.delta_role_reversal_shift.ci_lower_95:+.1%}, {analysis.delta_role_reversal_shift.ci_upper_95:+.1%}], $p = {analysis.delta_role_reversal_shift.permutation_p_value:.4f}$).",
        f"- **Lexical Bias:** $\\text{{Bias}}_{{\\text{{alpha}}}} = \\mathbf{{{analysis.alpha_lexical_token_bias.point_estimate:+.1%}}}$.",
        f"- **Instrument Ceiling:** $\\text{{Ceiling}} = \\mathbf{{{analysis.isolated_ceiling_overall_accuracy.point_estimate:.1%}}}$ without memory load.",
    ])

    return "\n".join(lines)


def run_e08c_experiment(
    phase: str = "exploratory",
    seed: int = 42,
    total_episodes: int = 4,
    model_name: str = "qwen2.5:3b",
    use_mock: bool = False,
    output_dir: Optional[Path] = None,
) -> Path:
    """Execute Experiment E08c primary-role counterbalance and ceiling control."""
    start_time = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_e08c_role_{timestamp}_{phase}"

    if output_dir is None:
        target_dir = Path("results") / "e08c_role_counterbalance" / run_id
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

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting E08c {phase.upper()} run: {total_episodes} pairs ({total_episodes*2} episodes)...")

    for i in range(total_episodes):
        ep_seed = seed + i * 5003

        # Generate Role A: Alpha-Primary
        ep_a = generator.generate_episode(twin_idx=i, seed=ep_seed, role_mapping="alpha_self_beta_peer")
        ep_a.episode_id = f"{ep_a.episode_id}_roleA"
        for p in ep_a.probes_attribution_5afc + ep_a.probes_self_peer_objective + ep_a.probes_self_peer_belief + list(ep_a.probes_framing_pair) + ep_a.probes_isolated_ceiling_5afc:
            p.metadata["role_mapping"] = "alpha_self_beta_peer"
        for s in ep_a.cue_conflict_specs + ep_a.channel_factorial_specs:
            s.metadata["role_mapping"] = "alpha_self_beta_peer"
            s.probe.metadata["role_mapping"] = "alpha_self_beta_peer"
        ep_a.pressure_probe_pre.metadata["role_mapping"] = "alpha_self_beta_peer"
        ep_a.pressure_probe_post.metadata["role_mapping"] = "alpha_self_beta_peer"

        # Generate Role B: Beta-Primary
        ep_b = generator.generate_episode(twin_idx=i, seed=ep_seed, role_mapping="beta_self_alpha_peer")
        ep_b.episode_id = f"{ep_b.episode_id}_roleB"
        for p in ep_b.probes_attribution_5afc + ep_b.probes_self_peer_objective + ep_b.probes_self_peer_belief + list(ep_b.probes_framing_pair) + ep_b.probes_isolated_ceiling_5afc:
            p.metadata["role_mapping"] = "beta_self_alpha_peer"
        for s in ep_b.cue_conflict_specs + ep_b.channel_factorial_specs:
            s.metadata["role_mapping"] = "beta_self_alpha_peer"
            s.probe.metadata["role_mapping"] = "beta_self_alpha_peer"
        ep_b.pressure_probe_pre.metadata["role_mapping"] = "beta_self_alpha_peer"
        ep_b.pressure_probe_post.metadata["role_mapping"] = "beta_self_alpha_peer"

        # Execute Role A
        trials_a = harness.execute_e08_episode(ep_a)
        all_trials.extend(trials_a)

        # Execute Role B
        trials_b = harness.execute_e08_episode(ep_b)
        all_trials.extend(trials_b)

        episodes_manifest.append({
            "pair_index": i,
            "episode_id_a": ep_a.episode_id,
            "episode_id_b": ep_b.episode_id,
            "trials_count": len(trials_a) + len(trials_b),
        })

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed Episode Pair {i+1}/{total_episodes} ({len(trials_a)+len(trials_b)} trials)")

    # Save trials.jsonl
    with open(target_dir / "trials.jsonl", "w", encoding="utf-8") as f:
        for t in all_trials:
            f.write(json.dumps(asdict(t)) + "\n")

    analysis = analyze_e08c_results(all_trials, num_bootstrap=2000, seed=seed)

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

    report_md = generate_e08c_markdown_report(manifest, analysis)
    with open(target_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] E08c run completed successfully. Output saved to {target_dir}")
    return target_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment E08c: Primary-Role Counterbalance")
    parser.add_argument("--phase", choices=["exploratory", "confirmatory"], default="exploratory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--model", type=str, default="qwen2.5:3b")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()
    run_e08c_experiment(
        phase=args.phase,
        seed=args.seed,
        total_episodes=args.episodes,
        model_name=args.model,
        use_mock=args.mock,
        output_dir=args.output_dir,
    )
