"""Experiment E08d: Role-Channel Ablation / Instrument Autopsy (Sprint S09d).

Investigates which elements of the prompt-level role packaging cause the direct-mention
5AFC positive control to fail, testing 4 ablation conditions:
1. Condition 1 (Full Role Package): Preamble + Legend + Role-labeled choices
2. Condition 2 (Actor-Only Choices + Legend + Preamble): Preamble + Legend + Actor-only choices
3. Condition 3 (No Legend + Preamble): Preamble + No Legend + Actor-only choices
4. Condition 4 (Fully Neutral Direct Lookup): Neutral Preamble + No Legend + Actor-only choices
"""

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from recurrence.backends.ollama import OllamaBackend
from recurrence.tasks.ownership import (
    OwnershipTaskGenerator,
    OwnershipEpisode,
    OwnershipProbe,
    get_actor_display_names,
    get_role_legend,
    get_role_preamble,
)
from recurrence.loop.ownership_experiment import (
    OwnershipHarness,
    OwnershipTrialResult,
)
from recurrence.analysis.ownership_metrics import (
    compute_clustered_bootstrap_ci,
    EstimandWithUncertainty,
)
from experiments.e08_source_ownership.run import MockOwnershipBackend


@dataclass
class ConditionAblationSummary:
    """Summary of 5AFC direct-mention performance for a single ablation condition."""
    condition_id: str
    condition_label: str
    total_trials: int
    overall_accuracy: float
    ci_lower_95: float
    ci_upper_95: float
    self_attribution_rate: float
    accuracy_by_source: Dict[str, float]
    attribution_by_actor: Dict[str, float]


@dataclass
class RoleAblationAnalysisSummary:
    """Analytical summary for Experiment E08d Role-Channel Ablation."""
    total_episode_pairs: int
    total_episodes: int
    total_trials: int
    conditions: Dict[str, ConditionAblationSummary]
    diagnostic_conclusion: str


def analyze_e08d_results(
    trials: List[OwnershipTrialResult],
    num_bootstrap: int = 2000,
    seed: int = 42,
) -> RoleAblationAnalysisSummary:
    """Analyze E08d ablation trials across all 4 conditions."""
    df = pd.DataFrame([asdict(t) for t in trials])
    episodes = df["episode_id"].unique().tolist()
    n_episodes = len(episodes)
    n_pairs = n_episodes // 2

    cond_defs = [
        ("c1_full_package", "Full Role Package (Preamble + Legend + Labeled Choices)"),
        ("c2_actor_only_choices", "Actor-Only Choices (Preamble + Legend + Actor IDs)"),
        ("c3_no_legend", "No Legend (Preamble + Actor IDs)"),
        ("c4_neutral_lookup", "Fully Neutral Direct Lookup (No Role Language)"),
    ]

    cond_summaries = {}
    sources = ["self", "environment", "experimenter", "peer_agent", "observer"]
    actors = ["agent_alpha", "agent_beta", "telemetry_sensor", "human_controller", "auditor_gamma"]

    for cid, clabel in cond_defs:
        sub = df[df["condition_name"] == cid]
        if len(sub) == 0:
            continue

        acc = float(sub["is_correct"].mean())
        ep_accs = [float(sub[sub["episode_id"] == ep]["is_correct"].mean()) for ep in sub["episode_id"].unique()]
        _, lo, hi, _ = compute_clustered_bootstrap_ci(ep_accs, baseline=0.20, num_bootstrap=num_bootstrap, seed=seed)

        # Self attribution rate (attributed to designated Self role)
        self_attr_count = 0
        for _, row in sub.iterrows():
            rm = row["metadata"].get("role_mapping", "alpha_self_beta_peer")
            self_actor = "agent_alpha" if rm == "alpha_self_beta_peer" else "agent_beta"
            if row["attributed_actor"] == self_actor:
                self_attr_count += 1
        self_rate = float(self_attr_count / len(sub)) if len(sub) > 0 else 0.0

        acc_by_src = {}
        for src in sources:
            sub_src = sub[sub["target_source"] == src]
            acc_by_src[src] = float(sub_src["is_correct"].mean()) if len(sub_src) > 0 else 0.0

        attr_by_act = {}
        for act in actors:
            attr_by_act[act] = float((sub["attributed_actor"] == act).mean()) if len(sub) > 0 else 0.0

        cond_summaries[cid] = ConditionAblationSummary(
            condition_id=cid,
            condition_label=clabel,
            total_trials=len(sub),
            overall_accuracy=acc,
            ci_lower_95=lo,
            ci_upper_95=hi,
            self_attribution_rate=self_rate,
            accuracy_by_source=acc_by_src,
            attribution_by_actor=attr_by_act,
        )

    # Diagnostic inference
    c1_acc = cond_summaries.get("c1_full_package", ConditionAblationSummary("", "", 0, 0, 0, 0, 0, {}, {})).overall_accuracy
    c4_acc = cond_summaries.get("c4_neutral_lookup", ConditionAblationSummary("", "", 0, 0, 0, 0, 0, {}, {})).overall_accuracy

    if c4_acc >= 0.85:
        diagnostic = (
            f"Role semantics interfere with literal source retrieval. Neutral lookup achieves ceiling ({c4_acc:.1%}) "
            f"whereas the full role package collapses ({c1_acc:.1%})."
        )
    elif c4_acc > c1_acc + 0.20:
        diagnostic = (
            f"Substantial role-interference effect: Neutral direct lookup ({c4_acc:.1%}) significantly outperforms "
            f"the full role package ({c1_acc:.1%})."
        )
    else:
        diagnostic = (
            f"The 5AFC direct-mention task format remains difficult even under neutral lookup ({c4_acc:.1%}), "
            f"motivating transition to a binary 2AFC reality-monitoring benchmark."
        )

    return RoleAblationAnalysisSummary(
        total_episode_pairs=n_pairs,
        total_episodes=n_episodes,
        total_trials=len(trials),
        conditions=cond_summaries,
        diagnostic_conclusion=diagnostic,
    )


def generate_e08d_markdown_report(
    manifest: Dict[str, Any],
    analysis: RoleAblationAnalysisSummary,
) -> str:
    """Generate Markdown report for Experiment E08d."""
    lines = [
        f"# Experiment E08d: Role-Channel Ablation / Instrument Autopsy Report (Sprint S09d)",
        f"",
        f"**Run ID:** `{manifest['run_id']}`  ",
        f"**Model:** `{manifest['target_model']}` (`{manifest['model_digest'][:12]}...`)  ",
        f"**Phase:** `{manifest['phase'].upper()}` (Seed: `{manifest['seed']}`)  ",
        f"**Date:** {manifest['start_time']}  ",
        f"**Scope:** {analysis.total_episode_pairs} Matched Pairs ({analysis.total_episodes} Episodes) | {analysis.total_trials} Total Direct Probes  ",
        f"**Primary Question:** *Which elements of prompt-level role packaging cause the direct-mention positive control to fail? Does neutral direct lookup jump to ceiling?*",
        f"",
        f"---",
        f"",
        f"## 1. Ablation Condition Matrix & Direct-Lookup Accuracy",
        f"",
        f"| Condition ID | Description | Trials | Overall Accuracy | 95% Clustered CI | Self-Attribution Rate | Diagnostic Inference |",
        f"| :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    for cid, c in analysis.conditions.items():
        diag_tag = "**Ceiling Solvable**" if c.overall_accuracy >= 0.85 else ("**Partial Relief**" if c.overall_accuracy >= 0.40 else "**Failed / Role-Captured**")
        lines.append(
            f"| **`{c.condition_id}`** | {c.condition_label} | {c.total_trials} | **{c.overall_accuracy:.1%}** | [{c.ci_lower_95:.1%}, {c.ci_upper_95:.1%}] | **{c.self_attribution_rate:.1%}** | {diag_tag} |"
        )

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 2. Per-Source Accuracy Breakdown Across Ablation Conditions",
        f"",
        f"| Condition | `self` | `environment` | `experimenter` | `peer_agent` | `observer` |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for cid, c in analysis.conditions.items():
        src_str = " | ".join([f"**{c.accuracy_by_source.get(s, 0.0):.1%}**" for s in ["self", "environment", "experimenter", "peer_agent", "observer"]])
        lines.append(f"| **`{cid}`** | {src_str} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Actor Attribution Distribution Breakdown",
        f"",
        f"| Condition | `agent_alpha` | `agent_beta` | `telemetry_sensor` | `human_controller` | `auditor_gamma` |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for cid, c in analysis.conditions.items():
        act_str = " | ".join([f"{c.attribution_by_actor.get(a, 0.0):.1%}" for a in ["agent_alpha", "agent_beta", "telemetry_sensor", "human_controller", "auditor_gamma"]])
        lines.append(f"| **`{cid}`** | {act_str} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 4. Scientific Autopsy Conclusion",
        f"",
        f"- **Diagnostic Finding:** {analysis.diagnostic_conclusion}",
    ])

    return "\n".join(lines)


def run_e08d_experiment(
    phase: str = "exploratory",
    seed: int = 42,
    total_episode_pairs: int = 4,
    model_name: str = "qwen2.5:3b",
    use_mock: bool = False,
    output_dir: Optional[Path] = None,
) -> Path:
    """Execute Experiment E08d role-channel ablation and instrument autopsy."""
    start_time = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_e08d_ablation_{timestamp}_{phase}"

    if output_dir is None:
        target_dir = Path("results") / "e08d_role_ablation" / run_id
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

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting E08d {phase.upper()} run: {total_episode_pairs} pairs ({total_episode_pairs*2} episodes)...")

    # Generate matched pairs
    twin_episodes: List[Tuple[OwnershipEpisode, OwnershipEpisode]] = []
    for pair_idx in range(total_episode_pairs):
        ep_seed = seed + pair_idx * 5003
        ep_a = generator.generate_episode(
            twin_idx=pair_idx,
            seed=ep_seed,
            role_mapping="alpha_self_beta_peer",
        )
        ep_a.episode_id = f"{ep_a.episode_id}_roleA"
        for p in ep_a.probes_isolated_ceiling_5afc:
            p.metadata["role_mapping"] = "alpha_self_beta_peer"

        ep_b = generator.generate_episode(
            twin_idx=pair_idx,
            seed=ep_seed,
            role_mapping="beta_self_alpha_peer",
        )
        ep_b.episode_id = f"{ep_b.episode_id}_roleB"
        for p in ep_b.probes_isolated_ceiling_5afc:
            p.metadata["role_mapping"] = "beta_self_alpha_peer"

        twin_episodes.append((ep_a, ep_b))

    for pair_idx, (ep_a, ep_b) in enumerate(twin_episodes):
        for ep in [ep_a, ep_b]:
            role_map = ep.role_mapping
            role_preamble = get_role_preamble(role_map)
            role_legend = get_role_legend(role_map)
            actor_display_names = get_actor_display_names(role_map)

            episodes_manifest.append({
                "episode_id": ep.episode_id,
                "twin_index": pair_idx,
                "role_mapping": role_map,
            })

            # For each direct isolated probe (5 sources per episode)
            for orig_probe in ep.probes_isolated_ceiling_5afc:
                target_actor = orig_probe.target_actor
                target_source = orig_probe.target_source

                # 1. Condition 1: Full Role Package (Preamble + Legend + Labeled Choices)
                prompt_c1, p_hash_c1 = harness._build_prompt(
                    events=None,
                    state=None,
                    probe=orig_probe,
                    role_preamble=role_preamble,
                    include_legend=True,
                    legend_text=role_legend,
                )
                pred_let_c1, pred_text_c1, p_tok_1, c_tok_1, lat_1, err_1 = harness._query_choice(prompt_c1, orig_probe)
                is_corr_c1 = (pred_let_c1 == orig_probe.correct_option)
                attr_c1 = None
                for act_name, disp in actor_display_names.items():
                    if disp == pred_text_c1 or act_name in pred_text_c1:
                        attr_c1 = act_name
                        break

                all_trials.append(OwnershipTrialResult(
                    trial_id=f"{ep.episode_id}_{orig_probe.probe_id}_c1",
                    episode_id=ep.episode_id,
                    experiment_submodule="e08d_role_ablation",
                    condition_name="c1_full_package",
                    probe_id=orig_probe.probe_id,
                    probe_type=orig_probe.probe_type,
                    question=orig_probe.question,
                    options=orig_probe.options,
                    predicted_letter=pred_let_c1,
                    predicted_text=pred_text_c1,
                    correct_letter=orig_probe.correct_option,
                    is_correct=is_corr_c1,
                    attributed_actor=attr_c1,
                    target_source=target_source,
                    target_actor=target_actor,
                    target_value=orig_probe.target_value,
                    prompt_hash=p_hash_c1,
                    prompt_tokens=p_tok_1,
                    completion_tokens=c_tok_1,
                    latency_ms=lat_1,
                    error_message=err_1,
                    metadata={"role_mapping": role_map, "condition": "c1_full_package", "key": orig_probe.metadata.get("key")},
                ))

                # 2. Condition 2: Actor-Only Choices + Legend + Preamble
                # Create raw actor options
                raw_options = {}
                letters = sorted(orig_probe.options.keys())
                for l in letters:
                    opt_str = orig_probe.options[l]
                    # Extract raw actor name
                    for act_name in actor_display_names.keys():
                        if act_name in opt_str:
                            raw_options[l] = act_name
                            break

                corr_let_raw = [l for l, act in raw_options.items() if act == target_actor][0]
                probe_raw = OwnershipProbe(
                    probe_id=orig_probe.probe_id,
                    probe_type=orig_probe.probe_type,
                    question=orig_probe.question,
                    options=raw_options,
                    correct_option=corr_let_raw,
                    target_source=target_source,
                    target_actor=target_actor,
                    target_value=orig_probe.target_value,
                    metadata=orig_probe.metadata,
                )

                prompt_c2, p_hash_c2 = harness._build_prompt(
                    events=None,
                    state=None,
                    probe=probe_raw,
                    role_preamble=role_preamble,
                    include_legend=True,
                    legend_text=role_legend,
                )
                pred_let_c2, pred_text_c2, p_tok_2, c_tok_2, lat_2, err_2 = harness._query_choice(prompt_c2, probe_raw)
                is_corr_c2 = (pred_let_c2 == corr_let_raw)
                attr_c2 = raw_options.get(pred_let_c2, pred_text_c2)

                all_trials.append(OwnershipTrialResult(
                    trial_id=f"{ep.episode_id}_{orig_probe.probe_id}_c2",
                    episode_id=ep.episode_id,
                    experiment_submodule="e08d_role_ablation",
                    condition_name="c2_actor_only_choices",
                    probe_id=orig_probe.probe_id,
                    probe_type=orig_probe.probe_type,
                    question=orig_probe.question,
                    options=raw_options,
                    predicted_letter=pred_let_c2,
                    predicted_text=pred_text_c2,
                    correct_letter=corr_let_raw,
                    is_correct=is_corr_c2,
                    attributed_actor=attr_c2,
                    target_source=target_source,
                    target_actor=target_actor,
                    target_value=orig_probe.target_value,
                    prompt_hash=p_hash_c2,
                    prompt_tokens=p_tok_2,
                    completion_tokens=c_tok_2,
                    latency_ms=lat_2,
                    error_message=err_2,
                    metadata={"role_mapping": role_map, "condition": "c2_actor_only_choices", "key": orig_probe.metadata.get("key")},
                ))

                # 3. Condition 3: No Legend + Preamble + Actor-Only Choices
                prompt_c3, p_hash_c3 = harness._build_prompt(
                    events=None,
                    state=None,
                    probe=probe_raw,
                    role_preamble=role_preamble,
                    include_legend=False,
                )
                pred_let_c3, pred_text_c3, p_tok_3, c_tok_3, lat_3, err_3 = harness._query_choice(prompt_c3, probe_raw)
                is_corr_c3 = (pred_let_c3 == corr_let_raw)
                attr_c3 = raw_options.get(pred_let_c3, pred_text_c3)

                all_trials.append(OwnershipTrialResult(
                    trial_id=f"{ep.episode_id}_{orig_probe.probe_id}_c3",
                    episode_id=ep.episode_id,
                    experiment_submodule="e08d_role_ablation",
                    condition_name="c3_no_legend",
                    probe_id=orig_probe.probe_id,
                    probe_type=orig_probe.probe_type,
                    question=orig_probe.question,
                    options=raw_options,
                    predicted_letter=pred_let_c3,
                    predicted_text=pred_text_c3,
                    correct_letter=corr_let_raw,
                    is_correct=is_corr_c3,
                    attributed_actor=attr_c3,
                    target_source=target_source,
                    target_actor=target_actor,
                    target_value=orig_probe.target_value,
                    prompt_hash=p_hash_c3,
                    prompt_tokens=p_tok_3,
                    completion_tokens=c_tok_3,
                    latency_ms=lat_3,
                    error_message=err_3,
                    metadata={"role_mapping": role_map, "condition": "c3_no_legend", "key": orig_probe.metadata.get("key")},
                ))

                # 4. Condition 4: Fully Neutral Direct Lookup (No role language)
                prompt_c4, p_hash_c4 = harness._build_prompt(
                    events=None,
                    state=None,
                    probe=probe_raw,
                    role_preamble="You are an automated source attribution evaluator. Answer the question directly using the single event given below.",
                    include_legend=False,
                )
                pred_let_c4, pred_text_c4, p_tok_4, c_tok_4, lat_4, err_4 = harness._query_choice(prompt_c4, probe_raw)
                is_corr_c4 = (pred_let_c4 == corr_let_raw)
                attr_c4 = raw_options.get(pred_let_c4, pred_text_c4)

                all_trials.append(OwnershipTrialResult(
                    trial_id=f"{ep.episode_id}_{orig_probe.probe_id}_c4",
                    episode_id=ep.episode_id,
                    experiment_submodule="e08d_role_ablation",
                    condition_name="c4_neutral_lookup",
                    probe_id=orig_probe.probe_id,
                    probe_type=orig_probe.probe_type,
                    question=orig_probe.question,
                    options=raw_options,
                    predicted_letter=pred_let_c4,
                    predicted_text=pred_text_c4,
                    correct_letter=corr_let_raw,
                    is_correct=is_corr_c4,
                    attributed_actor=attr_c4,
                    target_source=target_source,
                    target_actor=target_actor,
                    target_value=orig_probe.target_value,
                    prompt_hash=p_hash_c4,
                    prompt_tokens=p_tok_4,
                    completion_tokens=c_tok_4,
                    latency_ms=lat_4,
                    error_message=err_4,
                    metadata={"role_mapping": role_map, "condition": "c4_neutral_lookup", "key": orig_probe.metadata.get("key")},
                ))

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed Pair {pair_idx+1}/{total_episode_pairs} (Total trials: {len(all_trials)})")

    # Analyze results
    analysis = analyze_e08d_results(all_trials, num_bootstrap=2000, seed=seed)

    manifest = {
        "run_id": run_id,
        "experiment": "e08d_role_ablation",
        "phase": phase,
        "start_time": start_time,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "target_model": model_name,
        "model_digest": model_digest,
        "seed": seed,
        "total_episode_pairs": total_episode_pairs,
        "total_episodes": total_episode_pairs * 2,
        "total_trials": len(all_trials),
        "episodes": episodes_manifest,
    }

    # Save artifacts
    with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(target_dir / "trials.jsonl", "w", encoding="utf-8") as f:
        for t in all_trials:
            f.write(json.dumps(asdict(t)) + "\n")

    summary_payload = {
        "manifest": manifest,
        "analysis": asdict(analysis),
    }

    with open(target_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    report_md = generate_e08d_markdown_report(manifest, analysis)
    with open(target_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] E08d run completed successfully. Output saved to {target_dir}")
    return target_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment E08d: Role-Channel Ablation")
    parser.add_argument("--phase", choices=["exploratory", "confirmatory"], default="exploratory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--model", type=str, default="qwen2.5:3b")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    run_e08d_experiment(
        phase=args.phase,
        seed=args.seed,
        total_episode_pairs=args.pairs,
        model_name=args.model,
        use_mock=args.mock,
    )
