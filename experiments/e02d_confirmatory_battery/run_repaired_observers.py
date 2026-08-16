"""Experiment E02d.1: Frozen-Target Observer Interface Repair (N=200).

Reruns external observers (Input Only, Visible Answer, Reconstruction) against
the permanently frozen N=200 target trials from Experiment E02d using a clean
task-body renderer that strips all target JSON schema and response instructions.

Frozen Target Data Sources:
- Qwen 2.5 14B: results/e02d_confirmatory_battery/run_e02d_qwen2_5_14b_20260816_093151/trials.jsonl (H=3, D=16, Seed 1000)
- Qwen 2.5 3B: results/e02d_confirmatory_battery/run_e02d_qwen2_5_3b_20260816_104927/trials.jsonl (H=1, D=8, Seed 2000)
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from recurrence.backends.ollama import OllamaBackend
from recurrence.core.manifest import RunManifest
from recurrence.tasks.adaptive_metacognition import AdaptiveMetacognition2AFCTask, TaskItem
from recurrence.analysis.psychophysics import (
    compute_sdt_indices,
    compute_sdt_bootstrap_ci,
    compute_wilson_score_interval,
    compute_type2_sdt_metrics,
)
from recurrence.analysis.meta_d import fit_meta_d_mle


def extract_clean_task_body(prompt: str) -> str:
    """Extract clean task body containing only Context Information, Question, and Options.
    
    Strips away all target JSON schema, formatting, and response instructions.
    """
    marker = "\n\nRespond strictly with a JSON object"
    if marker in prompt:
        return prompt.split(marker)[0].strip()
    # Fallback: if schema block starts differently
    marker2 = "Respond strictly with a JSON object"
    if marker2 in prompt:
        return prompt.split(marker2)[0].strip()
    return prompt.strip()


def make_reconstruction_schema(candidate_1: str, candidate_2: str) -> Dict[str, Any]:
    """JSON Schema for Reconstruction observer emitting candidate probabilities."""
    return {
        "type": "object",
        "properties": {
            "p_candidate_1": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 100.0,
                "description": f"Probability that '{candidate_1}' is the correct value (0 to 100).",
            },
            "p_candidate_2": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 100.0,
                "description": f"Probability that '{candidate_2}' is the correct value (0 to 100).",
            },
            "rationale": {"type": "string"},
        },
        "required": ["p_candidate_1", "p_candidate_2"],
        "additionalProperties": False,
    }


def make_external_evaluator_schema() -> Dict[str, Any]:
    """JSON Schema for Input-Only and Visible-Answer observers."""
    return {
        "type": "object",
        "properties": {
            "probability": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 100.0,
                "description": "Estimated probability (0 to 100).",
            },
            "rationale": {"type": "string"},
        },
        "required": ["probability"],
        "additionalProperties": False,
    }


def run_e02d1_repaired_observers(
    model_name: str,
    hop_depth: int,
    distractor_count: int,
    frozen_trials_path: str,
    n_trials: int = 200,
    seed: int = 1000,
    temperature: float = 0.0,
    output_dir: str = "results/e02d_confirmatory_battery",
    backend_override: Optional[Any] = None,
) -> Path:
    """Rerun external observers with clean prompts against permanently frozen target trials."""
    start_time = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = model_name.replace(":", "_").replace(".", "_")
    run_id = f"run_e02d1_repaired_{model_clean}_{timestamp}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT E02d.1: FROZEN-TARGET OBSERVER INTERFACE REPAIR (N={n_trials})")
    print(f"Model: {model_name} | Coordinate: (H={hop_depth}, D={distractor_count}) | Seed: {seed}")
    print(f"Frozen Source: {frozen_trials_path}")
    print(f"{'='*70}\n")

    # 1. Load permanently frozen target trials
    frozen_df = pd.read_json(frozen_trials_path, lines=True)
    if len(frozen_df) != n_trials:
        raise ValueError(f"Expected {n_trials} trials in {frozen_trials_path}, found {len(frozen_df)}")

    frozen_records = frozen_df.to_dict(orient="records")
    frozen_map = {r["trial_id"]: r for r in frozen_records}

    if backend_override is not None:
        backend = backend_override
        model_digest = getattr(backend, "get_digest", lambda: "mock_digest")()
    else:
        backend = OllamaBackend(model_name=model_name, temperature=temperature, seed=seed)
        model_digest = backend.get_digest()

    task = AdaptiveMetacognition2AFCTask(
        task_family="multi_hop",
        ask_confidence=True,
        response_mode="direct_value",
    )

    # 2. Regenerate exact same task items
    items = task.generate_multi_hop_grid(
        grid_cells=[(hop_depth, distractor_count)],
        count_per_cell=n_trials,
        base_seed=seed,
        ask_confidence=True,
    )

    print(f"Regenerated {len(items)} items. Rerunning 3 clean external observers against frozen target choices...\n")

    schema_ext = make_external_evaluator_schema()
    repaired_records: List[Dict[str, Any]] = []

    for idx, it in enumerate(items, 1):
        f_trial = frozen_map.get(it.item_id)
        if f_trial is None:
            raise KeyError(f"Trial ID {it.item_id} not found in frozen trial map")

        opt_map = it.metadata.get("option_map", {})
        c1 = opt_map.get("A", "")
        c2 = opt_map.get("B", "")

        # Extract clean task body with no target JSON instructions
        clean_body = extract_clean_task_body(it.prompt)

        # Recover frozen target choice & Self probability
        target_choice = f_trial["selected_value"]
        target_parsed_letter = f_trial["parsed_answer"]
        target_correct = f_trial["correct"]
        self_conf = f_trial["probability_self"]
        compliance_self = f_trial["compliance_self"]

        # -------------------------------------------------------------
        # STEP 1: Input-Only Observer (Clean prompt + construct: target checkpoint success prob)
        # -------------------------------------------------------------
        prompt_input_only = (
            f"{clean_body}\n\n"
            f"EVALUATION TASK:\n"
            f"Evaluate the difficulty of the above relational multi-hop retrieval task.\n"
            f"Estimate the probability (0 to 100) that the target model (without seeing its response) would answer this item correctly."
        )
        resp_io = backend.generate_with_schema(prompt=prompt_input_only, json_schema=schema_ext)
        compliance_io = False
        io_conf = None
        if resp_io.parsed is not None and "probability" in resp_io.parsed:
            try:
                pval = float(resp_io.parsed["probability"])
                if 0.0 <= pval <= 100.0:
                    compliance_io = True
                    io_conf = pval
            except (ValueError, TypeError):
                compliance_io = False

        # -------------------------------------------------------------
        # STEP 2: Visible-Answer Observer (Clean prompt + frozen target choice)
        # -------------------------------------------------------------
        prompt_vis_ans = (
            f"{clean_body}\n\n"
            f"PROPOSED ANSWER:\n"
            f"The target model selected candidate value: \"{target_choice}\"\n\n"
            f"EVALUATION TASK:\n"
            f"Estimate the probability (0 to 100) that this proposed candidate value is the correct answer."
        )
        resp_va = backend.generate_with_schema(prompt=prompt_vis_ans, json_schema=schema_ext)
        compliance_va = False
        va_conf = None
        if resp_va.parsed is not None and "probability" in resp_va.parsed:
            try:
                pval = float(resp_va.parsed["probability"])
                if 0.0 <= pval <= 100.0:
                    compliance_va = True
                    va_conf = pval
            except (ValueError, TypeError):
                compliance_va = False

        # -------------------------------------------------------------
        # STEP 3: Reconstruction Observer (Clean prompt + independent 2-candidate solve)
        # -------------------------------------------------------------
        prompt_reconstruct = (
            f"{clean_body}\n\n"
            f"INDEPENDENT RECONSTRUCTION TASK:\n"
            f"Independently evaluate both candidate values: Candidate 1 = \"{c1}\", Candidate 2 = \"{c2}\".\n"
            f"Assign a probability (0 to 100) to each candidate reflecting how likely it is to be the true target value.\n"
            f"The probabilities must sum to 100."
        )
        schema_rec = make_reconstruction_schema(c1, c2)
        resp_rec = backend.generate_with_schema(prompt=prompt_reconstruct, json_schema=schema_rec)
        compliance_rec = False
        rec_conf_target = None
        if resp_rec.parsed is not None and "p_candidate_1" in resp_rec.parsed and "p_candidate_2" in resp_rec.parsed:
            try:
                p1 = float(resp_rec.parsed["p_candidate_1"])
                p2 = float(resp_rec.parsed["p_candidate_2"])
                if p1 <= 1.0 and p2 <= 1.0 and (p1 + p2) <= 1.5:
                    p1 *= 100.0
                    p2 *= 100.0
                total_p = p1 + p2
                if 95.0 <= total_p <= 105.0 and 0.0 <= p1 <= 100.0 and 0.0 <= p2 <= 100.0:
                    compliance_rec = True
                    rec_conf_target = p1 if target_choice == c1 else p2
            except (ValueError, TypeError):
                compliance_rec = False

        all_compliant = bool(compliance_self and compliance_io and compliance_va and compliance_rec)

        rec = {
            "trial_id": it.item_id,
            "hop_depth": hop_depth,
            "distractor_count": distractor_count,
            "ground_truth_letter": it.ground_truth,
            "ground_truth_value": f_trial.get("ground_truth_value"),
            "selected_value": target_choice,
            "parsed_answer": target_parsed_letter,
            "ground_truth": it.ground_truth,
            "correct": target_correct,
            "compliance_self": compliance_self,
            "compliance_input_only": compliance_io,
            "compliance_visible_answer": compliance_va,
            "compliance_reconstruction": compliance_rec,
            "all_observers_compliant": all_compliant,
            "probability_self": self_conf,
            "probability_input_only": io_conf,
            "probability_visible_answer": va_conf,
            "probability_reconstruction": rec_conf_target,
            "latency_self": f_trial.get("latency_self"),
        }
        repaired_records.append(rec)

        if idx % 10 == 0 or idx == n_trials:
            n_c = sum(1 for r in repaired_records if r["all_observers_compliant"])
            print(f"[{idx}/{n_trials}] All-Compliant: {n_c}/{idx} | IO: {io_conf} | VA: {va_conf} | Rec: {rec_conf_target}")

    # Save repaired trials
    trials_df = pd.DataFrame(repaired_records)
    trials_df.to_json(run_dir / "trials.jsonl", orient="records", lines=True)

    # -------------------------------------------------------------
    # STATISTICAL ANALYSIS & PAI ESTIMATION
    # -------------------------------------------------------------
    # 1. Hard Measurement Compliance Gate
    comp_self_rate = float(trials_df["compliance_self"].mean())
    comp_io_rate = float(trials_df["compliance_input_only"].mean())
    comp_va_rate = float(trials_df["compliance_visible_answer"].mean())
    comp_rec_rate = float(trials_df["compliance_reconstruction"].mean())
    min_comp_rate = min(comp_self_rate, comp_io_rate, comp_va_rate, comp_rec_rate)
    measurement_gate_passed = bool(min_comp_rate >= 0.95)

    # 2. Descriptive Self Metacognition on ALL Valid Self Trials (N=200)
    self_valid_df = trials_df[trials_df["compliance_self"]].copy()
    n_self_valid = len(self_valid_df)
    recs_self_valid = self_valid_df.to_dict(orient="records")
    sdt_t1_self = compute_sdt_indices(recs_self_valid)
    t1_ci_self = compute_sdt_bootstrap_ci(recs_self_valid, n_bootstraps=1000)
    t1_self_d_ci = (float(t1_ci_self["d_prime_ci_lower"]), float(t1_ci_self["d_prime_ci_upper"]))
    t1_self_c_ci = (float(t1_ci_self["criterion_c_ci_lower"]), float(t1_ci_self["criterion_c_ci_upper"]))

    cond_self_all = [
        {"probability": r["probability_self"], "correct": r["correct"], "ground_truth": r["ground_truth"], "parsed_answer": r["parsed_answer"]}
        for r in recs_self_valid if r["probability_self"] is not None
    ]
    m_self_all_trials = compute_type2_sdt_metrics(cond_self_all, fit_meta_d=True, n_bins=4)

    # 3. First-Order Performance on Shared 4-Observer Valid Intersection
    shared_valid_df = trials_df[trials_df["all_observers_compliant"]].copy()
    n_shared_valid = len(shared_valid_df)
    recs_shared = shared_valid_df.to_dict(orient="records")
    sdt_t1_shared = compute_sdt_indices(recs_shared)
    t1_ci_shared = compute_sdt_bootstrap_ci(recs_shared, n_bootstraps=1000)
    t1_shared_d_ci = (float(t1_ci_shared["d_prime_ci_lower"]), float(t1_ci_shared["d_prime_ci_upper"]))
    t1_shared_c_ci = (float(t1_ci_shared["criterion_c_ci_lower"]), float(t1_ci_shared["criterion_c_ci_upper"]))

    # 4. Type-2 Metrics on Shared Intersection
    def get_cond_metrics(prob_col: str, fit_m: bool = False) -> Dict[str, Any]:
        cond_recs = [
            {"probability": r[prob_col], "correct": r["correct"], "ground_truth": r["ground_truth"], "parsed_answer": r["parsed_answer"]}
            for r in recs_shared if r[prob_col] is not None
        ]
        return compute_type2_sdt_metrics(cond_recs, fit_meta_d=fit_m, n_bins=4)

    m_self_shared = get_cond_metrics("probability_self", fit_m=True)
    m_io = get_cond_metrics("probability_input_only", fit_m=False)
    m_va = get_cond_metrics("probability_visible_answer", fit_m=False)
    m_rec = get_cond_metrics("probability_reconstruction", fit_m=False)

    auroc_self = float(m_self_shared.get("auroc2") if m_self_shared.get("auroc2") is not None else 0.5)
    auroc_io = float(m_io.get("auroc2") if m_io.get("auroc2") is not None else 0.5)
    auroc_va = float(m_va.get("auroc2") if m_va.get("auroc2") is not None else 0.5)
    auroc_rec = float(m_rec.get("auroc2") if m_rec.get("auroc2") is not None else 0.5)

    max_external_auroc = max(auroc_io, auroc_va, auroc_rec)
    point_pai = auroc_self - max_external_auroc

    # 5. Stratified Paired Bootstrapping (B=2000)
    boot_pai: List[float] = []
    boot_self_v_io: List[float] = []
    boot_self_v_va: List[float] = []
    boot_self_v_rec: List[float] = []

    corrects = shared_valid_df["correct"].values.astype(int)
    p_self = shared_valid_df["probability_self"].values.astype(float)
    p_io = shared_valid_df["probability_input_only"].values.astype(float)
    p_va = shared_valid_df["probability_visible_answer"].values.astype(float)
    p_rec = shared_valid_df["probability_reconstruction"].values.astype(float)

    idx_corr = np.where(corrects == 1)[0]
    idx_inc = np.where(corrects == 0)[0]
    n_corr = len(idx_corr)
    n_inc = len(idx_inc)

    rng = np.random.default_rng(seed)
    if n_corr > 0 and n_inc > 0:
        for _ in range(2000):
            b_corr_idx = rng.choice(idx_corr, size=n_corr, replace=True)
            b_inc_idx = rng.choice(idx_inc, size=n_inc, replace=True)
            idx_b = np.concatenate([b_corr_idx, b_inc_idx])

            def boot_auroc(p_arr: np.ndarray) -> float:
                c = p_arr[b_corr_idx]
                i = p_arr[b_inc_idx]
                if len(c) == 0 or len(i) == 0 or np.std(p_arr[idx_b]) < 1e-4:
                    return 0.5
                conc = sum(1.0 for x in c for y in i if x > y)
                ties = sum(0.5 for x in c for y in i if x == y)
                return float((conc + ties) / (len(c) * len(i)))

            b_a_self = boot_auroc(p_self)
            b_a_io = boot_auroc(p_io)
            b_a_va = boot_auroc(p_va)
            b_a_rec = boot_auroc(p_rec)

            b_max_ext = max(b_a_io, b_a_va, b_a_rec)
            boot_pai.append(b_a_self - b_max_ext)
            boot_self_v_io.append(b_a_self - b_a_io)
            boot_self_v_va.append(b_a_self - b_a_va)
            boot_self_v_rec.append(b_a_self - b_a_rec)

    pai_ci = (float(np.percentile(boot_pai, 2.5)), float(np.percentile(boot_pai, 97.5))) if boot_pai else (0.0, 0.0)
    ci_io = (float(np.percentile(boot_self_v_io, 2.5)), float(np.percentile(boot_self_v_io, 97.5))) if boot_self_v_io else (0.0, 0.0)
    ci_va = (float(np.percentile(boot_self_v_va, 2.5)), float(np.percentile(boot_self_v_va, 97.5))) if boot_self_v_va else (0.0, 0.0)
    ci_rec = (float(np.percentile(boot_self_v_rec, 2.5)), float(np.percentile(boot_self_v_rec, 97.5))) if boot_self_v_rec else (0.0, 0.0)

    pai_sesoi_passed = bool(pai_ci[0] > 0.05) if measurement_gate_passed else False

    summary = {
        "run_id": run_id,
        "experiment": "e02d.1_frozen_target_observer_repair",
        "model_name": model_name,
        "model_digest": model_digest,
        "hop_depth": hop_depth,
        "distractor_count": distractor_count,
        "total_trials": n_trials,
        "frozen_trials_source": frozen_trials_path,
        "valid_immediate_self_trials": n_self_valid,
        "valid_shared_intersection_trials": n_shared_valid,
        "compliance_rates": {
            "self": comp_self_rate,
            "input_only": comp_io_rate,
            "visible_answer": comp_va_rate,
            "reconstruction": comp_rec_rate,
            "minimum_compliance_rate": min_comp_rate,
            "measurement_gate_passed": measurement_gate_passed,
        },
        "type1_metrics_immediate_self": {
            "accuracy": float(self_valid_df["correct"].mean()) if n_self_valid > 0 else 0.0,
            "accuracy_ci": compute_wilson_score_interval(int(self_valid_df["correct"].sum()), n_self_valid) if n_self_valid > 0 else (0.0, 0.0),
            "sdt_d_prime": sdt_t1_self.get("d_prime"),
            "sdt_d_prime_ci": t1_self_d_ci,
            "sdt_criterion_c": sdt_t1_self.get("criterion_c"),
            "sdt_criterion_c_ci": t1_self_c_ci,
        },
        "type1_metrics_shared_intersection": {
            "accuracy": float(np.mean(corrects)) if n_shared_valid > 0 else 0.0,
            "accuracy_ci": compute_wilson_score_interval(int(np.sum(corrects)), n_shared_valid) if n_shared_valid > 0 else (0.0, 0.0),
            "sdt_d_prime": sdt_t1_shared.get("d_prime"),
            "sdt_d_prime_ci": t1_shared_d_ci,
            "sdt_criterion_c": sdt_t1_shared.get("criterion_c"),
            "sdt_criterion_c_ci": t1_shared_c_ci,
        },
        "descriptive_self_all_trials": m_self_all_trials,
        "type2_metrics_shared_intersection": {
            "self": m_self_shared,
            "input_only": m_io,
            "visible_answer": m_va,
            "reconstruction": m_rec,
        },
        "privileged_access_index": {
            "point_estimate": point_pai,
            "bootstrap_95_ci": list(pai_ci),
            "max_external_auroc": max_external_auroc,
            "prespecified_sesoi_passed": pai_sesoi_passed,
            "historical_sesoi_010_passed": bool(pai_ci[0] > 0.10) if measurement_gate_passed else False,
            "pairwise_contrasts": {
                "self_minus_input_only": {
                    "point_estimate": float(auroc_self - auroc_io),
                    "mean_bootstrap": float(np.mean(boot_self_v_io)) if boot_self_v_io else 0.0,
                    "bootstrap_95_ci": list(ci_io),
                },
                "self_minus_visible_answer": {
                    "point_estimate": float(auroc_self - auroc_va),
                    "mean_bootstrap": float(np.mean(boot_self_v_va)) if boot_self_v_va else 0.0,
                    "bootstrap_95_ci": list(ci_va),
                },
                "self_minus_reconstruction": {
                    "point_estimate": float(auroc_self - auroc_rec),
                    "mean_bootstrap": float(np.mean(boot_self_v_rec)) if boot_self_v_rec else 0.0,
                    "bootstrap_95_ci": list(ci_rec),
                },
            },
        },
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    manifest = RunManifest(
        experiment_id="e02d.1_repaired_observers",
        run_id=run_id,
        seed=seed,
        model_tag=model_name,
        model_digest=model_digest,
        parameters={
            "hop_depth": hop_depth,
            "distractor_count": distractor_count,
            "total_trials": n_trials,
            "valid_trials": n_shared_valid,
            "frozen_source": frozen_trials_path,
            "temperature": temperature,
            "measurement_gate_passed": measurement_gate_passed,
        },
    )
    manifest.compute_environment_hash()
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest.model_dump(), f, indent=2)

    auroc_all = float(m_self_all_trials.get("auroc2") if m_self_all_trials.get("auroc2") is not None else 0.5)
    brier_all = float(m_self_all_trials.get("brier_score") if m_self_all_trials.get("brier_score") is not None else 0.0)
    mean_conf_all = float(m_self_all_trials.get("mean_confidence") if m_self_all_trials.get("mean_confidence") is not None else 0.0)

    # Generate Markdown Report
    t1_self = summary["type1_metrics_immediate_self"]
    t1_sh = summary["type1_metrics_shared_intersection"]

    sesoi_text = (
        "**PASSED (Privileged Access Supported)**"
        if pai_sesoi_passed
        else "**SESOI threshold not met / no meaningful positive privileged-access advantage resolved.**"
    )

    report_md = f"""# Experiment E02d.1: Frozen-Target Observer Interface Repair Report (N={n_trials})

**Run ID:** `{run_id}`  
**Model:** `{model_name}` (`{model_digest[:12]}...`)  
**Frozen Operating Coordinate:** `(H={hop_depth}, D={distractor_count})`  
**Frozen Target Source:** `{frozen_trials_path}`  
**Valid Immediate-Self Trials:** {n_self_valid}/{n_trials} ({comp_self_rate*100:.1f}%)  
**Shared 4-Observer Valid Intersection:** {n_shared_valid}/{n_trials} ({n_shared_valid/n_trials*100:.1f}%)  
**Measurement Gate ($\\\\ge 95\\%$ across all observers):** {'**PASS**' if measurement_gate_passed else '**FAIL (Diagnostic Only)**'}  

---

## 1. Type-1 Manipulation Checks & First-Order Operating Point

### Primary Target Trials (All Valid Immediate-Self Runs, N={n_self_valid})
| Metric | Point Estimate | 95% Bootstrap CI | Calibration Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **{t1_self['accuracy']*100:.1f}%** | [{t1_self['accuracy_ci'][0]*100:.1f}%, {t1_self['accuracy_ci'][1]*100:.1f}%] | $60\\% \\le \\text{{Acc}} \\le 80\\%$ | {'PASS' if 0.60 <= t1_self['accuracy'] <= 0.80 else 'FAIL'} |
| **SDT $d'$** | **{t1_self['sdt_d_prime']:+.2f}** | [{t1_self['sdt_d_prime_ci'][0]:+.2f}, {t1_self['sdt_d_prime_ci'][1]:+.2f}] | $0.90 \\le d' \\le 1.40$ | {'PASS' if 0.90 <= (t1_self['sdt_d_prime'] or 0) <= 1.40 else 'FAIL'} |
| **SDT $c$** | **{t1_self['sdt_criterion_c']:+.2f}** | [{t1_self['sdt_criterion_c_ci'][0]:+.2f}, {t1_self['sdt_criterion_c_ci'][1]:+.2f}] | $|c| \\le 0.50$ | {'PASS' if abs(t1_self['sdt_criterion_c'] or 0) <= 0.50 else 'FAIL'} |

### Shared 4-Observer Valid Intersection (PAI Evaluation Set, N={n_shared_valid})
| Metric | Point Estimate | 95% Bootstrap CI | Calibration Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **{t1_sh['accuracy']*100:.1f}%** | [{t1_sh['accuracy_ci'][0]*100:.1f}%, {t1_sh['accuracy_ci'][1]*100:.1f}%] | $60\\% \\le \\text{{Acc}} \\le 80\\%$ | {'PASS' if 0.60 <= t1_sh['accuracy'] <= 0.80 else 'FAIL'} |
| **SDT $d'$** | **{t1_sh['sdt_d_prime']:+.2f}** | [{t1_sh['sdt_d_prime_ci'][0]:+.2f}, {t1_sh['sdt_d_prime_ci'][1]:+.2f}] | $0.90 \\le d' \\le 1.40$ | {'PASS' if 0.90 <= (t1_sh['sdt_d_prime'] or 0) <= 1.40 else 'FAIL'} |
| **SDT $c$** | **{t1_sh['sdt_criterion_c']:+.2f}** | [{t1_sh['sdt_criterion_c_ci'][0]:+.2f}, {t1_sh['sdt_criterion_c_ci'][1]:+.2f}] | $|c| \\le 0.50$ | {'PASS' if abs(t1_sh['sdt_criterion_c'] or 0) <= 0.50 else 'FAIL'} |

---

## 2. Type-2 Observer Ladder & Metacognitive Performance

| Observer Condition | Informational Input | AUROC2 | Brier Score | Mean Prob | Compliance | Meta-$d'$ / $M$-Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Immediate Self** | Same-invocation target choice + confidence | **{auroc_self:.3f}** | {m_self_shared.get('brier_score', 0):.3f} | {m_self_shared.get('mean_confidence', 0):.1f}% | {comp_self_rate*100:.1f}% | {f"{m_self_shared.get('meta_d_prime', 0):.2f} ({m_self_shared.get('m_ratio', 0):.2f})" if m_self_shared.get('meta_d_prime') is not None else ('N/A (confidence_degenerate)' if m_self_shared.get('meta_d_status') == 'confidence_degenerate' else 'N/A')} |
| **Input Only** | Clean Context Only (Difficulty Baseline) | **{auroc_io:.3f}** | {m_io.get('brier_score', 0):.3f} | {m_io.get('mean_confidence', 0):.1f}% | {comp_io_rate*100:.1f}% | N/A |
| **Visible Answer** | Clean Context + Frozen Target Choice | **{auroc_va:.3f}** | {m_va.get('brier_score', 0):.3f} | {m_va.get('mean_confidence', 0):.1f}% | {comp_va_rate*100:.1f}% | N/A |
| **Reconstruction** | Clean Context + Independent 2-Candidate Solve | **{auroc_rec:.3f}** | {m_rec.get('brier_score', 0):.3f} | {m_rec.get('mean_confidence', 0):.1f}% | {comp_rec_rate*100:.1f}% | N/A |

*Note: In accordance with standard SDT, Meta-$d'$ and $M$-ratio are defined only for the primary agent's own first-order decision distribution (Immediate Self).*

### Descriptive Self-Metacognition on ALL Valid Target Trials (N={n_self_valid})
- **Self AUROC2:** **{f"{m_self_all_trials['auroc2']:.3f}" if m_self_all_trials.get('auroc2') is not None else 'N/A'}**
- **Self Brier Score:** {f"{m_self_all_trials['brier_score']:.3f}" if m_self_all_trials.get('brier_score') is not None else 'N/A'}
- **Self Mean Confidence:** {m_self_all_trials.get('mean_confidence', 0):.1f}%
- **Self Meta-$d'$ Status:** `{m_self_all_trials.get('meta_d_status')}`
- **Self Meta-$d'$ / $M$-Ratio:** {f"{m_self_all_trials.get('meta_d_prime', 0):.2f} ({m_self_all_trials.get('m_ratio', 0):.2f})" if m_self_all_trials.get('meta_d_prime') is not None else 'N/A'}


---

## 3. Privileged Access Index (PAI) & Contrast Hypotheses

$$\\text{{PAI}} = \\text{{AUROC2}}(\\text{{Self}}) - \\max\\left(\\text{{AUROC2}}_{{\\text{{Input}}}}, \\text{{AUROC2}}_{{\\text{{Visible}}}}, \\text{{AUROC2}}_{{\\text{{Reconstruct}}}}\\right)$$

- **Point Estimate PAI:** **{point_pai:+.3f}**  
- **95% Stratified Bootstrap CI:** [{pai_ci[0]:+.3f}, {pai_ci[1]:+.3f}]  
- **Strongest External Comparator:** {max_external_auroc:.3f}  
- **Preregistered SESOI ($> +0.05$):** {sesoi_text}  
- **Secondary Benchmark ($> +0.10$):** {'PASSED' if summary['privileged_access_index']['historical_sesoi_010_passed'] else 'SESOI +0.10 not met'}  

### Pairwise Observer Contrasts
- **Self vs Input Only:** $\\Delta = {summary['privileged_access_index']['pairwise_contrasts']['self_minus_input_only']['point_estimate']:+.3f}$ [95% CI: {ci_io[0]:+.3f}, {ci_io[1]:+.3f}]
- **Self vs Visible Answer:** $\\Delta = {summary['privileged_access_index']['pairwise_contrasts']['self_minus_visible_answer']['point_estimate']:+.3f}$ [95% CI: {ci_va[0]:+.3f}, {ci_va[1]:+.3f}]
- **Self vs Reconstruction:** $\\Delta = {summary['privileged_access_index']['pairwise_contrasts']['self_minus_reconstruction']['point_estimate']:+.3f}$ [95% CI: {ci_rec[0]:+.3f}, {ci_rec[1]:+.3f}]
"""

    with open(run_dir / "report.md", "w") as f:
        f.write(report_md)

    print(f"\nE02d.1 observer repair complete! Report: {run_dir / 'report.md'}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run H0-v2 E02d.1 frozen-target observer repair (N=200).")
    parser.add_argument("--model", type=str, required=True, help="Ollama model name (e.g. qwen2.5:14b)")
    parser.add_argument("--hop-depth", type=int, required=True, help="Frozen hop depth H")
    parser.add_argument("--distractor-count", type=int, required=True, help="Frozen distractor count D")
    parser.add_argument("--frozen-trials", type=str, required=True, help="Path to frozen trials.jsonl")
    parser.add_argument("--trials", type=int, default=200, help="Total fresh trials (default: 200)")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for item generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature")
    args = parser.parse_args()

    run_e02d1_repaired_observers(
        model_name=args.model,
        hop_depth=args.hop_depth,
        distractor_count=args.distractor_count,
        frozen_trials_path=args.frozen_trials,
        n_trials=args.trials,
        seed=args.seed,
        temperature=args.temperature,
    )
