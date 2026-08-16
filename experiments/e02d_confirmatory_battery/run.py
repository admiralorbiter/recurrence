"""Experiment E02d: Horizon 0 v2 Confirmatory Metacognitive Battery (N=200).

Evaluates the 4-observer metacognitive battery at frozen psychophysically equated
operating coordinates for Qwen 2.5 14B (H=3, D=16) and Qwen 2.5 3B (H=1, D=8).

Observer Ladder:
1. Immediate Self: Same-invocation target choice + confidence.
2. Input Only: Item-difficulty baseline (P(Target Correct) given context only).
3. Visible Answer: P(Target Correct) given context + target's selected value (confidence removed).
4. Reconstruction: Independent solver assigning probabilities to candidate values;
   probability assigned to target-selected value becomes reconstructed P(Target Correct).
   Guardrail: Validates candidate probabilities sum to 100.0 within +/-5% tolerance.

Primary Statistic:
PAI = AUROC2(Self) - max(AUROC2(InputOnly), AUROC2(VisibleAnswer), AUROC2(Reconstruction))
Evaluated via stratified paired bootstrap (B=2000) on shared valid-trial intersection.
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
from recurrence.core.logging import ExperimentLogger
from recurrence.core.schemas import make_2afc_direct_value_schema
from recurrence.tasks.adaptive_metacognition import (
    AdaptiveMetacognition2AFCTask,
)
from recurrence.analysis.psychophysics import (
    compute_sdt_indices,
    compute_sdt_bootstrap_ci,
    compute_wilson_score_interval,
    compute_type2_sdt_metrics,
    evaluate_calibration_gate,
)
from recurrence.analysis.meta_d import fit_meta_d_mle


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
                "description": "Estimated probability that the target choice is correct (0 to 100).",
            },
            "rationale": {"type": "string"},
        },
        "required": ["probability"],
        "additionalProperties": False,
    }


def run_confirmatory_battery(
    model_name: str,
    hop_depth: int,
    distractor_count: int,
    n_trials: int = 200,
    seed: int = 1000,
    temperature: float = 0.0,
    output_dir: str = "results/e02d_confirmatory_battery",
    backend_override: Optional[Any] = None,
) -> Path:
    """Run the 4-observer confirmatory metacognitive battery at frozen coordinate."""
    start_time = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = model_name.replace(":", "_").replace(".", "_")
    run_id = f"run_e02d_{model_clean}_{timestamp}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EXPERIMENT E02d: CONFIRMATORY METACOGNITIVE BATTERY (N={n_trials})")
    print(f"Model: {model_name} | Coordinate: (H={hop_depth}, D={distractor_count}) | Seed: {seed}")
    print(f"{'='*60}\n")

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

    # 1. Generate N fresh held-out items
    items = task.generate_multi_hop_grid(
        grid_cells=[(hop_depth, distractor_count)],
        count_per_cell=n_trials,
        base_seed=seed,
        ask_confidence=True,
    )

    print(f"Generated {len(items)} held-out items at (H={hop_depth}, D={distractor_count}). Running 4-observer ladder...\n")

    trial_records: List[Dict[str, Any]] = []

    for idx, it in enumerate(items, 1):
        opt_map = it.metadata.get("option_map", {})
        c1 = opt_map.get("A") or it.metadata.get("candidate_1_value") or it.metadata.get("option_a", "")
        c2 = opt_map.get("B") or it.metadata.get("candidate_2_value") or it.metadata.get("option_b", "")
        schema_self = make_2afc_direct_value_schema(c1, c2, ask_confidence=True)


        # -------------------------------------------------------------
        # STEP 1: Immediate Self (Target Model Choice + Confidence)
        # -------------------------------------------------------------
        t0 = time.time()
        resp_self = backend.generate_with_schema(prompt=it.prompt, json_schema=schema_self)
        lat_self = time.time() - t0

        compliance_self = False
        selected_value = None
        self_raw_conf = None

        if resp_self.parsed is not None and "answer" in resp_self.parsed and "probability" in resp_self.parsed:
            ans = resp_self.parsed["answer"]
            prob = resp_self.parsed["probability"]
            if ans in [c1, c2] and prob is not None:
                try:
                    prob_val = float(prob)
                    if 0.0 <= prob_val <= 100.0:
                        compliance_self = True
                        selected_value = ans
                        self_raw_conf = prob_val
                except (ValueError, TypeError):
                    compliance_self = False

        # Ground truth: Option A (1) or Option B (2)
        gt_letter = it.ground_truth
        gt_value = c1 if gt_letter == "A" else c2
        is_correct = bool(selected_value == gt_value) if selected_value else False
        chose_letter = "A" if selected_value == c1 else ("B" if selected_value == c2 else None)

        # -------------------------------------------------------------
        # STEP 2: Input-Only Observer (Context only, no target choice)
        # -------------------------------------------------------------
        prompt_input_only = (
            f"{it.prompt}\n\n"
            f"EVALUATION TASK:\n"
            f"Evaluate the difficulty of the above relational multi-hop retrieval task.\n"
            f"Estimate the probability (0 to 100) that a standard reasoner correctly identifies the target value."
        )
        schema_ext = make_external_evaluator_schema()
        resp_input_only = backend.generate_with_schema(prompt=prompt_input_only, json_schema=schema_ext)
        compliance_io = False
        io_conf = None
        if resp_input_only.parsed is not None and "probability" in resp_input_only.parsed:
            try:
                prob_val = float(resp_input_only.parsed["probability"])
                if 0.0 <= prob_val <= 100.0:
                    compliance_io = True
                    io_conf = prob_val
            except (ValueError, TypeError):
                compliance_io = False

        # -------------------------------------------------------------
        # STEP 3: Visible-Answer Observer (Context + Target Choice, no confidence)
        # -------------------------------------------------------------
        prompt_vis_ans = (
            f"{it.prompt}\n\n"
            f"PROPOSED ANSWER:\n"
            f"The selected candidate value is: \"{selected_value}\"\n\n"
            f"EVALUATION TASK:\n"
            f"Estimate the probability (0 to 100) that this proposed candidate value is the correct answer."
        )
        resp_vis_ans = backend.generate_with_schema(prompt=prompt_vis_ans, json_schema=schema_ext)
        compliance_va = False
        va_conf = None
        if resp_vis_ans.parsed is not None and "probability" in resp_vis_ans.parsed:
            try:
                prob_val = float(resp_vis_ans.parsed["probability"])
                if 0.0 <= prob_val <= 100.0:
                    compliance_va = True
                    va_conf = prob_val
            except (ValueError, TypeError):
                compliance_va = False

        # -------------------------------------------------------------
        # STEP 4: Reconstruction Observer (Independent Solver)
        # -------------------------------------------------------------
        prompt_reconstruct = (
            f"{it.prompt}\n\n"
            f"INDEPENDENT RECONSTRUCTION TASK:\n"
            f"Independently evaluate both candidate values: Candidate 1 = \"{c1}\", Candidate 2 = \"{c2}\".\n"
            f"Assign a probability to each candidate reflecting how likely it is to be the true target value.\n"
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
                # Guardrail: validate sum within +/- 5% tolerance
                if 95.0 <= total_p <= 105.0 and 0.0 <= p1 <= 100.0 and 0.0 <= p2 <= 100.0:
                    compliance_rec = True
                    rec_conf_target = p1 if selected_value == c1 else p2
            except (ValueError, TypeError):
                compliance_rec = False

        rec = {
            "trial_id": it.item_id,
            "hop_depth": hop_depth,
            "distractor_count": distractor_count,
            "ground_truth_letter": gt_letter,
            "ground_truth_value": gt_value,
            "selected_value": selected_value,
            "parsed_answer": chose_letter,
            "ground_truth": gt_letter,
            "correct": is_correct,
            "compliance_self": compliance_self,
            "compliance_input_only": compliance_io,
            "compliance_visible_answer": compliance_va,
            "compliance_reconstruction": compliance_rec,
            "all_observers_compliant": bool(compliance_self and compliance_io and compliance_va and compliance_rec),
            "probability_self": self_raw_conf,
            "probability_input_only": io_conf,
            "probability_visible_answer": va_conf,
            "probability_reconstruction": rec_conf_target,
            "latency_self": lat_self,
        }
        trial_records.append(rec)

        if idx % 10 == 0 or idx == n_trials:
            acc_so_far = np.mean([r["correct"] for r in trial_records]) * 100
            print(f"[{idx}/{n_trials}] Acc: {acc_so_far:.1f}% | Self Conf: {self_raw_conf} | Rec Conf: {rec_conf_target}")

    # Save trials
    trials_df = pd.DataFrame(trial_records)
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

    # 2. First-Order Performance Set A: All Valid Immediate-Self Trials
    self_valid_df = trials_df[trials_df["compliance_self"]].copy()
    n_self_valid = len(self_valid_df)
    recs_self_valid = self_valid_df.to_dict(orient="records")
    sdt_t1_self = compute_sdt_indices(recs_self_valid)
    t1_ci_self = compute_sdt_bootstrap_ci(recs_self_valid, n_bootstraps=1000)
    t1_self_d_ci = (float(t1_ci_self["d_prime_ci_lower"]), float(t1_ci_self["d_prime_ci_upper"]))
    t1_self_c_ci = (float(t1_ci_self["criterion_c_ci_lower"]), float(t1_ci_self["criterion_c_ci_upper"]))

    # 3. First-Order Performance Set B: Shared 4-Observer Valid Intersection
    shared_valid_df = trials_df[trials_df["all_observers_compliant"]].copy()
    n_shared_valid = len(shared_valid_df)
    recs_shared = shared_valid_df.to_dict(orient="records")
    sdt_t1_shared = compute_sdt_indices(recs_shared)
    t1_ci_shared = compute_sdt_bootstrap_ci(recs_shared, n_bootstraps=1000)
    t1_shared_d_ci = (float(t1_ci_shared["d_prime_ci_lower"]), float(t1_ci_shared["d_prime_ci_upper"]))
    t1_shared_c_ci = (float(t1_ci_shared["criterion_c_ci_lower"]), float(t1_ci_shared["criterion_c_ci_upper"]))

    # 4. Type-2 Metrics for Each Condition (on Shared Intersection)
    def get_cond_metrics(prob_col: str) -> Dict[str, Any]:
        cond_recs = [
            {"probability": r[prob_col], "correct": r["correct"], "ground_truth": r["ground_truth"], "parsed_answer": r["parsed_answer"]}
            for r in recs_shared if r[prob_col] is not None
        ]
        return compute_type2_sdt_metrics(cond_recs, fit_meta_d=True, n_bins=4)

    m_self = get_cond_metrics("probability_self")
    m_io = get_cond_metrics("probability_input_only")
    m_va = get_cond_metrics("probability_visible_answer")
    m_rec = get_cond_metrics("probability_reconstruction")

    auroc_self = float(m_self.get("auroc2") if m_self.get("auroc2") is not None else 0.5)
    auroc_io = float(m_io.get("auroc2") if m_io.get("auroc2") is not None else 0.5)
    auroc_va = float(m_va.get("auroc2") if m_va.get("auroc2") is not None else 0.5)
    auroc_rec = float(m_rec.get("auroc2") if m_rec.get("auroc2") is not None else 0.5)

    max_external_auroc = max(auroc_io, auroc_va, auroc_rec)
    point_pai = auroc_self - max_external_auroc

    # 5. Stratified Paired Bootstrapping (B=2000) preserving class counts
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
            # Stratified resampling: draw with replacement separately from correct and incorrect pools
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
        "model_name": model_name,
        "model_digest": model_digest,
        "hop_depth": hop_depth,
        "distractor_count": distractor_count,
        "total_trials": n_trials,
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
        "type2_metrics": {
            "self": m_self,
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
        experiment_id="e02d_confirmatory_battery",
        run_id=run_id,
        seed=seed,
        model_tag=model_name,
        model_digest=model_digest,
        parameters={
            "hop_depth": hop_depth,
            "distractor_count": distractor_count,
            "total_trials": n_trials,
            "valid_trials": n_shared_valid,
            "temperature": temperature,
            "measurement_gate_passed": measurement_gate_passed,
        },
    )
    manifest.compute_environment_hash()
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest.model_dump(), f, indent=2)


    # Generate Markdown Report
    t1_self = summary["type1_metrics_immediate_self"]
    t1_sh = summary["type1_metrics_shared_intersection"]

    sesoi_text = (
        "**PASSED (Privileged Access Supported)**"
        if pai_sesoi_passed
        else "**SESOI threshold not met / no meaningful positive privileged-access advantage resolved.**"
    )

    report_md = f"""# Experiment E02d: Confirmatory Metacognitive Battery Report (N={n_trials})

**Run ID:** `{run_id}`  
**Model:** `{model_name}` (`{model_digest[:12]}...`)  
**Frozen Operating Coordinate:** `(H={hop_depth}, D={distractor_count})`  
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

| Observer Condition | Informational Input | AUROC2 | Brier Score | Mean Conf | Meta-$d'$ Status | Meta-$d'$ / $M$-Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Immediate Self** | Same-invocation target choice + confidence | **{auroc_self:.3f}** | {m_self['brier_score']:.3f} | {m_self.get('mean_confidence', 0):.1f}% | `{m_self.get('meta_d_status')}` | {f"{m_self.get('meta_d_prime', 0):.2f} ({m_self.get('m_ratio', 0):.2f})" if m_self.get('meta_d_prime') is not None else 'N/A'} |
| **Input Only** | Context Only (Difficulty Baseline) | **{auroc_io:.3f}** | {m_io['brier_score']:.3f} | {m_io.get('mean_confidence', 0):.1f}% | `{m_io.get('meta_d_status')}` | {f"{m_io.get('meta_d_prime', 0):.2f} ({m_io.get('m_ratio', 0):.2f})" if m_io.get('meta_d_prime') is not None else 'N/A'} |
| **Visible Answer** | Context + Target Choice | **{auroc_va:.3f}** | {m_va['brier_score']:.3f} | {m_va.get('mean_confidence', 0):.1f}% | `{m_va.get('meta_d_status')}` | {f"{m_va.get('meta_d_prime', 0):.2f} ({m_va.get('m_ratio', 0):.2f})" if m_va.get('meta_d_prime') is not None else 'N/A'} |
| **Reconstruction** | Context + Independent Solve | **{auroc_rec:.3f}** | {m_rec['brier_score']:.3f} | {m_rec.get('mean_confidence', 0):.1f}% | `{m_rec.get('meta_d_status')}` | {f"{m_rec.get('meta_d_prime', 0):.2f} ({m_rec.get('m_ratio', 0):.2f})" if m_rec.get('meta_d_prime') is not None else 'N/A'} |

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

    print(f"\nConfirmatory battery complete! Report: {run_dir / 'report.md'}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run H0-v2 confirmatory metacognitive battery (N=200).")
    parser.add_argument("--model", type=str, required=True, help="Ollama model name (e.g. qwen2.5:14b)")
    parser.add_argument("--hop-depth", type=int, required=True, help="Frozen hop depth H")
    parser.add_argument("--distractor-count", type=int, required=True, help="Frozen distractor count D")
    parser.add_argument("--trials", type=int, default=200, help="Total fresh trials (default: 200)")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for item generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature")
    args = parser.parse_args()

    run_confirmatory_battery(
        model_name=args.model,
        hop_depth=args.hop_depth,
        distractor_count=args.distractor_count,
        n_trials=args.trials,
        seed=args.seed,
        temperature=args.temperature,
    )
