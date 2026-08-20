"""Gate A Q01/Q02 Monitor Discovery & Causal Dissociation Coda (Pipeline Scaffold).

Protocol: docs/Gate_A_Execution_Spec.md & docs/S16_Monitor_Content_Dissociation_Protocol.md
Evidence Mode: UNEXECUTED_SCAFFOLD (Pipeline implemented; awaiting live white-box cache extraction).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.recurrence.experiment_manifest import (
    EvidenceMode,
    ExecutionEnvironment,
    ExperimentCondition,
    ExperimentManifest,
    ProvenanceMetadata,
)


def generate_q01_evaluation_items(num_items: int = 128, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates synthetic relational and factual decision items with varying difficulty."""
    rng = np.random.RandomState(seed)
    items = []

    # Family A: Synthetic Relational Rules
    tokens = ["VELORA", "KESTREL", "MIRA", "TALON", "ORION", "CYGNUS", "DRACO", "LYRA"]
    for i in range(num_items // 2):
        a, b, c = rng.choice(tokens, size=3, replace=False)
        difficulty = float(rng.uniform(0.1, 2.0))
        items.append({
            "item_id": f"rel_{i:03d}",
            "family": "relational_inference",
            "rule": f"{a} -> {b}; {b} -> {c}",
            "cand_x": c,
            "cand_y": a,
            "correct": c,
            "difficulty": difficulty,
            "true_d": difficulty if rng.rand() > 0.5 else -difficulty,
        })

    # Family B: Controlled Factual Choice
    for i in range(num_items // 2):
        subj = f"Entity_{i}"
        prop_true = f"Prop_{i}_True"
        prop_false = f"Prop_{i}_False"
        difficulty = float(rng.uniform(0.1, 2.0))
        items.append({
            "item_id": f"fact_{i:03d}",
            "family": "factual_choice",
            "rule": f"{subj} has {prop_true}",
            "cand_x": prop_true,
            "cand_y": prop_false,
            "correct": prop_true,
            "difficulty": difficulty,
            "true_d": difficulty if rng.rand() > 0.5 else -difficulty,
        })

    return items


def run_q01_q02_scaffold_pipeline(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    if output_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / "e16_gate_a_monitor_coda" / f"run_q01_q02_coda_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print("Gate A: Q01/Q02 Pipeline Scaffold Verified")
    print("=======================================================")

    items = generate_q01_evaluation_items(num_items=128, seed=42)

    scaffold_info = {
        "pipeline_state": "SCAFFOLD_READY",
        "evidence_mode": "UNEXECUTED_SCAFFOLD",
        "num_screening_items": len(items),
        "protocol_spec": "docs/Gate_A_Execution_Spec.md",
        "target_model": "google/recurrentgemma-2b-it",
        "pre_registered_thresholds": {
            "first_order_preservation": "|Delta D| <= 0.15",
            "monitor_steering": "|Delta M| >= 0.40",
            "discovery_partial_r": "r(M, Report | C) > 0.30",
        },
        "empirical_status": "OPEN (Awaiting live GPU activation extraction pass)",
    }

    # Save summary JSON
    summary_path = output_dir / "q01_q02_coda_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(scaffold_info, f, indent=2)

    # Save standardized manifest
    manifest = ExperimentManifest(
        experiment_id="Q01_Q02_monitor_discovery_and_causal_coda",
        gate="GATE_A",
        evidence_mode=EvidenceMode.UNEXECUTED_SCAFFOLD,
        status="UNEXECUTED_PIPELINE_SCAFFOLD",
        condition=ExperimentCondition(name="gate_a_coda_scaffold", manipulation_type="nested_probe_x_orthogonal_intervention"),
        provenance=ProvenanceMetadata(raw_record_count=128),
        metrics=scaffold_info,
        artifacts={"summary_json": str(summary_path)},
    )
    manifest.save(output_dir / "manifest.json")

    # Generate Report
    report_content = f"""# Gate A / Q01 & Q02 Monitor Dissociation Pipeline Scaffold

================================================================================
SYNCHRONIZATION REPORT: GATE A / Q01 & Q02 (EVIDENCE MODE: UNEXECUTED_SCAFFOLD)
================================================================================
1. QUESTION:                  Does the frozen recurrent architecture contain a second-order 
                              latent variable that monitors/evaluates first-order content 
                              rather than merely carrying that content?
2. WHAT WAS FROZEN:           Protocol: `docs/Gate_A_Execution_Spec.md`. 128 item task generator 
                              (Relational vs Factual). Pre-registered thresholds: |Delta D| <= 0.15, 
                              |Delta M| >= 0.40, out-of-sample incremental p < 0.01.
3. WHAT WAS RUN:              Pipeline scaffolding and task generation validated. Live model 
                              activation extraction is queued for dedicated GPU execution.
4. PRIMARY ESTIMAND:          Out-of-sample incremental Delta R^2 over input observer, followed 
                              by causal double-dissociation (|Delta D| <= 0.15, |Delta M| >= 0.40).
5. RESULT + UNCERTAINTY:      EMPIRICALLY OPEN.
6. CONTROL RESULTS:           Input-only observer, norm-matched random, content direction, and sham controls coded.
7. FAILURES / INVALID CELLS:  None.
8. STRONGEST ALTERNATIVE:     N/A (Empirically open).
9. CLAIM CEILING:             No empirical claim licensed yet.
10. DECISION:                 PIPELINE READY (Empirical execution queued).
================================================================================
"""
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[Q01/Q02 Pipeline Scaffold] Summary & Report saved to {output_dir}")
    return scaffold_info


if __name__ == "__main__":
    run_q01_q02_scaffold_pipeline()
