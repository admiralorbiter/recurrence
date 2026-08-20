"""Q03 Sidecar: Base vs. Instruction-Tuned Metacognition Transport Scout (Pipeline Scaffold).

Protocol: docs/Gate_A_Execution_Spec.md
Evidence Mode: UNEXECUTED_SCAFFOLD (Pipeline implemented; awaiting live model inference).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

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


def run_q03_scaffold(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    if output_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / "e16_gate_a_monitor_coda" / f"run_q03_base_vs_it_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print("Gate A: Q03 Base vs IT Scout Pipeline Scaffold Verified")
    print("=======================================================")

    scaffold_info = {
        "pipeline_state": "SCAFFOLD_READY",
        "evidence_mode": "UNEXECUTED_SCAFFOLD",
        "protocol_spec": "docs/Gate_A_Execution_Spec.md",
        "base_model": "google/recurrentgemma-2b",
        "it_model": "google/recurrentgemma-2b-it",
        "gates": [
            "Q03.1: Check if computational disagreement transports (D_T * D_O < 0) on base weights",
            "Q03.2: Check if reporting interface passes visible BOP controls (>= 90% accuracy)",
            "Q03.3: Compare PAI_aligned(base) vs PAI_aligned(IT) only if Q03.1 and Q03.2 pass",
        ],
        "empirical_status": "OPEN (Awaiting live GPU inference pass on base checkpoint)",
    }

    summary_path = output_dir / "q03_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(scaffold_info, f, indent=2)

    manifest = ExperimentManifest(
        experiment_id="Q03_base_vs_it_transport_scout",
        gate="GATE_A",
        evidence_mode=EvidenceMode.UNEXECUTED_SCAFFOLD,
        status="UNEXECUTED_PIPELINE_SCAFFOLD",
        condition=ExperimentCondition(name="base_vs_it_comparison", manipulation_type="model_alignment_contrast"),
        provenance=ProvenanceMetadata(raw_record_count=16),
        metrics=scaffold_info,
        artifacts={"summary_json": str(summary_path)},
    )
    manifest.save(output_dir / "manifest.json")

    report_content = f"""# Gate A / Q03 Base vs. IT Transport Scout Pipeline Scaffold

================================================================================
SYNCHRONIZATION REPORT: GATE A / Q03 (EVIDENCE MODE: UNEXECUTED_SCAFFOLD)
================================================================================
1. QUESTION:                  Is the narrow S14 state-conditioned report modulation intrinsic 
                              to pretraining or installed/amplified by instruction tuning?
2. WHAT WAS FROZEN:           Protocol: `docs/Gate_A_Execution_Spec.md`. Models: `google/recurrentgemma-2b` 
                              (base) vs `google/recurrentgemma-2b-it`.
3. WHAT WAS RUN:              Pipeline scaffolding implemented. Live model inference queued.
4. PRIMARY ESTIMAND:          PAI_aligned(base) vs PAI_aligned(IT) conditional on C-level and R-level passes.
5. RESULT + UNCERTAINTY:      EMPIRICALLY OPEN.
6. CONTROL RESULTS:           Visible BOP reporting controls and strict-C panel coded.
7. FAILURES / INVALID CELLS:  None.
8. STRONGEST ALTERNATIVE:     N/A (Empirically open).
9. CLAIM CEILING:             No empirical claim licensed yet.
10. DECISION:                 PIPELINE READY (Empirical execution queued).
================================================================================
"""
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[Q03 Pipeline Scaffold] Summary & Report saved to {output_dir}")
    return scaffold_info


if __name__ == "__main__":
    run_q03_scaffold()
