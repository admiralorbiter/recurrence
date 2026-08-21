//! Independent Deterministic Contract Acceptance Verifier for CONTRACT-E-Q17C
//! Recomputes exact sign-flip permutation tests for continuous reset deltas and donor-aligned swap deltas,
//! asserts same-history swap stability, competence preservation, and structural zero-sidecar API enforcement.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q17CResultsSummary {
    pub contract_id: String,
    pub total_seeds: usize,
    pub g1_passed_count: usize,
    pub g1_passed_rate: f32,
    pub g2_passed_count: usize,
    pub g2_passed_rate: f32,
    pub g3_continuous_deltas_reset: Vec<f32>,
    pub g3_permutation_p_val: f64,
    pub g3_passed_count: usize,
    pub g4_donor_aligned_swap_deltas: Vec<f32>,
    pub g4_permutation_p_val: f64,
    pub g4_swap_transfer_count: usize,
    pub g4_swap_transfer_rate: f32,
    pub g5_same_history_stable_count: usize,
    pub g5_same_history_stable_rate: f32,
    pub g6_competence_preservation_count: usize,
    pub g6_competence_preservation_rate: f32,
    pub g7_mcnemar_n10: usize,
    pub g7_mcnemar_n01: usize,
    pub g7_mcnemar_delta: i32,
    pub g7_mcnemar_p_val: f64,
    pub g8_zero_sidecar_count: usize,
    pub all_criteria_passed: bool,
}

pub fn exact_paired_sign_flip_p_val(deltas: &[f32]) -> f64 {
    let n = deltas.len();
    if n == 0 {
        return 1.0;
    }
    let observed_t: f32 = deltas.iter().sum();
    let num_comb = 1usize << n;
    let mut extreme_count = 0usize;

    for mask in 0..num_comb {
        let mut sim_t = 0.0f32;
        for i in 0..n {
            let sign = if (mask & (1 << i)) != 0 { 1.0 } else { -1.0 };
            sim_t += sign * deltas[i];
        }
        if sim_t >= observed_t - 1e-6 {
            extreme_count += 1;
        }
    }
    extreme_count as f64 / num_comb as f64
}

fn main() {
    println!("================================================================================");
    println!("DETERMINISTIC CONTRACT ACCEPTANCE VERIFIER: CONTRACT-E-Q17C");
    println!("================================================================================\n");

    let results_path = Path::new("crates/continuity_garden_core/data/q17c_endogenous_results.json");
    if !results_path.exists() {
        eprintln!("[VERIFIER ERROR] Results file not found: {:?}", results_path);
        std::process::exit(1);
    }

    let file = File::open(results_path).expect("Failed to open results JSON");
    let reader = BufReader::new(file);
    let summary: Q17CResultsSummary = serde_json::from_reader(reader).expect("Failed to parse results JSON");

    let mut violations: Vec<String> = Vec::new();

    // 1. Contract ID Check
    if summary.contract_id != "CONTRACT-E-Q17C" {
        violations.push(format!("Contract ID mismatch: expected CONTRACT-E-Q17C, got {}", summary.contract_id));
    }

    // 2. Total Seeds Check
    if summary.total_seeds != 16 {
        violations.push(format!("Total evaluated seeds must be exactly 16, got {}", summary.total_seeds));
    }

    // 3. Gate 1: Endogenous Two-Hop Conflict (Floor >= 10/16)
    if summary.g1_passed_count < 10 {
        violations.push(format!("Gate 1 (Zero-Shot Conflict) floor is >= 10/16, got {}/16", summary.g1_passed_count));
    }

    // 4. Gate 2: Endogenous Laundering Discrimination (Floor >= 10/16)
    if summary.g2_passed_count < 10 {
        violations.push(format!("Gate 2 (Laundering Discrimination) floor is >= 10/16, got {}/16", summary.g2_passed_count));
    }

    // 5. Gate 3: Continuous Latent Reset Lesion Effect (Exact Paired Permutation p < 0.01)
    let recomputed_p_reset = exact_paired_sign_flip_p_val(&summary.g3_continuous_deltas_reset);
    if recomputed_p_reset >= 0.01 {
        violations.push(format!("Gate 3 (Reset Lesion Permutation) p-val must be < 0.01, recomputed {:.4e}", recomputed_p_reset));
    }

    // 6. Gate 4: Continuous Donor-Aligned State Swap Effect (Transfer >= 12/16, Permutation p < 0.01)
    if summary.g4_swap_transfer_count < 12 {
        violations.push(format!("Gate 4 (State Swap Transfer) floor is >= 12/16, got {}/16", summary.g4_swap_transfer_count));
    }
    let recomputed_p_swap = exact_paired_sign_flip_p_val(&summary.g4_donor_aligned_swap_deltas);
    if recomputed_p_swap >= 0.01 {
        violations.push(format!("Gate 4 (Donor-Aligned Swap Permutation) p-val must be < 0.01, recomputed {:.4e}", recomputed_p_swap));
    }

    // 7. Gate 5: Same-History Swap Stability (Floor >= 15/16)
    if summary.g5_same_history_stable_count < 15 {
        violations.push(format!("Gate 5 (Same-History Swap Stability) floor is >= 15/16, got {}/16", summary.g5_same_history_stable_count));
    }

    // 8. Gate 6: First-Order Competence Preservation (Floor >= 15/16)
    if summary.g6_competence_preservation_count < 15 {
        violations.push(format!("Gate 6 (Competence Preservation) floor is >= 15/16, got {}/16", summary.g6_competence_preservation_count));
    }

    // 9. Gate 7: Temporal Shuffle Superiority (Delta >= 3, p < 0.05)
    if summary.g7_mcnemar_delta < 3 || summary.g7_mcnemar_p_val >= 0.05 {
        violations.push(format!("Gate 7 (Temporal Shuffle Superiority) Delta must be >= 3 and p < 0.05, got Delta={}, p={:.4e}", summary.g7_mcnemar_delta, summary.g7_mcnemar_p_val));
    }

    // 10. Gate 8: Structural Zero-Sidecar Invariant (16/16)
    if summary.g8_zero_sidecar_count != 16 {
        violations.push(format!("Gate 8 (Zero-Sidecar Invariant) must be 16/16, got {}/16", summary.g8_zero_sidecar_count));
    }

    if !violations.is_empty() {
        eprintln!("[VERIFIER FAILED] CONTRACT-E-Q17C Acceptance Violations ({}):", violations.len());
        for v in &violations {
            eprintln!("  - [VIOLATION] {}", v);
        }
        std::process::exit(1);
    }

    println!("ALL CONTRACT-E-Q17C FROZEN ACCEPTANCE CRITERIA CLEANLY SATISFIED:");
    println!("  - Gate 1 (Endogenous 2-Hop Conflict):   {}/16 seeds (PASS)", summary.g1_passed_count);
    println!("  - Gate 2 (Endogenous Laundering):       {}/16 seeds (PASS)", summary.g2_passed_count);
    println!("  - Gate 3 (Latent Reset Lesion p-val):   p={:.4e} < 0.01 (PASS)", recomputed_p_reset);
    println!("  - Gate 4 (Donor-Aligned Swap Effect):   {}/16 transferred, p={:.4e} < 0.01 (PASS)", summary.g4_swap_transfer_count, recomputed_p_swap);
    println!("  - Gate 5 (Same-History Swap Stability): {}/16 stable (PASS)", summary.g5_same_history_stable_count);
    println!("  - Gate 6 (First-Order Competence):      {}/16 preserved (PASS)", summary.g6_competence_preservation_count);
    println!("  - Gate 7 (Temporal Shuffle Superiority): Delta=+{}, p={:.4e} (PASS)", summary.g7_mcnemar_delta, summary.g7_mcnemar_p_val);
    println!("  - Gate 8 (Structural Zero-Sidecar):     16/16 verified (PASS)");
    println!("================================================================================");
}
