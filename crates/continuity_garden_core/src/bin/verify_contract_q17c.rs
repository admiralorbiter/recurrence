//! Independent Deterministic Contract Acceptance Verifier for CONTRACT-E-Q17C
//! Recomputes exact statistics directly from per-seed raw telemetry:
//! - Asserts frozen architecture dimension d=128
//! - Recomputes exact sign-flip permutation tests for continuous reset deltas and donor-aligned swap deltas
//! - Recomputes reset near-chance behavior floor
//! - Recomputes same-history twin stability, competence preservation, McNemar shuffle superiority, and zero-sidecar invariants.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawSeedEvaluationQ17C {
    pub seed: u64,
    pub g1_conflict_score_forward: f32,
    pub g1_conflict_score_reverse: f32,
    pub g1_conflict_margin: f32,
    pub g1_passed: bool,
    pub g2_laundered_agreement_score: f32,
    pub g2_unlaundered_baseline_score: f32,
    pub g2_laundering_discrimination_margin: f32,
    pub g2_passed: bool,
    pub g3_m_persistent: f32,
    pub g3_m_reset: f32,
    pub g3_delta_reset: f32,
    pub g3_reset_accuracy: f32,
    pub g3_reset_near_chance: bool,
    pub g3_passed: bool,
    pub g4_m_h1_own: f32,
    pub g4_m_h1_donor_h2: f32,
    pub g4_m_h2_own: f32,
    pub g4_m_h2_donor_h1: f32,
    pub g4_delta_swap_aligned: f32,
    pub g4_h1_transfer_passed: bool,
    pub g4_h2_transfer_passed: bool,
    pub g4_passed: bool,
    pub g5_m_h1_twin_a: f32,
    pub g5_m_h1_twin_b: f32,
    pub g5_twin_delta: f32,
    pub g5_passed: bool,
    pub g6_sensor_accuracy_baseline: f32,
    pub g6_sensor_accuracy_after_swap: f32,
    pub g6_passed: bool,
    pub g7_m_shuffled: f32,
    pub g7_shuffled_passed: bool,
    pub g7_passed: bool,
    pub g8_zero_sidecar_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q17CResultsSummary {
    pub contract_id: String,
    pub hidden_dim: usize,
    pub total_seeds: usize,
    pub raw_seed_results: Vec<RawSeedEvaluationQ17C>,
    pub g1_passed_count: usize,
    pub g1_passed_rate: f32,
    pub g2_passed_count: usize,
    pub g2_passed_rate: f32,
    pub g3_continuous_deltas_reset: Vec<f32>,
    pub g3_permutation_p_val: f64,
    pub g3_reset_near_chance_count: usize,
    pub g3_passed_count: usize,
    pub g4_donor_aligned_swap_deltas: Vec<f32>,
    pub g4_permutation_p_val: f64,
    pub g4_swap_transfer_count: usize,
    pub g4_swap_transfer_rate: f32,
    pub g5_same_history_stable_count: usize,
    pub g5_same_history_stable_rate: f32,
    pub g6_competence_preservation_count: usize,
    pub g6_competence_preservation_rate: f32,
    pub g7_intact_pass_count: usize,
    pub g7_shuffled_pass_count: usize,
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

pub fn exact_mcnemar_p_val(b: usize, c: usize) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    let mut p = 0.0f64;
    let mut binom = 1.0f64;
    for i in 0..=k {
        if i > 0 {
            binom = binom * (n - i + 1) as f64 / i as f64;
        }
        p += binom * 0.5f64.powi(n as i32);
    }
    (p * 2.0).min(1.0)
}

fn main() {
    println!("================================================================================");
    println!("DETERMINISTIC CONTRACT ACCEPTANCE VERIFIER: CONTRACT-E-Q17C (CONFIRMATORY)");
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

    // 1. Contract ID & Architecture Invariant Checks
    if summary.contract_id != "CONTRACT-E-Q17C" {
        violations.push(format!("Contract ID mismatch: expected CONTRACT-E-Q17C, got {}", summary.contract_id));
    }
    if summary.hidden_dim != 128 {
        violations.push(format!("Frozen architecture dimension must be d=128, got d={}", summary.hidden_dim));
    }
    if summary.total_seeds != 16 {
        violations.push(format!("Total evaluated seeds must be exactly 16, got {}", summary.total_seeds));
    }
    if summary.raw_seed_results.len() != 16 {
        violations.push(format!("Raw seed record count must be 16, got {}", summary.raw_seed_results.len()));
    }

    // 2. Independently Recompute Gate 1: Endogenous 2-Hop Conflict (Floor >= 10/16)
    let recomputed_g1_count = summary.raw_seed_results.iter().filter(|r| r.g1_conflict_score_forward > 0.0 && r.g1_conflict_score_reverse < 0.0).count();
    if recomputed_g1_count < 10 {
        violations.push(format!("Gate 1 (Zero-Shot Conflict) floor is >= 10/16, recomputed {}/16", recomputed_g1_count));
    }

    // 3. Independently Recompute Gate 2: Genuine Laundering Discrimination (Floor >= 10/16)
    let recomputed_g2_count = summary.raw_seed_results.iter().filter(|r| r.g2_laundered_agreement_score > 0.0 && r.g2_laundered_agreement_score > r.g2_unlaundered_baseline_score).count();
    if recomputed_g2_count < 10 {
        violations.push(format!("Gate 2 (Laundering Discrimination) floor is >= 10/16, recomputed {}/16", recomputed_g2_count));
    }

    // 4. Independently Recompute Gate 3: Continuous Latent Reset Lesion Effect (Exact Paired Permutation p < 0.01) & Near-Chance behavior
    let recomputed_reset_deltas: Vec<f32> = summary.raw_seed_results.iter().map(|r| r.g3_m_persistent - r.g3_m_reset).collect();
    let recomputed_p_reset = exact_paired_sign_flip_p_val(&recomputed_reset_deltas);
    if recomputed_p_reset >= 0.01 {
        violations.push(format!("Gate 3 (Reset Lesion Permutation) p-val must be < 0.01, recomputed {:.4e}", recomputed_p_reset));
    }
    let recomputed_reset_near_chance = summary.raw_seed_results.iter().filter(|r| r.g3_m_reset.abs() < 0.35).count();
    if recomputed_reset_near_chance < 12 {
        violations.push(format!("Gate 3 (Reset Near-Chance Floor) requires >= 12/16 seeds near chance, recomputed {}/16", recomputed_reset_near_chance));
    }

    // 5. Independently Recompute Gate 4: Continuous Donor-Aligned State Swap Effect (Transfer >= 12/16, Permutation p < 0.01)
    let recomputed_swap_transfer_count = summary.raw_seed_results.iter().filter(|r| r.g4_m_h1_donor_h2 < 0.0 && r.g4_m_h2_donor_h1 > 0.0).count();
    if recomputed_swap_transfer_count < 12 {
        violations.push(format!("Gate 4 (State Swap Transfer) floor is >= 12/16, recomputed {}/16", recomputed_swap_transfer_count));
    }
    let recomputed_swap_deltas: Vec<f32> = summary.raw_seed_results.iter().map(|r| {
        let s_h1 = 1.0f32;
        let s_h2 = -1.0f32;
        0.5 * (s_h2 * (r.g4_m_h1_donor_h2 - r.g4_m_h1_own) + s_h1 * (r.g4_m_h2_donor_h1 - r.g4_m_h2_own))
    }).collect();
    let recomputed_p_swap = exact_paired_sign_flip_p_val(&recomputed_swap_deltas);
    if recomputed_p_swap >= 0.01 {
        violations.push(format!("Gate 4 (Donor-Aligned Swap Permutation) p-val must be < 0.01, recomputed {:.4e}", recomputed_p_swap));
    }

    // 6. Independently Recompute Gate 5: Same-History Twin Stability (Floor >= 15/16)
    let recomputed_g5_count = summary.raw_seed_results.iter().filter(|r| (r.g5_m_h1_twin_b - r.g5_m_h1_twin_a).abs() < 0.15).count();
    if recomputed_g5_count < 15 {
        violations.push(format!("Gate 5 (Same-History Swap Stability) floor is >= 15/16, recomputed {}/16", recomputed_g5_count));
    }

    // 7. Independently Recompute Gate 6: First-Order Sensor Task Competence (Floor >= 15/16 at >= 90%)
    let recomputed_g6_count = summary.raw_seed_results.iter().filter(|r| r.g6_sensor_accuracy_baseline >= 0.90 && r.g6_sensor_accuracy_after_swap >= 0.90).count();
    if recomputed_g6_count < 15 {
        violations.push(format!("Gate 6 (Competence Preservation) floor is >= 15/16, recomputed {}/16", recomputed_g6_count));
    }

    // 8. Independently Recompute Gate 7: Genuine Temporal Shuffle Superiority (McNemar Delta >= 3, p < 0.05)
    let n10 = summary.raw_seed_results.iter().filter(|r| r.g1_passed && !r.g7_shuffled_passed).count();
    let n01 = summary.raw_seed_results.iter().filter(|r| !r.g1_passed && r.g7_shuffled_passed).count();
    let mcnemar_delta = n10 as i32 - n01 as i32;
    let mcnemar_p = exact_mcnemar_p_val(n10, n01);
    if mcnemar_delta < 3 || mcnemar_p >= 0.05 {
        violations.push(format!("Gate 7 (Temporal Shuffle Superiority) Delta must be >= 3 and p < 0.05, recomputed Delta={}, p={:.4e}", mcnemar_delta, mcnemar_p));
    }

    // 9. Independently Recompute Gate 8: Structural Zero-Sidecar Invariant (16/16)
    let recomputed_g8_count = summary.raw_seed_results.iter().filter(|r| r.g8_zero_sidecar_verified).count();
    if recomputed_g8_count != 16 {
        violations.push(format!("Gate 8 (Zero-Sidecar Invariant) must be 16/16, recomputed {}/16", recomputed_g8_count));
    }

    if !violations.is_empty() {
        eprintln!("[VERIFIER FAILED] CONTRACT-E-Q17C Acceptance Violations ({}):", violations.len());
        for v in &violations {
            eprintln!("  - [VIOLATION] {}", v);
        }
        std::process::exit(1);
    }

    println!("ALL CONTRACT-E-Q17C FROZEN ACCEPTANCE CRITERIA INDEPENDENTLY RECOMPUTED & CLEANLY SATISFIED:");
    println!("  - Architecture Dimension Invariant:     d={} (PASS)", summary.hidden_dim);
    println!("  - Gate 1 (Endogenous 2-Hop Conflict):   {}/16 seeds (PASS)", recomputed_g1_count);
    println!("  - Gate 2 (Endogenous Laundering):       {}/16 seeds (PASS)", recomputed_g2_count);
    println!("  - Gate 3 (Latent Reset Lesion p-val):   p={:.4e} < 0.01, near-chance {}/16 (PASS)", recomputed_p_reset, recomputed_reset_near_chance);
    println!("  - Gate 4 (Donor-Aligned Swap Effect):   {}/16 transferred, p={:.4e} < 0.01 (PASS)", recomputed_swap_transfer_count, recomputed_p_swap);
    println!("  - Gate 5 (Same-History Swap Stability): {}/16 stable (PASS)", recomputed_g5_count);
    println!("  - Gate 6 (First-Order Competence):      {}/16 preserved at >=90% (PASS)", recomputed_g6_count);
    println!("  - Gate 7 (Temporal Shuffle Superiority): Delta=+{}, p={:.4e} (PASS)", mcnemar_delta, mcnemar_p);
    println!("  - Gate 8 (Structural Zero-Sidecar):     16/16 verified (PASS)");
    println!("================================================================================");
}
