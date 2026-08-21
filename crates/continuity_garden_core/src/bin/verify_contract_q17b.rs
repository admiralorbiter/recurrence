//! Deterministic Contract Acceptance Verifier for CONTRACT-E-Q17B
//! Asserts independent dataset target sum matching, continuous lesion permutation test, transposed laundering arm,
//! and directional transposition falsification directly from q17b_summary.json.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::exit;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedOutcomeQ17B {
    pub seed: u64,
    pub gate1_multihop_passed: bool,
    pub gate2_laundering_passed: bool,
    pub gate3_superior_to_shuffled: bool,
    pub gate4_transposition_passed: bool,
    pub gate4_transposition_return: f32,
    pub gate5_transposition_laundering_passed: bool,
    pub gate6_path_break_passed: bool,
    pub gate6_delta_a: f32,
    pub self_sup_multihop_acc: f32,
    pub shuffled_control_multihop_acc: f32,
    pub dataset_intact_target_sum: usize,
    pub dataset_shuffled_target_sum: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q17BSummary {
    pub protocol: String,
    pub total_seeds: usize,
    pub dataset_intact_sample_count: usize,
    pub dataset_shuffled_sample_count: usize,
    pub dataset_intact_target_sum: usize,
    pub dataset_shuffled_target_sum: usize,
    pub matched_control_verified: bool,
    pub gate1_multihop_count: usize,
    pub gate1_passed: bool,
    pub gate2_laundering_count: usize,
    pub gate2_passed: bool,
    pub gate3_n10: usize,
    pub gate3_n01: usize,
    pub gate3_delta: i32,
    pub gate3_p_value: f64,
    pub gate3_passed: bool,
    pub gate4_transposition_passed_count: usize,
    pub gate4_transposition_mean_return: f32,
    pub gate4_passed: bool,
    pub gate5_transposition_laundering_count: usize,
    pub gate5_passed: bool,
    pub gate6_p_value: f64,
    pub gate6_passed: bool,
    pub all_gates_passed: bool,
    pub seed_outcomes: Vec<SeedOutcomeQ17B>,
}

fn exact_sign_flip_p_value(diffs: &[f64]) -> f64 {
    let n = diffs.len();
    if n == 0 {
        return 1.0;
    }
    let observed_stat: f64 = diffs.iter().sum();
    let total_combinations = 1usize << n;
    let mut extreme_count = 0usize;

    for mask in 0..total_combinations {
        let mut sim_stat = 0.0f64;
        for i in 0..n {
            let sign = if (mask & (1 << i)) != 0 { 1.0 } else { -1.0 };
            sim_stat += sign * diffs[i].abs();
        }
        if sim_stat >= observed_stat - 1e-12 {
            extreme_count += 1;
        }
    }

    (extreme_count as f64) / (total_combinations as f64)
}

fn main() {
    println!("================================================================================");
    println!("DETERMINISTIC CONTRACT ACCEPTANCE VERIFIER: CONTRACT-E-Q17B");
    println!("================================================================================");

    let summary_path = Path::new("results/e28_q17b_self_supervised_composition/q17b_summary.json");
    if !summary_path.exists() {
        eprintln!("[FAIL] Summary artifact does not exist at {:?}", summary_path);
        exit(1);
    }

    let mut file = File::open(summary_path).unwrap();
    let mut data = String::new();
    file.read_to_string(&mut data).unwrap();

    let summary: Q17BSummary = match serde_json::from_str(&data) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[FAIL] Malformed JSON in summary artifact: {}", e);
            exit(1);
        }
    };

    let mut violations: Vec<String> = Vec::new();

    // 1. Total Seeds Invariant
    if summary.total_seeds != 16 {
        violations.push(format!("Total seeds must be 16, observed {}", summary.total_seeds));
    }
    if summary.seed_outcomes.len() != 16 {
        violations.push(format!("Seed outcomes count must be 16, observed {}", summary.seed_outcomes.len()));
    }

    // 2. Independent Dataset Target Sum Aggregation
    let recomputed_intact_sum: usize = summary.seed_outcomes.iter().map(|o| o.dataset_intact_target_sum).sum();
    let recomputed_shuffled_sum: usize = summary.seed_outcomes.iter().map(|o| o.dataset_shuffled_target_sum).sum();

    if recomputed_intact_sum != recomputed_shuffled_sum {
        violations.push(format!(
            "Independently recomputed dataset target sums differ: intact ({}) != shuffled ({})",
            recomputed_intact_sum, recomputed_shuffled_sum
        ));
    }
    if summary.dataset_intact_target_sum != recomputed_intact_sum {
        violations.push(format!(
            "Summary intact target sum mismatch: summary ({}) != recomputed ({})",
            summary.dataset_intact_target_sum, recomputed_intact_sum
        ));
    }

    // 3. Gate 1: Zero-Shot Multi-Hop Conflict Floor (>= 10/16)
    if summary.gate1_multihop_count < 10 {
        violations.push(format!("Gate 1 floor >= 10/16 seeds, observed {}/16", summary.gate1_multihop_count));
    }

    // 4. Gate 2: Laundering Discrimination Floor (>= 10/16)
    if summary.gate2_laundering_count < 10 {
        violations.push(format!("Gate 2 floor >= 10/16 seeds, observed {}/16", summary.gate2_laundering_count));
    }

    // 5. Gate 3: Temporal Shuffle Control Superiority (n10 - n01 >= 3, p < 0.05)
    if summary.gate3_delta < 3 {
        violations.push(format!("Gate 3 Delta (n10 - n01) floor >= 3, observed {}", summary.gate3_delta));
    }
    if summary.gate3_p_value >= 0.05 {
        violations.push(format!("Gate 3 p-value floor < 0.05, observed {:.4e}", summary.gate3_p_value));
    }

    // 6. Gate 4: Directional Transposition Falsification (<= 2/16 seeds, return < 0.00)
    if summary.gate4_transposition_passed_count > 2 {
        violations.push(format!("Gate 4 Transposition ceiling <= 2/16 seeds, observed {}/16", summary.gate4_transposition_passed_count));
    }
    if summary.gate4_transposition_mean_return >= 0.0 {
        violations.push(format!("Gate 4 Mean Return ceiling < 0.00, observed {:.3}", summary.gate4_transposition_mean_return));
    }

    // 7. Gate 5: Transposition Laundering Floor (>= 10/16)
    if summary.gate5_transposition_laundering_count < 10 {
        violations.push(format!("Gate 5 floor >= 10/16 seeds, observed {}/16", summary.gate5_transposition_laundering_count));
    }

    // 8. Gate 6: Mechanistic Path-Break Continuous Permutation Test (p < 0.01)
    let continuous_deltas: Vec<f64> = summary.seed_outcomes.iter().map(|o| o.gate6_delta_a as f64).collect();
    let recomputed_g6_p = exact_sign_flip_p_value(&continuous_deltas);
    if recomputed_g6_p >= 0.01 {
        violations.push(format!("Gate 6 continuous permutation p-value floor < 0.01, recomputed {:.4e}", recomputed_g6_p));
    }

    if !violations.is_empty() {
        eprintln!("\nCONTRACT VERIFICATION FAILED ({} violations):", violations.len());
        for v in &violations {
            eprintln!("  - [VIOLATION] {}", v);
        }
        exit(1);
    }

    println!("\nALL CONTRACT-E-Q17B FROZEN ACCEPTANCE CRITERIA CLEANLY SATISFIED:");
    println!("  - Dataset Matched Control Recomputed:      {} samples, {} target sum (EXACT MATCH)", summary.dataset_intact_sample_count, recomputed_intact_sum);
    println!("  - Gate 1 (Zero-Shot Multi-Hop Conflict):   {}/16 seeds (PASS)", summary.gate1_multihop_count);
    println!("  - Gate 2 (Laundering Discrimination):       {}/16 seeds (PASS)", summary.gate2_laundering_count);
    println!("  - Gate 3 (Temporal Shuffle Superiority):    n10={}, n01={}, Delta={}, p={:.4e} (PASS)", summary.gate3_n10, summary.gate3_n01, summary.gate3_delta, summary.gate3_p_value);
    println!("  - Gate 4 (Directional Transposition Fals):  {}/16 passed, return={:.3} (PASS)", summary.gate4_transposition_passed_count, summary.gate4_transposition_mean_return);
    println!("  - Gate 5 (Transposition Laundering Arm):    {}/16 seeds (PASS)", summary.gate5_transposition_laundering_count);
    println!("  - Gate 6 (Continuous Delta Permutation):    p={:.4e} (PASS)", recomputed_g6_p);
    println!("================================================================================");
}
