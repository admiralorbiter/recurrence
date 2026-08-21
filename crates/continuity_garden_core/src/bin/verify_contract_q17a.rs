//! Deterministic Contract Acceptance Verifier for CONTRACT-E-Q17A-R1
//! Reads persisted summary artifact and asserts every pre-registered frozen gate.

use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::process;

#[derive(Debug, Deserialize)]
pub struct Q17aSeedAudit {
    pub seed: u64,
    pub gate1_conflict_acc: f32,
    pub gate1_conflict_return: f32,
    pub gate2_laundering_acc: f32,
    pub gate2_laundering_return: f32,
    pub gate3_corrob_acc: f32,
    pub gate4_conflict_ind_acc: f32,
    pub gate5_n10: usize,
    pub gate5_n01: usize,
    pub gate6_delta_a: f32,
    pub transposition_conflict_acc: f32,
    pub transposition_conflict_return: f32,
    pub transposition_laundering_acc: f32,
}

#[derive(Debug, Deserialize)]
pub struct Q17aSummaryReport {
    pub protocol: String,
    pub total_seeds: usize,
    pub gate1_passed_count: usize,
    pub gate2_passed_count: usize,
    pub gate3_passed_count: usize,
    pub gate4_passed_count: usize,
    pub gate5_passed_count: usize,
    pub gate6_passed_count: usize,
    pub transposition_passed_count: usize,
    pub transposition_mean_return: f32,
    pub transposition_laundering_passed_count: usize,
    pub all_gates_passed_count: usize,
    pub mcnemar_p_value_supporting: f64,
    pub permutation_p_value: f64,
    pub seed_audits: Vec<Q17aSeedAudit>,
}

fn main() {
    println!("================================================================================");
    println!("DETERMINISTIC CONTRACT ACCEPTANCE VERIFIER: CONTRACT-E-Q17A-R1");
    println!("================================================================================");

    let summary_path = Path::new("results/e27_q17_endogenous_transitivity/q17a_summary.json");
    if !summary_path.exists() {
        eprintln!("FAIL: Results summary artifact does not exist at {:?}", summary_path);
        process::exit(1);
    }

    let content = match fs::read_to_string(summary_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL: Could not read {:?}: {}", summary_path, e);
            process::exit(1);
        }
    };

    let summary: Q17aSummaryReport = match serde_json::from_str(&content) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAIL: JSON deserialization error: {}", e);
            process::exit(1);
        }
    };

    let mut violations = Vec::new();

    if summary.total_seeds != 16 {
        violations.push(format!("Total seeds must be 16, observed {}", summary.total_seeds));
    }
    if summary.gate1_passed_count < 12 {
        violations.push(format!("Gate 1 (Conflict Choice) floor >= 12/16, observed {}/16", summary.gate1_passed_count));
    }
    if summary.gate2_passed_count < 11 {
        violations.push(format!("Gate 2 (Laundering Discrimination) floor >= 11/16, observed {}/16", summary.gate2_passed_count));
    }
    if summary.gate3_passed_count < 15 {
        violations.push(format!("Gate 3 (Independent Corroboration) floor >= 15/16, observed {}/16", summary.gate3_passed_count));
    }
    if summary.gate4_passed_count < 15 {
        violations.push(format!("Gate 4 (Independent Conflict) floor >= 15/16, observed {}/16", summary.gate4_passed_count));
    }

    let mut total_n10 = 0;
    let mut total_n01 = 0;
    for audit in &summary.seed_audits {
        total_n10 += audit.gate5_n10;
        total_n01 += audit.gate5_n01;
    }
    let diff = total_n10 as i32 - total_n01 as i32;
    if diff < 3 {
        violations.push(format!("Gate 5 (Composition Ablation Floor) n10 - n01 >= 3, observed diff = {}", diff));
    }

    if summary.permutation_p_value >= 0.01 {
        violations.push(format!("Gate 6 (Exact Permutation Test) p < 0.01, observed p = {:.6e}", summary.permutation_p_value));
    }

    // Transposition Falsification Criteria
    if summary.transposition_passed_count > 2 {
        violations.push(format!("Transposition Conflict Accuracy must collapse to <= 2/16 seeds, observed {}/16 seeds", summary.transposition_passed_count));
    }
    if summary.transposition_mean_return >= 0.00 {
        violations.push(format!("Transposition Conflict Mean Return must be negative (< 0.00), observed {:.3}", summary.transposition_mean_return));
    }
    if summary.transposition_laundering_passed_count < 10 {
        violations.push(format!("Transposition Laundering Verification floor >= 10/16 seeds, observed {}/16 seeds", summary.transposition_laundering_passed_count));
    }

    if !violations.is_empty() {
        eprintln!("\nCONTRACT VERIFICATION FAILED with {} violations:", violations.len());
        for v in &violations {
            eprintln!("  - [VIOLATION] {}", v);
        }
        process::exit(1);
    }

    println!("\nALL CONTRACT-E-Q17A-R1 FROZEN ACCEPTANCE & FALSIFICATION CRITERIA CLEANLY SATISFIED.");
    println!("  - Gate 1: {}/16 seeds", summary.gate1_passed_count);
    println!("  - Gate 2: {}/16 seeds", summary.gate2_passed_count);
    println!("  - Gate 3: {}/16 seeds", summary.gate3_passed_count);
    println!("  - Gate 4: {}/16 seeds", summary.gate4_passed_count);
    println!("  - Gate 5: n10={}, n01={}, diff={}", total_n10, total_n01, diff);
    println!("  - Gate 6: p = {:.6e}", summary.permutation_p_value);
    println!("  - Transposition Collapse: {}/16 seeds, mean return = {:.3}", summary.transposition_passed_count, summary.transposition_mean_return);
    println!("  - Transposition Laundering: {}/16 seeds", summary.transposition_laundering_passed_count);
    println!("================================================================================");
}
