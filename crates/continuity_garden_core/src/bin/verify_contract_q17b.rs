//! Deterministic Contract Acceptance Verifier for CONTRACT-E-Q17B
//! Asserts all Gates 1-6, transposition falsification, and temporal-shuffle controls directly from q17b_summary.json.

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
    pub self_sup_multihop_acc: f32,
    pub shuffled_control_multihop_acc: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q17BSummary {
    pub protocol: String,
    pub total_seeds: usize,
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

    // 2. Gate 1: Zero-Shot Multi-Hop Conflict Floor (>= 10/16)
    if summary.gate1_multihop_count < 10 {
        violations.push(format!("Gate 1 floor >= 10/16 seeds, observed {}/16", summary.gate1_multihop_count));
    }

    // 3. Gate 2: Laundering Discrimination Floor (>= 10/16)
    if summary.gate2_laundering_count < 10 {
        violations.push(format!("Gate 2 floor >= 10/16 seeds, observed {}/16", summary.gate2_laundering_count));
    }

    // 4. Gate 3: Temporal Shuffle Control Superiority (n10 - n01 >= 3, p < 0.05)
    if summary.gate3_delta < 3 {
        violations.push(format!("Gate 3 Delta (n10 - n01) floor >= 3, observed {}", summary.gate3_delta));
    }
    if summary.gate3_p_value >= 0.05 {
        violations.push(format!("Gate 3 p-value floor < 0.05, observed {:.4e}", summary.gate3_p_value));
    }

    // 5. Gate 4: Directional Transposition Falsification (<= 2/16 seeds, return < 0.00)
    if summary.gate4_transposition_passed_count > 2 {
        violations.push(format!("Gate 4 Transposition ceiling <= 2/16 seeds, observed {}/16", summary.gate4_transposition_passed_count));
    }
    if summary.gate4_transposition_mean_return >= 0.0 {
        violations.push(format!("Gate 4 Mean Return ceiling < 0.00, observed {:.3}", summary.gate4_transposition_mean_return));
    }

    // 6. Gate 5: Transposition Laundering Floor (>= 10/16)
    if summary.gate5_transposition_laundering_count < 10 {
        violations.push(format!("Gate 5 floor >= 10/16 seeds, observed {}/16", summary.gate5_transposition_laundering_count));
    }

    // 7. Gate 6: Mechanistic Path-Break Specificity (p < 0.01)
    if summary.gate6_p_value >= 0.01 {
        violations.push(format!("Gate 6 p-value floor < 0.01, observed {:.4e}", summary.gate6_p_value));
    }

    if !violations.is_empty() {
        eprintln!("\nCONTRACT VERIFICATION FAILED ({} violations):", violations.len());
        for v in &violations {
            eprintln!("  - [VIOLATION] {}", v);
        }
        exit(1);
    }

    println!("\nALL CONTRACT-E-Q17B FROZEN ACCEPTANCE CRITERIA CLEANLY SATISFIED:");
    println!("  - Gate 1 (Zero-Shot Multi-Hop Conflict):   {}/16 seeds (PASS)", summary.gate1_multihop_count);
    println!("  - Gate 2 (Laundering Discrimination):       {}/16 seeds (PASS)", summary.gate2_laundering_count);
    println!("  - Gate 3 (Temporal Shuffle Superiority):    n10={}, n01={}, Delta={}, p={:.4e} (PASS)", summary.gate3_n10, summary.gate3_n01, summary.gate3_delta, summary.gate3_p_value);
    println!("  - Gate 4 (Directional Transposition Fals):  {}/16 passed, return={:.3} (PASS)", summary.gate4_transposition_passed_count, summary.gate4_transposition_mean_return);
    println!("  - Gate 5 (Transposition Laundering Invar):  {}/16 seeds (PASS)", summary.gate5_transposition_laundering_count);
    println!("  - Gate 6 (Mechanistic Path-Break Specific): p={:.4e} (PASS)", summary.gate6_p_value);
    println!("================================================================================");
}
