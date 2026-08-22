//! Scout Q17E-G: Typed Relational Trainability Factorization
//!
//! Investigates trainability of typed relational architectures across:
//! - Axis 1 (Edge Encoder): tanh vs linear
//! - Axis 2 (Relational Update): Plain Accumulator vs Additive Residual Accumulator (m_{t+1} = m_t + eta * m_tilde_{t+1})
//! - Persistent Capacity: m in R^128 (persistent), e in R^32 (transient)
//! - Plus a 96-d monolithic recurrent baseline control
//!
//! Evaluates:
//! - Pre-registered k=2 validity gate (>= 14/16)
//! - Symmetric causal boundary double dissociation assay:
//!     C_theta(m2_intact, e3_intact) vs C_theta(m2_donor, e3_intact) vs C_theta(m2_intact, e3_donor)
//! - Zero-shot k=3 directional margin & paired sign-flip p-value
//! - Independent cloned-twin state surgery & swap effect
//! - Independent transposition reversals
//! - Deranged shuffle superiority
//! - Task-aligned sensitivities S_early and S_late

use continuity_garden_core::typed_model::{
    sigmoid, SerializedScoutGOrganism, TransitionObservation, TypedTrainabilityModel, EDGE_DIM,
    OBS_DIM, QUERY_DIM, REL_DIM, TRAIN_BATCHES_PER_EPOCH, TRAIN_EPOCHS, TRAIN_LR,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

fn compute_sign_flip_p_val(margins: &[f32]) -> f64 {
    let n = margins.len();
    if n == 0 {
        return 1.0;
    }
    let observed_mean = margins.iter().sum::<f32>() / n as f32;
    if observed_mean <= 0.0 {
        return 1.0;
    }

    let total_perms = 1 << n;
    let mut extreme_count = 0;

    for mask in 0..total_perms {
        let mut perm_sum = 0.0f32;
        for (i, &m) in margins.iter().enumerate() {
            let sign = if (mask >> i) & 1 == 1 { 1.0f32 } else { -1.0f32 };
            perm_sum += sign * m;
        }
        let perm_mean = perm_sum / n as f32;
        if perm_mean >= observed_mean {
            extreme_count += 1;
        }
    }

    (extreme_count as f64) / (total_perms as f64)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoutGResult {
    pub seed_index: usize,
    pub condition_name: String,
    pub is_linear_edge: bool,
    pub is_residual_acc: bool,
    pub k2_passed: bool,
    pub k2_margin: f32,
    pub k3_margin: f32,
    pub k3_passed: bool,
    // Symmetric Causal Boundary Assay
    pub k3_m_swap_effect: f32,
    pub k3_e_swap_effect: f32,
    pub k3_m_surgery_passed: bool,
    pub k3_transposition_passed: bool,
    pub k3_shuffle_passed: bool,
    pub s_early: f32,
    pub s_late: f32,
    pub sensor_accuracy: f32,
}

fn evaluate_scout_g_seed(seed_index: usize, is_linear: bool, is_residual: bool, eta: f32, cond_name: &str) -> (ScoutGResult, SerializedScoutGOrganism) {
    let seed = 88000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;

    let mut model = TypedTrainabilityModel::new_init(seed, is_linear, is_residual, eta);
    model.meta_train_bptt(aux_train_seed, TRAIN_EPOCHS);

    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;

    // k=2 Evaluation: A -> B -> C
    let m0 = vec![0.0f32; REL_DIM];
    let (e1, _) = model.encode_edge(&TransitionObservation::new(a, 1, b));
    let (m1, _, _) = model.compose_relation(&m0, &e1);
    let (e2, _) = model.encode_edge(&TransitionObservation::new(b, 2, c));
    let (m2, _, _) = model.compose_relation(&m1, &e2);
    let k2_margin = model.query_composition(&m2, (a, c)) - model.query_composition(&m2, (c, a));
    let k2_passed = k2_margin > 0.0;

    // k=3 Intact Evaluation: A(1) -> B(2) -> C(3) -> D(4) (Nuisance xi_0 = 0.0)
    let (e3_i, dt_e3_i) = model.encode_edge(&TransitionObservation::with_noise(c, 1, d, 0.0));
    let (m3_intact, _, dt_m3_i) = model.compose_relation(&m2, &e3_i);
    let k3_margin = model.query_composition(&m3_intact, (a, d)) - model.query_composition(&m3_intact, (d, a));
    let k3_passed = k3_margin > 0.0;

    // Independent Cloned-Twin Donor Stream: D(4) -1-> C(3) -2-> B(2) -1-> A(1) + Nuisance xi_1 = 0.01 (same action semantics 1,2,1!)
    let (e1_d, _) = model.encode_edge(&TransitionObservation::with_noise(d, 1, c, 0.01));
    let (m1_d, _, _) = model.compose_relation(&m0, &e1_d);
    let (e2_d, _) = model.encode_edge(&TransitionObservation::with_noise(c, 2, b, 0.01));
    let (m2_donor, _, _) = model.compose_relation(&m1_d, &e2_d);
    let (e3_donor, _) = model.encode_edge(&TransitionObservation::with_noise(b, 1, a, 0.01));

    // --- Symmetric Causal Boundary Assay ---
    // 1. Relational-State Swap: C_theta(m2_donor, e3_intact)
    let (m3_rel_swap, _, _) = model.compose_relation(&m2_donor, &e3_i);
    let m_margin_rel_swap = model.query_composition(&m3_rel_swap, (a, d)) - model.query_composition(&m3_rel_swap, (d, a));
    let k3_m_swap_effect = k3_margin - m_margin_rel_swap;
    let k3_m_surgery_passed = k3_margin > 0.0 && m_margin_rel_swap < 0.0;

    // 2. Local-Edge Swap: C_theta(m2_intact, e3_donor)
    let (m3_edge_swap, _, _) = model.compose_relation(&m2, &e3_donor);
    let m_margin_edge_swap = model.query_composition(&m3_edge_swap, (a, d)) - model.query_composition(&m3_edge_swap, (d, a));
    let k3_e_swap_effect = k3_margin - m_margin_edge_swap;

    // Independent Transposition Control: D(4) -1-> C(3) -2-> B(2) -1-> A(1) + Nuisance xi_2 = -0.01
    let (e1_t, _) = model.encode_edge(&TransitionObservation::with_noise(d, 1, c, -0.01));
    let (m1_t, _, _) = model.compose_relation(&m0, &e1_t);
    let (e2_t, _) = model.encode_edge(&TransitionObservation::with_noise(c, 2, b, -0.01));
    let (m2_t, _, _) = model.compose_relation(&m1_t, &e2_t);
    let (e3_t, _) = model.encode_edge(&TransitionObservation::with_noise(b, 1, a, -0.01));
    let (m3_trans, _, _) = model.compose_relation(&m2_t, &e3_t);
    let k3_transposition_score = model.query_composition(&m3_trans, (a, d)) - model.query_composition(&m3_trans, (d, a));
    let k3_transposition_passed = k3_transposition_score < 0.0;

    // Shuffle [e2, e3, e1]: (B, 2, C) -> (C, 1, D) -> (A, 1, B)
    let (es1, _) = model.encode_edge(&TransitionObservation::new(b, 2, c));
    let (ms1, _, _) = model.compose_relation(&m0, &es1);
    let (es2, _) = model.encode_edge(&TransitionObservation::new(c, 1, d));
    let (ms2, _, _) = model.compose_relation(&ms1, &es2);
    let (es3, _) = model.encode_edge(&TransitionObservation::new(a, 1, b));
    let (ms3, _, _) = model.compose_relation(&ms2, &es3);
    let shuf_score = model.query_composition(&ms3, (a, d)) - model.query_composition(&ms3, (d, a));
    let k3_shuffle_passed = k3_margin > shuf_score;

    // Task-Aligned Early and Late Sensitivities
    let q_ad_s = a as f32 / 5.0;
    let q_ad_d = d as f32 / 5.0;
    let mut dm_dm3 = vec![0.0f32; REL_DIM];
    for i in 0..REL_DIM {
        let e_fwd = model.w_q[i * QUERY_DIM] * q_ad_s + model.w_q[i * QUERY_DIM + 1] * q_ad_d;
        let e_rev = model.w_q[i * QUERY_DIM] * q_ad_d + model.w_q[i * QUERY_DIM + 1] * q_ad_s;
        dm_dm3[i] = model.w_r[i] * (e_fwd - e_rev);
    }

    // J_e1 = de1/dx1 = diag(dt_e1) * W_e (dim: EDGE_DIM x OBS_DIM)
    let (e1_fwd, dt_e1_fwd) = model.encode_edge(&TransitionObservation::new(a, 1, b));
    let mut j_e1 = vec![0.0f32; EDGE_DIM * OBS_DIM];
    for i in 0..EDGE_DIM {
        for j in 0..OBS_DIM {
            j_e1[i * OBS_DIM + j] = dt_e1_fwd[i] * model.w_e[i * OBS_DIM + j];
        }
    }

    // J_m1 = dm1/de1 * J_e1 = diag(dt_m1) * W_c * J_e1
    let (m1_fwd, _, dt_m1_fwd) = model.compose_relation(&m0, &e1_fwd);
    let mut j_m1 = vec![0.0f32; REL_DIM * OBS_DIM];
    for i in 0..REL_DIM {
        let scale = if is_residual { eta } else { 1.0 };
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for k in 0..EDGE_DIM {
                sum += model.w_c[i * EDGE_DIM + k] * j_e1[k * OBS_DIM + l];
            }
            j_m1[i * OBS_DIM + l] = dt_m1_fwd[i] * scale * sum;
        }
    }

    // J_m2 = dm2/dm1 * J_m1 = (I + eta * diag(dt_m2) * W_m) * J_m1
    let (e2_fwd, _) = model.encode_edge(&TransitionObservation::new(b, 2, c));
    let (m2_fwd, _, dt_m2_fwd) = model.compose_relation(&m1_fwd, &e2_fwd);
    let mut j_m2 = vec![0.0f32; REL_DIM * OBS_DIM];
    for i in 0..REL_DIM {
        let scale = if is_residual { eta } else { 1.0 };
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for k in 0..REL_DIM {
                sum += model.w_m[i * REL_DIM + k] * j_m1[k * OBS_DIM + l];
            }
            if is_residual {
                j_m2[i * OBS_DIM + l] = j_m1[i * OBS_DIM + l] + dt_m2_fwd[i] * scale * sum;
            } else {
                j_m2[i * OBS_DIM + l] = dt_m2_fwd[i] * sum;
            }
        }
    }

    // J_m3 = dm3/dm2 * J_m2
    let mut j_m3 = vec![0.0f32; REL_DIM * OBS_DIM];
    for i in 0..REL_DIM {
        let scale = if is_residual { eta } else { 1.0 };
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for k in 0..REL_DIM {
                sum += model.w_m[i * REL_DIM + k] * j_m2[k * OBS_DIM + l];
            }
            if is_residual {
                j_m3[i * OBS_DIM + l] = j_m2[i * OBS_DIM + l] + dt_m3_i[i] * scale * sum;
            } else {
                j_m3[i * OBS_DIM + l] = dt_m3_i[i] * sum;
            }
        }
    }

    let mut dm_dx1 = vec![0.0f32; OBS_DIM];
    for j in 0..OBS_DIM {
        let mut sum = 0.0f32;
        for i in 0..REL_DIM {
            sum += dm_dm3[i] * j_m3[i * OBS_DIM + j];
        }
        dm_dx1[j] = sum;
    }
    let s_early = dm_dx1.iter().map(|&v| v * v).sum::<f32>().sqrt();

    // S_late = ||dm_dm3 * dm3/de3 * de3/dx3||
    let mut j_e3 = vec![0.0f32; EDGE_DIM * OBS_DIM];
    for i in 0..EDGE_DIM {
        for j in 0..OBS_DIM {
            j_e3[i * OBS_DIM + j] = dt_e3_i[i] * model.w_e[i * OBS_DIM + j];
        }
    }
    let mut dm_dx3 = vec![0.0f32; OBS_DIM];
    for j in 0..OBS_DIM {
        let mut sum = 0.0f32;
        let scale = if is_residual { eta } else { 1.0 };
        for i in 0..REL_DIM {
            let mut sum_k = 0.0f32;
            for k in 0..EDGE_DIM {
                sum_k += model.w_c[i * EDGE_DIM + k] * j_e3[k * OBS_DIM + j];
            }
            sum += dm_dm3[i] * dt_m3_i[i] * scale * sum_k;
        }
        dm_dx3[j] = sum;
    }
    let s_late = dm_dx3.iter().map(|&v| v * v).sum::<f32>().sqrt();

    // Sensor Accuracy
    let mut rng_sensor = ChaCha8Rng::seed_from_u64(seed ^ 0x66666666);
    let mut sensor_correct = 0;
    for trial_id in 0..20 {
        let is_gold_valid = trial_id < 10;
        let cue_feat = if is_gold_valid {
            0.4 + rng_sensor.gen::<f32>() * 0.2
        } else {
            -3.5 - rng_sensor.gen::<f32>() * 0.5
        };
        let prob = model.query_sensor(&m3_intact, cue_feat);
        if (prob >= 0.5) == is_gold_valid {
            sensor_correct += 1;
        }
    }
    let sensor_accuracy = sensor_correct as f32 / 20.0;

    let parameter_sha256 = model.compute_parameter_sha256();
    let serialized_organism = SerializedScoutGOrganism {
        seed_index,
        seed,
        aux_train_seed,
        parameter_sha256,
        k2_margin,
        k3_margin,
        model: model.clone(),
    };

    let result = ScoutGResult {
        seed_index,
        condition_name: cond_name.to_string(),
        is_linear_edge: is_linear,
        is_residual_acc: is_residual,
        k2_passed,
        k2_margin,
        k3_margin,
        k3_passed,
        k3_m_swap_effect,
        k3_e_swap_effect,
        k3_m_surgery_passed,
        k3_transposition_passed,
        k3_shuffle_passed,
        s_early,
        s_late,
        sensor_accuracy,
    };

    (result, serialized_organism)
}

fn main() {
    println!("=================================================================================================================================");
    println!("SCOUT Q17E-G: Typed Relational Trainability Factorization (m in R^128, e in R^32)");
    println!("=================================================================================================================================");

    let conditions = [
        ("1. tanh Edge + Plain Accumulator", false, false, 0.0),
        ("2. linear Edge + Plain Accumulator", true, false, 0.0),
        ("3. tanh Edge + Additive Residual Accumulator (eta=0.50)", false, true, 0.50),
        ("4. linear Edge + Additive Residual Accumulator (eta=0.50)", true, true, 0.50),
        ("5. linear Edge + Additive Residual Accumulator (eta=1.00)", true, true, 1.00),
    ];

    let mut all_results = Vec::new();
    let mut winning_serialized_models = Vec::new();

    for (name, is_linear, is_residual, eta) in &conditions {
        println!("\n--- Condition: {} ---", name);
        let pairs: Vec<(ScoutGResult, SerializedScoutGOrganism)> = (1..=16)
            .into_par_iter()
            .map(|i| evaluate_scout_g_seed(i, *is_linear, *is_residual, *eta, name))
            .collect();

        let results: Vec<ScoutGResult> = pairs.iter().map(|p| p.0.clone()).collect();
        all_results.extend(results.clone());

        if name.starts_with("5.") {
            winning_serialized_models = pairs.iter().map(|p| p.1.clone()).collect();
        }

        let k2_pass = results.iter().filter(|r| r.k2_passed).count();
        let k2_valid = k2_pass >= 14;

        let k3_pass = results.iter().filter(|r| r.k3_passed).count();
        let k3_margins: Vec<f32> = results.iter().map(|r| r.k3_margin).collect();
        let k3_p = compute_sign_flip_p_val(&k3_margins);

        let m_surg_pass = results.iter().filter(|r| r.k3_m_surgery_passed).count();
        let mean_m_swap: f32 = results.iter().map(|r| r.k3_m_swap_effect).sum::<f32>() / 16.0;
        let mean_e_swap: f32 = results.iter().map(|r| r.k3_e_swap_effect).sum::<f32>() / 16.0;
        let trans_pass = results.iter().filter(|r| r.k3_transposition_passed).count();
        let shuf_pass = results.iter().filter(|r| r.k3_shuffle_passed).count();

        let s_early: f32 = results.iter().map(|r| r.s_early).sum::<f32>() / 16.0;
        let s_late: f32 = results.iter().map(|r| r.s_late).sum::<f32>() / 16.0;
        let sensor_pass = results.iter().filter(|r| r.sensor_accuracy >= 0.90).count();

        println!("  k=2 Baseline Retention:                         {}/16 ({:.1}%) -> {}", k2_pass, k2_pass as f32 / 16.0 * 100.0, if k2_valid { "VALID FOR k=3 INTERPRETATION" } else { "TRAINABILITY DEFECT (< 14/16)" });
        if k2_valid {
            println!("  k=3 Positive Direction (m_3 > 0):               {}/16 (p={:.4})", k3_pass, k3_p);
            println!("  SYMMETRIC CAUSAL BOUNDARY ASSAY:");
            println!("    - Relational State Swap C(m2_donor, e3_intact):   Mean Swap: {:+.4} | Surgery Flips: {}/16 ({:.1}%)", mean_m_swap, m_surg_pass, m_surg_pass as f32 / 16.0 * 100.0);
            println!("    - Local Edge Swap C(m2_intact, e3_donor):         Mean Swap: {:+.4}", mean_e_swap);
            println!("  k=3 Transposition Reversals:                    {}/16 ({:.1}%)", trans_pass, trans_pass as f32 / 16.0 * 100.0);
            println!("  k=3 Deranged Shuffle Superiority:               {}/16 ({:.1}%)", shuf_pass, shuf_pass as f32 / 16.0 * 100.0);
            println!("  Task-Aligned Early Sensitivity (S_early):       {:.4}", s_early);
            println!("  Task-Aligned Last-Edge Sensitivity (S_late):    {:.4}", s_late);
        } else {
            println!("  [NOTE: Zero-shot k=3 skipped from scientific interpretation due to k=2 trainability barrier]");
        }
        println!("  1-Hop Sensor Accuracy (>= 90%):                 {}/16 ({:.1}%)", sensor_pass, sensor_pass as f32 / 16.0 * 100.0);
    }

    println!("\n=================================================================================================================================");

    let data_dir = Path::new("crates/continuity_garden_core/data");
    std::fs::create_dir_all(data_dir).expect("Failed to create data directory");
    
    let out_file = data_dir.join("q17e_g_typed_trainability_results.json");
    let file = File::create(&out_file).expect("Failed to create scout G results JSON");
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &all_results).expect("Failed to write JSON");
    println!("Persisted full Scout G telemetry to: {}", out_file.display());

    let models_file = data_dir.join("q17e_g_serialized_models.json");
    let m_file = File::create(&models_file).expect("Failed to create serialized models JSON");
    let m_writer = BufWriter::new(m_file);
    serde_json::to_writer_pretty(m_writer, &winning_serialized_models).expect("Failed to write serialized models JSON");
    println!("Persisted 16 winning Condition 5 serialized models (with SHA-256 digests) to: {}", models_file.display());
}
