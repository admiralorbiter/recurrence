//! SCOUT-E-Q17E-I: Binding Identifiability Under Matched Broken Joins
//!
//! Trains the winning linear-edge/additive-residual typed architecture (m in R^128, e in R^32)
//! under randomized entity permutations with matched broken-join and wrong-terminal negatives.
//! Evaluates whether the trained representation acquires true compositional binding
//! (destination grounding + source sensitivity) without 3-hop supervision.

use std::fs;
use std::path::Path;

use continuity_garden_core::typed_model::{
    sigmoid, TransitionObservation, TypedTrainabilityModel, EDGE_DIM, OBS_DIM, QUERY_DIM, REL_DIM,
    TRAIN_BATCHES_PER_EPOCH, TRAIN_EPOCHS, TRAIN_LR,
};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoutISeedResult {
    pub seed_index: usize,
    pub seed: u64,
    pub k2_train_valid: bool,
    pub k2_intact_margin: f32,
    pub k2_broken_join_rejection: f32, // Margin drop when join is broken
    pub k2_wrong_dst_rejection: f32,   // Margin drop when dst is wrong
    pub k3_zero_shot_margin: f32,      // Intact A->B->C + C->D query(A, D)
    pub k3_passed: bool,
    pub pre_edge_margin: f32,          // Query(A, D) on m2 (before C->D)
    pub zero_edge_margin: f32,         // m2 + 0 edge
    pub wrong_src_edge_margin: f32,    // m2 + (X -> D) (X != C)
    pub wrong_dst_edge_margin: f32,    // m2 + (C -> E) (E != D)
    pub donor_hist_margin: f32,        // donor_m2 + (C -> D)
    pub true_binding_passed: bool,     // Intact > 0 && Intact > WrongDst && Intact > WrongSrc && Intact > Zero && Intact > PreEdge
    pub source_grounded: bool,         // Intact > WrongSrc
    pub dest_grounded: bool,           // Intact > WrongDst
    pub sensor_accuracy: f32,
}

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

/// Trains the typed model with matched broken-join curriculum on randomized node permutations
fn train_broken_join_model(model: &mut TypedTrainabilityModel, train_seed: u64, epochs: usize) {
    let mut rng = ChaCha8Rng::seed_from_u64(train_seed);
    let nodes = [1, 2, 3, 4, 5, 6];
    let lr = TRAIN_LR;

    for _epoch in 0..epochs {
        for _batch in 0..TRAIN_BATCHES_PER_EPOCH {
            // Sample random permutation of nodes for this trajectory
            let mut perm_nodes = nodes;
            perm_nodes.shuffle(&mut rng);
            let u = perm_nodes[0];
            let v = perm_nodes[1];
            let w = perm_nodes[2];
            let x = perm_nodes[3]; // For broken join
            let y = perm_nodes[4]; // For wrong dst

            let a1 = rng.gen_range(1..=3);
            let a2 = rng.gen_range(1..=3);

            let xi1 = (rng.gen::<f32>() - 0.5) * 0.02;
            let xi2 = (rng.gen::<f32>() - 0.5) * 0.02;

            // 1. POSITIVE 2-HOP TRAJECTORY: u -> v -> w
            let obs1 = TransitionObservation::with_noise(u, a1, v, xi1);
            let m0 = vec![0.0f32; REL_DIM];
            let (e1, dt_e1) = model.encode_edge(&obs1);
            let (m1, _, dt_m1) = model.compose_relation(&m0, &e1);

            let obs2 = TransitionObservation::with_noise(v, a2, w, xi2);
            let (e2, dt_e2) = model.encode_edge(&obs2);
            let (m2, _, dt_m2) = model.compose_relation(&m1, &e2);

            let mut grad_m2 = vec![0.0f32; REL_DIM];
            let mut grad_m1 = vec![0.0f32; REL_DIM];

            // Terminal positive queries on m2: (u -> w) = 1, (w -> u) = 0, (u -> y) = 0
            for &(q_pair, target_y) in &[
                ((u, w), 1.0f32),
                ((w, u), 0.0f32),
                ((u, y), 0.0f32),
            ] {
                let q_s = q_pair.0 as f32 / 5.0;
                let q_d = q_pair.1 as f32 / 5.0;

                let mut logit = model.b_r;
                for i in 0..REL_DIM {
                    let e_q = model.w_q[i * QUERY_DIM] * q_s + model.w_q[i * QUERY_DIM + 1] * q_d;
                    logit += model.w_r[i] * m2[i] * e_q;
                }
                let pred = sigmoid(logit);
                let err = pred - target_y;

                model.b_r -= lr * err * 0.33;
                for i in 0..REL_DIM {
                    let e_q = model.w_q[i * QUERY_DIM] * q_s + model.w_q[i * QUERY_DIM + 1] * q_d;
                    let d_w_r = err * m2[i] * e_q;
                    let d_e_q = err * model.w_r[i] * m2[i];
                    let d_m2_i = err * model.w_r[i] * e_q;

                    model.w_r[i] -= lr * d_w_r * 0.33;
                    model.w_q[i * QUERY_DIM] -= lr * d_e_q * q_s * 0.33;
                    model.w_q[i * QUERY_DIM + 1] -= lr * d_e_q * q_d * 0.33;
                    grad_m2[i] += d_m2_i * 0.33;
                }
            }

            // Shared Prefix supervision on m1: (u -> v) = 1, (v -> u) = 0, (u -> x) = 0
            for &(q_pair, target_y) in &[
                ((u, v), 1.0f32),
                ((v, u), 0.0f32),
                ((u, x), 0.0f32),
            ] {
                let q_s = q_pair.0 as f32 / 5.0;
                let q_d = q_pair.1 as f32 / 5.0;

                let mut logit = model.b_r;
                for i in 0..REL_DIM {
                    let e_q = model.w_q[i * QUERY_DIM] * q_s + model.w_q[i * QUERY_DIM + 1] * q_d;
                    logit += model.w_r[i] * m1[i] * e_q;
                }
                let pred = sigmoid(logit);
                let err = pred - target_y;

                model.b_r -= lr * err * 0.33;
                for i in 0..REL_DIM {
                    let e_q = model.w_q[i * QUERY_DIM] * q_s + model.w_q[i * QUERY_DIM + 1] * q_d;
                    let d_w_r = err * m1[i] * e_q;
                    let d_e_q = err * model.w_r[i] * m1[i];
                    let d_m1_i = err * model.w_r[i] * e_q;

                    model.w_r[i] -= lr * d_w_r * 0.33;
                    model.w_q[i * QUERY_DIM] -= lr * d_e_q * q_s * 0.33;
                    model.w_q[i * QUERY_DIM + 1] -= lr * d_e_q * q_d * 0.33;
                    grad_m1[i] += d_m1_i * 0.33;
                }
            }

            // Backprop Step 2: C_theta(m1, e2) -> m2
            let mut d_act_m2 = vec![0.0f32; REL_DIM];
            let mut grad_e2 = vec![0.0f32; EDGE_DIM];
            for i in 0..REL_DIM {
                let scale = if model.is_residual_accumulator { model.eta_residual } else { 1.0 };
                d_act_m2[i] = grad_m2[i] * scale * dt_m2[i];
                model.b_m[i] -= lr * d_act_m2[i];
                for j in 0..EDGE_DIM {
                    model.w_c[i * EDGE_DIM + j] -= lr * d_act_m2[i] * e2[j];
                    grad_e2[j] += d_act_m2[i] * model.w_c[i * EDGE_DIM + j];
                }
            }
            for j in 0..REL_DIM {
                let mut sum = 0.0f32;
                for i in 0..REL_DIM {
                    sum += model.w_m[i * REL_DIM + j] * d_act_m2[i];
                    model.w_m[i * REL_DIM + j] -= lr * d_act_m2[i] * m1[j];
                }
                if model.is_residual_accumulator {
                    grad_m1[j] += grad_m2[j] + sum; // Identity residual pass-through!
                } else {
                    grad_m1[j] += sum;
                }
            }

            // Backprop Edge 2: E(x2) -> e2
            let x2 = obs2.to_vec();
            for i in 0..EDGE_DIM {
                let d_act_e2 = grad_e2[i] * dt_e2[i];
                model.b_e[i] -= lr * d_act_e2;
                for j in 0..OBS_DIM {
                    model.w_e[i * OBS_DIM + j] -= lr * d_act_e2 * x2[j];
                }
            }

            // Backprop Step 1: C_theta(m0, e1) -> m1
            let mut d_act_m1 = vec![0.0f32; REL_DIM];
            let mut grad_e1 = vec![0.0f32; EDGE_DIM];
            for i in 0..REL_DIM {
                let scale = if model.is_residual_accumulator { model.eta_residual } else { 1.0 };
                d_act_m1[i] = grad_m1[i] * scale * dt_m1[i];
                model.b_m[i] -= lr * d_act_m1[i];
                for j in 0..EDGE_DIM {
                    model.w_c[i * EDGE_DIM + j] -= lr * d_act_m1[i] * e1[j];
                    grad_e1[j] += d_act_m1[i] * model.w_c[i * EDGE_DIM + j];
                }
            }

            // Backprop Edge 1: E(x1) -> e1
            let x1 = obs1.to_vec();
            for i in 0..EDGE_DIM {
                let d_act_e1 = grad_e1[i] * dt_e1[i];
                model.b_e[i] -= lr * d_act_e1;
                for j in 0..OBS_DIM {
                    model.w_e[i * OBS_DIM + j] -= lr * d_act_e1 * x1[j];
                }
            }

            // 2. NEGATIVE: BROKEN JOIN (u -> v, then x -> w where x != v, query (u, w))
            let obs2_brk = TransitionObservation::with_noise(x, a2, w, xi2);
            let (e2_brk, dt_e2_brk) = model.encode_edge(&obs2_brk);
            let (m2_brk, _, dt_m2_brk) = model.compose_relation(&m1, &e2_brk);

            let q_s = u as f32 / 5.0;
            let q_d = w as f32 / 5.0;
            let mut logit_brk = model.b_r;
            for i in 0..REL_DIM {
                let e_q = model.w_q[i * QUERY_DIM] * q_s + model.w_q[i * QUERY_DIM + 1] * q_d;
                logit_brk += model.w_r[i] * m2_brk[i] * e_q;
            }
            let pred_brk = sigmoid(logit_brk);
            let err_brk = pred_brk - 0.0; // Target: 0.0

            model.b_r -= lr * err_brk * 0.33;
            let mut grad_m2_brk = vec![0.0f32; REL_DIM];
            for i in 0..REL_DIM {
                let e_q = model.w_q[i * QUERY_DIM] * q_s + model.w_q[i * QUERY_DIM + 1] * q_d;
                let d_w_r = err_brk * m2_brk[i] * e_q;
                let d_e_q = err_brk * model.w_r[i] * m2_brk[i];
                let d_m2_i = err_brk * model.w_r[i] * e_q;

                model.w_r[i] -= lr * d_w_r * 0.33;
                model.w_q[i * QUERY_DIM] -= lr * d_e_q * q_s * 0.33;
                model.w_q[i * QUERY_DIM + 1] -= lr * d_e_q * q_d * 0.33;
                grad_m2_brk[i] += d_m2_i * 0.33;
            }

            // Backprop broken join step 2
            let mut d_act_brk = vec![0.0f32; REL_DIM];
            let mut grad_e2_brk = vec![0.0f32; EDGE_DIM];
            for i in 0..REL_DIM {
                let scale = if model.is_residual_accumulator { model.eta_residual } else { 1.0 };
                d_act_brk[i] = grad_m2_brk[i] * scale * dt_m2_brk[i];
                model.b_m[i] -= lr * d_act_brk[i];
                for j in 0..EDGE_DIM {
                    model.w_c[i * EDGE_DIM + j] -= lr * d_act_brk[i] * e2_brk[j];
                    grad_e2_brk[j] += d_act_brk[i] * model.w_c[i * EDGE_DIM + j];
                }
            }
            let x2_brk = obs2_brk.to_vec();
            for i in 0..EDGE_DIM {
                let d_act_e2 = grad_e2_brk[i] * dt_e2_brk[i];
                model.b_e[i] -= lr * d_act_e2;
                for j in 0..OBS_DIM {
                    model.w_e[i * OBS_DIM + j] -= lr * d_act_e2 * x2_brk[j];
                }
            }

            // Sensor competence training
            let sensor_prob = model.query_sensor(&m2, 0.5);
            let sensor_err = sensor_prob - 0.95;
            model.b_sensor -= lr * sensor_err * 0.1;
            for i in 0..REL_DIM {
                model.w_sensor[i] -= lr * sensor_err * m2[i] * 0.01;
            }
        }
    }
}

fn evaluate_scout_i_seed(seed_index: usize) -> ScoutISeedResult {
    let seed = 88000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;

    let mut model = TypedTrainabilityModel::new_init(seed, true, true, 1.0);
    train_broken_join_model(&mut model, aux_train_seed, TRAIN_EPOCHS);

    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;
    let e = 5;
    let x = 6;

    let m0 = vec![0.0f32; REL_DIM];

    // 1. Two-step training validity: A -> B -> C
    let (e1, _) = model.encode_edge(&TransitionObservation::new(a, 1, b));
    let (m1, _, _) = model.compose_relation(&m0, &e1);
    let (e2, _) = model.encode_edge(&TransitionObservation::new(b, 2, c));
    let (m2, _, _) = model.compose_relation(&m1, &e2);

    let k2_intact_margin = model.query_composition(&m2, (a, c)) - model.query_composition(&m2, (c, a));
    let k2_train_valid = k2_intact_margin > 0.0;

    // Negatives on k=2:
    // Broken join: A -> B, then X -> C (X != B)
    let (e2_broken, _) = model.encode_edge(&TransitionObservation::new(x, 2, c));
    let (m2_broken, _, _) = model.compose_relation(&m1, &e2_broken);
    let score_broken = model.query_composition(&m2_broken, (a, c));
    let k2_broken_join_rejection = model.query_composition(&m2, (a, c)) - score_broken;

    // Wrong destination: A -> B, then B -> E (E != C)
    let (e2_wrong_dst, _) = model.encode_edge(&TransitionObservation::new(b, 2, e));
    let (m2_wrong_dst, _, _) = model.compose_relation(&m1, &e2_wrong_dst);
    let score_wrong_dst = model.query_composition(&m2_wrong_dst, (a, c));
    let k2_wrong_dst_rejection = model.query_composition(&m2, (a, c)) - score_wrong_dst;

    // 2. DECISIVE ZERO-SHOT BATTERY ON 3-HOP (Query (A, D) = (1, 4)):
    // A. Pre-edge margin on m2
    let pre_edge_margin = model.query_composition(&m2, (a, d)) - model.query_composition(&m2, (d, a));

    // B. Intact 3-step sequence: m2 + (C -> D)
    let (e3_intact, _) = model.encode_edge(&TransitionObservation::new(c, 1, d));
    let (m3_intact, _, _) = model.compose_relation(&m2, &e3_intact);
    let k3_zero_shot_margin = model.query_composition(&m3_intact, (a, d)) - model.query_composition(&m3_intact, (d, a));
    let k3_passed = k3_zero_shot_margin > 0.0;

    // C. Zero final edge: m2 + 0
    let (m3_zero, _, _) = model.compose_relation(&m2, &vec![0.0f32; EDGE_DIM]);
    let zero_edge_margin = model.query_composition(&m3_zero, (a, d)) - model.query_composition(&m3_zero, (d, a));

    // D. Wrong source final edge: m2 + (X -> D) where X != C (e.g. X=5 -> 4)
    let (e3_wrong_src, _) = model.encode_edge(&TransitionObservation::new(5, 1, d));
    let (m3_wrong_src, _, _) = model.compose_relation(&m2, &e3_wrong_src);
    let wrong_src_edge_margin = model.query_composition(&m3_wrong_src, (a, d)) - model.query_composition(&m3_wrong_src, (d, a));

    // E. Wrong destination final edge: m2 + (C -> E) where E != D (e.g. 3 -> 5)
    let (e3_wrong_dst, _) = model.encode_edge(&TransitionObservation::new(c, 1, e));
    let (m3_wrong_dst, _, _) = model.compose_relation(&m2, &e3_wrong_dst);
    let wrong_dst_edge_margin = model.query_composition(&m3_wrong_dst, (a, d)) - model.query_composition(&m3_wrong_dst, (d, a));

    // F. Donor history transplant: D -> C -> B (donor) + intact C -> D
    let (e1_d, _) = model.encode_edge(&TransitionObservation::new(d, 1, c));
    let (m1_d, _, _) = model.compose_relation(&m0, &e1_d);
    let (e2_d, _) = model.encode_edge(&TransitionObservation::new(c, 2, b));
    let (m2_donor, _, _) = model.compose_relation(&m1_d, &e2_d);
    let (m3_donor, _, _) = model.compose_relation(&m2_donor, &e3_intact);
    let donor_hist_margin = model.query_composition(&m3_donor, (a, d)) - model.query_composition(&m3_donor, (d, a));

    let source_grounded = k3_zero_shot_margin > wrong_src_edge_margin;
    let dest_grounded = k3_zero_shot_margin > wrong_dst_edge_margin;
    let true_binding_passed = k3_passed
        && source_grounded
        && dest_grounded
        && (k3_zero_shot_margin > zero_edge_margin)
        && (k3_zero_shot_margin > pre_edge_margin)
        && (donor_hist_margin < 0.0);

    // Sensor competence
    let mut rng_sens = ChaCha8Rng::seed_from_u64(seed ^ 0x9999);
    let mut sens_corr = 0;
    for i in 0..20 {
        let is_pos = i < 10;
        let feat = if is_pos { 0.5 + rng_sens.gen::<f32>() * 0.1 } else { -3.5 - rng_sens.gen::<f32>() * 0.5 };
        let p = model.query_sensor(&m3_intact, feat);
        if (p >= 0.5) == is_pos {
            sens_corr += 1;
        }
    }
    let sensor_accuracy = sens_corr as f32 / 20.0;

    ScoutISeedResult {
        seed_index,
        seed,
        k2_train_valid,
        k2_intact_margin,
        k2_broken_join_rejection,
        k2_wrong_dst_rejection,
        k3_zero_shot_margin,
        k3_passed,
        pre_edge_margin,
        zero_edge_margin,
        wrong_src_edge_margin,
        wrong_dst_edge_margin,
        donor_hist_margin,
        true_binding_passed,
        source_grounded,
        dest_grounded,
        sensor_accuracy,
    }
}

fn main() {
    println!("================================================================================");
    println!("SCOUT-E-Q17E-I: Binding Identifiability Under Matched Broken Joins");
    println!("Randomized Node Permutations + Matched Broken Joins (Without 3-Hop Supervision)");
    println!("================================================================================");

    let results: Vec<ScoutISeedResult> = (1..=16)
        .into_par_iter()
        .map(|i| evaluate_scout_i_seed(i))
        .collect();

    let n = results.len() as f32;
    let k2_valid_count = results.iter().filter(|r| r.k2_train_valid).count();
    let k3_pass_count = results.iter().filter(|r| r.k3_passed).count();
    let true_binding_count = results.iter().filter(|r| r.true_binding_passed).count();
    let src_ground_count = results.iter().filter(|r| r.source_grounded).count();
    let dst_ground_count = results.iter().filter(|r| r.dest_grounded).count();

    let avg_k2_margin = results.iter().map(|r| r.k2_intact_margin).sum::<f32>() / n;
    let avg_brk_drop = results.iter().map(|r| r.k2_broken_join_rejection).sum::<f32>() / n;
    let avg_wd_drop = results.iter().map(|r| r.k2_wrong_dst_rejection).sum::<f32>() / n;

    let avg_k3_margin = results.iter().map(|r| r.k3_zero_shot_margin).sum::<f32>() / n;
    let avg_pre_margin = results.iter().map(|r| r.pre_edge_margin).sum::<f32>() / n;
    let avg_zero_margin = results.iter().map(|r| r.zero_edge_margin).sum::<f32>() / n;
    let avg_w_src_margin = results.iter().map(|r| r.wrong_src_edge_margin).sum::<f32>() / n;
    let avg_w_dst_margin = results.iter().map(|r| r.wrong_dst_edge_margin).sum::<f32>() / n;
    let avg_donor_margin = results.iter().map(|r| r.donor_hist_margin).sum::<f32>() / n;

    let k3_margins: Vec<f32> = results.iter().map(|r| r.k3_zero_shot_margin).collect();
    let k3_p_val = compute_sign_flip_p_val(&k3_margins);

    println!("\n--------------------------------------------------------------------------------");
    println!("1. TWO-STEP DEVELOPMENTAL VALIDITY & DISCRIMINATION:");
    println!("   k=2 Retention Rate:                        {}/16 ({:.1}%)", k2_valid_count, k2_valid_count as f32 / n * 100.0);
    println!("   k=2 Intact Margin:                         {:>+6.2}", avg_k2_margin);
    println!("   k=2 Broken-Join Drop (A->B, X->C vs A->C): {:>+6.2}", avg_brk_drop);
    println!("   k=2 Wrong-Dst Drop (A->B, B->E vs A->C):   {:>+6.2}", avg_wd_drop);
    println!("--------------------------------------------------------------------------------");
    println!("2. ZERO-SHOT 3-HOP BINDING ASSAY (Query: A -> D, unseen):");
    println!("   A. Pre-Edge Margin on m2:                  {:>+6.2}", avg_pre_margin);
    println!("   B. Intact 3-Step Sequence (A->B->C->D):    {:>+6.2} (Pass: {}/16, p={:.4})", avg_k3_margin, k3_pass_count, k3_p_val);
    println!("   C. Zero Final Edge (m2 + 0):               {:>+6.2} (Drop: {:>+6.2})", avg_zero_margin, avg_k3_margin - avg_zero_margin);
    println!("   D. Wrong Source Edge (m2 + X->D):          {:>+6.2} (Drop: {:>+6.2}) -> Grounded: {}/16", avg_w_src_margin, avg_k3_margin - avg_w_src_margin, src_ground_count);
    println!("   E. Wrong Dst Edge (m2 + C->E):             {:>+6.2} (Drop: {:>+6.2}) -> Grounded: {}/16", avg_w_dst_margin, avg_k3_margin - avg_w_dst_margin, dst_ground_count);
    println!("   F. Donor History Transplant:               {:>+6.2} (Drop: {:>+6.2})", avg_donor_margin, avg_k3_margin - avg_donor_margin);
    println!("--------------------------------------------------------------------------------");
    println!("3. TRUE COMPOSITIONAL BINDING VERDICT (All 6 criteria): {}/16 ({:.1}%)", true_binding_count, true_binding_count as f32 / n * 100.0);
    println!("================================================================================");

    for r in &results {
        println!(
            "Seed [{:>2}] k2:{:>+5.2} | Pre:{:>+5.2} | Intact:{:>+5.2} | Zero:{:>+5.2} | W_Src:{:>+5.2} | W_Dst:{:>+5.2} | Donor:{:>+5.2} | Binding: {}",
            r.seed_index, r.k2_intact_margin, r.pre_edge_margin, r.k3_zero_shot_margin, r.zero_edge_margin, r.wrong_src_edge_margin, r.wrong_dst_edge_margin, r.donor_hist_margin,
            if r.true_binding_passed { "PASS" } else { "FAIL" }
        );
    }

    let out_dir = Path::new("crates/continuity_garden_core/data");
    fs::create_dir_all(out_dir).expect("Failed to create data directory");
    let out_path = out_dir.join("q17e_i_broken_joins_results.json");
    let json_bytes = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    fs::write(&out_path, json_bytes).expect("Failed to write results file");
    println!("\nPersisted Scout I telemetry to: {}", out_path.display());
}
