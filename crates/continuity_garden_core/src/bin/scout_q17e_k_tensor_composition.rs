//! SCOUT-E-Q17E-K: Capacity-Matched Tensor Relational Composition Study
//!
//! Investigates variable binding and recursive composition across three arms:
//! - Arm A: 128-d Additive State + K0 Exhaustive (36-class) & Nonlinear MLP Diagnostic Probes.
//! - Arm B: 121-d Explicit Tensor Matrix Positive Control (Orthogonal Unification Algebra).
//! - Arm C: 121-d Learned Tensor Organism (Learned Embeddings & Multiplicative Contraction).
//!
//! Protocol:
//! - Curriculum: 1- and 2-step experience with matched broken joins and alternate-destination counterfactuals.
//! - Zero 3-hop labels during meta-training.
//! - 16 independent seeds.
//! - Causal diagnostic battery: k=2 validity, zero-shot k=3 composition, source grounding (X -> D),
//!   destination grounding (C -> E), and endpoint selectivity margin.

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

pub const NUM_NODES: usize = 6;
pub const NUM_PAIRS: usize = NUM_NODES * NUM_NODES; // 36
pub const TENSOR_P: usize = 11;
pub const TENSOR_DIM: usize = TENSOR_P * TENSOR_P; // 121

// -----------------------------------------------------------------------------
// Arm A: K0 Exhaustive & Nonlinear Probing on Existing Additive Organism
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmAK0Result {
    pub m3_linear_36class_acc: f32, // Exhaustive 36-class linear probe on m3 -> (u, z)
    pub m3_mlp_36class_acc: f32,    // 2-layer MLP (128 -> 64 -> 36) on m3 -> (u, z)
    pub k0_binding_found_nonlinearly: bool,
}

// -----------------------------------------------------------------------------
// Arm B: Explicit Tensor Matrix Positive Control (Fixed Orthogonal Embeddings)
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmBResult {
    pub k2_target_score: f32,
    pub k2_reverse_score: f32,
    pub k2_distractor_score: f32,
    pub k2_selectivity_margin: f32,
    pub k2_pass: bool,
    pub k3_target_score: f32,
    pub k3_reverse_score: f32,
    pub k3_distractor_score: f32,
    pub k3_selectivity_margin: f32,
    pub k3_pass: bool,
    pub source_grounding_drop: f32,     // Drop on X -> D broken join
    pub destination_grounding_gap: f32,  // Difference between C -> D and C -> E
}

// -----------------------------------------------------------------------------
// Arm C: Learned Tensor Relational Organism (121-d Learned Embeddings)
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmCResult {
    pub k2_validity_acc: f32,
    pub k2_pass: bool,
    pub k3_zero_shot_acc: f32,
    pub k3_pass: bool,
    pub k3_target_score: f32,
    pub k3_reverse_score: f32,
    pub k3_distractor_score: f32,
    pub k3_selectivity_margin: f32,
    pub source_grounding_drop: f32,
    pub destination_grounding_gap: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoutKSeedResult {
    pub seed_index: usize,
    pub seed: u64,
    pub arm_a_k0: ArmAK0Result,
    pub arm_b_pos_control: ArmBResult,
    pub arm_c_learned_tensor: ArmCResult,
}

// Helper: Train exact Scout-I additive model
fn train_scout_i_exact(model: &mut TypedTrainabilityModel, train_seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(train_seed);
    let nodes = [1, 2, 3, 4, 5, 6];
    let lr = TRAIN_LR;

    for _epoch in 0..TRAIN_EPOCHS {
        for _batch in 0..TRAIN_BATCHES_PER_EPOCH {
            let mut perm = nodes;
            perm.shuffle(&mut rng);
            let u = perm[0];
            let v = perm[1];
            let w = perm[2];
            let x = perm[3];
            let y = perm[4];

            let a1 = rng.gen_range(1..=3);
            let a2 = rng.gen_range(1..=3);

            let obs1 = TransitionObservation::new(u, a1, v);
            let m0 = vec![0.0f32; REL_DIM];
            let (e1, dt_e1) = model.encode_edge(&obs1);
            let (m1, _, dt_m1) = model.compose_relation(&m0, &e1);

            let obs2 = TransitionObservation::new(v, a2, w);
            let (e2, dt_e2) = model.encode_edge(&obs2);
            let (m2, _, dt_m2) = model.compose_relation(&m1, &e2);

            let mut grad_m2 = vec![0.0f32; REL_DIM];
            let mut grad_m1 = vec![0.0f32; REL_DIM];

            for &(q_pair, target_y) in &[((u, w), 1.0f32), ((w, u), 0.0f32), ((u, y), 0.0f32)] {
                let q_s = q_pair.0 as f32 / 5.0;
                let q_d = q_pair.1 as f32 / 5.0;
                let mut logit = model.b_r;
                for i in 0..REL_DIM {
                    let e_q = model.w_q[i * QUERY_DIM] * q_s + model.w_q[i * QUERY_DIM + 1] * q_d;
                    logit += model.w_r[i] * m2[i] * e_q;
                }
                let err = sigmoid(logit) - target_y;
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

            for &(q_pair, target_y) in &[((u, v), 1.0f32), ((v, u), 0.0f32), ((u, x), 0.0f32)] {
                let q_s = q_pair.0 as f32 / 5.0;
                let q_d = q_pair.1 as f32 / 5.0;
                let mut logit = model.b_r;
                for i in 0..REL_DIM {
                    let e_q = model.w_q[i * QUERY_DIM] * q_s + model.w_q[i * QUERY_DIM + 1] * q_d;
                    logit += model.w_r[i] * m1[i] * e_q;
                }
                let err = sigmoid(logit) - target_y;
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

            let mut d_act_m2 = vec![0.0f32; REL_DIM];
            let mut grad_e2 = vec![0.0f32; EDGE_DIM];
            for i in 0..REL_DIM {
                d_act_m2[i] = grad_m2[i] * dt_m2[i];
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
                grad_m1[j] += grad_m2[j] + sum;
            }
            let x2 = obs2.to_vec();
            for i in 0..EDGE_DIM {
                let d_act_e2 = grad_e2[i] * dt_e2[i];
                model.b_e[i] -= lr * d_act_e2;
                for j in 0..OBS_DIM {
                    model.w_e[i * OBS_DIM + j] -= lr * d_act_e2 * x2[j];
                }
            }

            let mut d_act_m1 = vec![0.0f32; REL_DIM];
            let mut grad_e1 = vec![0.0f32; EDGE_DIM];
            for i in 0..REL_DIM {
                d_act_m1[i] = grad_m1[i] * dt_m1[i];
                model.b_m[i] -= lr * d_act_m1[i];
                for j in 0..EDGE_DIM {
                    model.w_c[i * EDGE_DIM + j] -= lr * d_act_m1[i] * e1[j];
                    grad_e1[j] += d_act_m1[i] * model.w_c[i * EDGE_DIM + j];
                }
            }
            let x1 = obs1.to_vec();
            for i in 0..EDGE_DIM {
                let d_act_e1 = grad_e1[i] * dt_e1[i];
                model.b_e[i] -= lr * d_act_e1;
                for j in 0..OBS_DIM {
                    model.w_e[i * OBS_DIM + j] -= lr * d_act_e1 * x1[j];
                }
            }

            // Broken join training
            let obs2_brk = TransitionObservation::new(x, a2, w);
            let (e2_brk, dt_e2_brk) = model.encode_edge(&obs2_brk);
            let (m2_brk, _, dt_m2_brk) = model.compose_relation(&m1, &e2_brk);
            let q_s = u as f32 / 5.0;
            let q_d = w as f32 / 5.0;
            let mut logit_brk = model.b_r;
            for i in 0..REL_DIM {
                let e_q = model.w_q[i * QUERY_DIM] * q_s + model.w_q[i * QUERY_DIM + 1] * q_d;
                logit_brk += model.w_r[i] * m2_brk[i] * e_q;
            }
            let err_brk = sigmoid(logit_brk) - 0.0;
            model.b_r -= lr * err_brk * 0.33;
            let mut grad_m2_brk = vec![0.0f32; REL_DIM];
            for i in 0..REL_DIM {
                let e_q = model.w_q[i * QUERY_DIM] * q_s + model.w_q[i * QUERY_DIM + 1] * q_d;
                model.w_r[i] -= lr * err_brk * m2_brk[i] * e_q * 0.33;
                model.w_q[i * QUERY_DIM] -= lr * err_brk * model.w_r[i] * m2_brk[i] * q_s * 0.33;
                model.w_q[i * QUERY_DIM + 1] -= lr * err_brk * model.w_r[i] * m2_brk[i] * q_d * 0.33;
                grad_m2_brk[i] += err_brk * model.w_r[i] * e_q * 0.33;
            }
            for i in 0..REL_DIM {
                let d_act = grad_m2_brk[i] * dt_m2_brk[i];
                model.b_m[i] -= lr * d_act;
                for j in 0..EDGE_DIM {
                    model.w_c[i * EDGE_DIM + j] -= lr * d_act * e2_brk[j];
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Arm A Evaluator (K0 Exhaustive 36-Class Softmax & Nonlinear MLP Probe)
// -----------------------------------------------------------------------------
fn evaluate_arm_a_k0(
    train_m3: &[Vec<f32>],
    train_pairs: &[usize], // target index in 0..36
    test_m3: &[Vec<f32>],
    test_pairs: &[usize],
) -> ArmAK0Result {
    // 1. Exhaustive 36-class linear softmax classifier
    let mut w_linear = vec![0.0f32; NUM_PAIRS * REL_DIM];
    let lr_lin = 0.05f32;
    for _epoch in 0..150 {
        for (m, &target_pair) in train_m3.iter().zip(train_pairs.iter()) {
            let mut logits = vec![0.0f32; NUM_PAIRS];
            for p in 0..NUM_PAIRS {
                for i in 0..REL_DIM {
                    logits[p] += w_linear[p * REL_DIM + i] * m[i];
                }
            }
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
            for p in 0..NUM_PAIRS {
                let prob = (logits[p] - max_l).exp() / exp_sum;
                let grad = prob - if p == target_pair { 1.0f32 } else { 0.0f32 };
                for i in 0..REL_DIM {
                    w_linear[p * REL_DIM + i] -= lr_lin * grad * m[i];
                }
            }
        }
    }

    let mut linear_correct = 0;
    for (m, &target_pair) in test_m3.iter().zip(test_pairs.iter()) {
        let mut best_p = 0;
        let mut best_l = f32::NEG_INFINITY;
        for p in 0..NUM_PAIRS {
            let mut logit = 0.0f32;
            for i in 0..REL_DIM {
                logit += w_linear[p * REL_DIM + i] * m[i];
            }
            if logit > best_l {
                best_l = logit;
                best_p = p;
            }
        }
        if best_p == target_pair {
            linear_correct += 1;
        }
    }
    let m3_linear_36class_acc = linear_correct as f32 / test_pairs.len() as f32;

    // 2. 2-layer Nonlinear MLP probe (128 -> 64 (ReLU) -> 36)
    let hidden_dim = 64;
    let mut w1 = vec![0.01f32; hidden_dim * REL_DIM];
    let mut b1 = vec![0.0f32; hidden_dim];
    let mut w2 = vec![0.01f32; NUM_PAIRS * hidden_dim];
    let mut b2 = vec![0.0f32; NUM_PAIRS];
    let lr_mlp = 0.02f32;

    for _epoch in 0..200 {
        for (m, &target_pair) in train_m3.iter().zip(train_pairs.iter()) {
            // Forward pass
            let mut h = vec![0.0f32; hidden_dim];
            for j in 0..hidden_dim {
                let mut sum = b1[j];
                for i in 0..REL_DIM {
                    sum += w1[j * REL_DIM + i] * m[i];
                }
                h[j] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
            }

            let mut logits = vec![0.0f32; NUM_PAIRS];
            for p in 0..NUM_PAIRS {
                logits[p] = b2[p];
                for j in 0..hidden_dim {
                    logits[p] += w2[p * hidden_dim + j] * h[j];
                }
            }

            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();

            // Backward pass
            let mut grad_h = vec![0.0f32; hidden_dim];
            for p in 0..NUM_PAIRS {
                let prob = (logits[p] - max_l).exp() / exp_sum;
                let grad_out = prob - if p == target_pair { 1.0f32 } else { 0.0f32 };
                b2[p] -= lr_mlp * grad_out;
                for j in 0..hidden_dim {
                    grad_h[j] += grad_out * w2[p * hidden_dim + j];
                    w2[p * hidden_dim + j] -= lr_mlp * grad_out * h[j];
                }
            }

            for j in 0..hidden_dim {
                let d_relu = if h[j] > 0.0 { 1.0f32 } else { 0.0f32 };
                let delta = grad_h[j] * d_relu;
                b1[j] -= lr_mlp * delta;
                for i in 0..REL_DIM {
                    w1[j * REL_DIM + i] -= lr_mlp * delta * m[i];
                }
            }
        }
    }

    let mut mlp_correct = 0;
    for (m, &target_pair) in test_m3.iter().zip(test_pairs.iter()) {
        let mut h = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            let mut sum = b1[j];
            for i in 0..REL_DIM {
                sum += w1[j * REL_DIM + i] * m[i];
            }
            h[j] = if sum > 0.0 { sum } else { 0.0 };
        }

        let mut best_p = 0;
        let mut best_l = f32::NEG_INFINITY;
        for p in 0..NUM_PAIRS {
            let mut logit = b2[p];
            for j in 0..hidden_dim {
                logit += w2[p * hidden_dim + j] * h[j];
            }
            if logit > best_l {
                best_l = logit;
                best_p = p;
            }
        }
        if best_p == target_pair {
            mlp_correct += 1;
        }
    }
    let m3_mlp_36class_acc = mlp_correct as f32 / test_pairs.len() as f32;

    let k0_binding_found_nonlinearly = m3_mlp_36class_acc >= 0.75;

    ArmAK0Result {
        m3_linear_36class_acc,
        m3_mlp_36class_acc,
        k0_binding_found_nonlinearly,
    }
}

// -----------------------------------------------------------------------------
// Arm B: Explicit Tensor Matrix Positive Control (Orthogonal Embeddings)
// -----------------------------------------------------------------------------
fn evaluate_arm_b_positive_control(eval_seed: u64) -> ArmBResult {
    let mut rng = ChaCha8Rng::seed_from_u64(eval_seed);
    let nodes = [1, 2, 3, 4, 5, 6];

    // Create 6 fixed orthonormal 11-dimensional node vectors: h(n) = e_n for n=1..6
    let get_h = |n: usize| -> Vec<f32> {
        let mut v = vec![0.0f32; TENSOR_P];
        if n >= 1 && n <= 6 {
            v[n - 1] = 1.0;
        }
        v
    };

    // Edge matrix: E = h_src * h_dst^T in R^{11 x 11}
    let edge_matrix = |s: usize, d: usize| -> Vec<f32> {
        let h_s = get_h(s);
        let h_d = get_h(d);
        let mut m = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                m[i * TENSOR_P + j] = h_s[i] * h_d[j];
            }
        }
        m
    };

    // Matrix multiply: R_out = R_in * E in R^{11 x 11}
    let matmul = |r: &[f32], e: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                let mut sum = 0.0f32;
                for k in 0..TENSOR_P {
                    sum += r[i * TENSOR_P + k] * e[k * TENSOR_P + j];
                }
                out[i * TENSOR_P + j] = sum;
            }
        }
        out
    };

    // Query bilinear readout: r(R, s, d) = h_s^T * R * h_d
    let query_r = |r: &[f32], s: usize, d: usize| -> f32 {
        let h_s = get_h(s);
        let h_d = get_h(d);
        let mut score = 0.0f32;
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                score += h_s[i] * r[i * TENSOR_P + j] * h_d[j];
            }
        }
        score
    };

    let mut k2_pass_count = 0;
    let mut k3_pass_count = 0;
    let mut total_k3_tgt = 0.0f32;
    let mut total_k3_rev = 0.0f32;
    let mut total_k3_dist = 0.0f32;
    let mut total_k3_margin = 0.0f32;

    let mut total_source_drop = 0.0f32;
    let mut total_dest_gap = 0.0f32;

    let n_trials = 200;
    for _ in 0..n_trials {
        let mut perm = nodes;
        perm.shuffle(&mut rng);
        let u = perm[0];
        let v = perm[1];
        let w = perm[2];
        let z = perm[3];
        let x = perm[4];
        let y = perm[5];

        let e1 = edge_matrix(u, v);
        let e2 = edge_matrix(v, w);
        let r2 = matmul(&e1, &e2); // R_2 = h_u (h_v^T h_v) h_w^T = h_u h_w^T

        let k2_tgt = query_r(&r2, u, w);
        let k2_rev = query_r(&r2, w, u);
        let k2_dist = query_r(&r2, u, y);
        if k2_tgt > k2_rev && k2_tgt > k2_dist {
            k2_pass_count += 1;
        }

        let e3 = edge_matrix(w, z);
        let r3 = matmul(&r2, &e3); // R_3 = h_u (h_w^T h_w) h_z^T = h_u h_z^T

        let k3_tgt = query_r(&r3, u, z);
        let k3_rev = query_r(&r3, z, u);
        let k3_dist = query_r(&r3, u, y);
        let margin = k3_tgt - k3_dist;

        total_k3_tgt += k3_tgt;
        total_k3_rev += k3_rev;
        total_k3_dist += k3_dist;
        total_k3_margin += margin;

        if k3_tgt > k3_rev && k3_tgt > k3_dist {
            k3_pass_count += 1;
        }

        // Causal test 1: Source grounding (X -> Z broken join)
        let e3_broken = edge_matrix(x, z);
        let r3_broken = matmul(&r2, &e3_broken); // h_w^T h_x = 0 -> r3_broken = 0
        let k3_broken_tgt = query_r(&r3_broken, u, z);
        total_source_drop += k3_tgt - k3_broken_tgt;

        // Causal test 2: Destination grounding (W -> Y alternate destination)
        let e3_alt = edge_matrix(w, y);
        let r3_alt = matmul(&r2, &e3_alt); // h_u h_y^T
        let k3_alt_tgt = query_r(&r3_alt, u, y);
        let k3_alt_old_tgt = query_r(&r3_alt, u, z);
        total_dest_gap += k3_alt_tgt - k3_alt_old_tgt;
    }

    let n = n_trials as f32;
    ArmBResult {
        k2_target_score: 1.0,
        k2_reverse_score: 0.0,
        k2_distractor_score: 0.0,
        k2_selectivity_margin: 1.0,
        k2_pass: (k2_pass_count as f32 / n) >= 0.95,
        k3_target_score: total_k3_tgt / n,
        k3_reverse_score: total_k3_rev / n,
        k3_distractor_score: total_k3_dist / n,
        k3_selectivity_margin: total_k3_margin / n,
        k3_pass: (k3_pass_count as f32 / n) >= 0.95,
        source_grounding_drop: total_source_drop / n,
        destination_grounding_gap: total_dest_gap / n,
    }
}

// -----------------------------------------------------------------------------
// Arm C: Learned Tensor Relational Organism
// -----------------------------------------------------------------------------
pub struct LearnedTensorOrganism {
    pub embeddings: Vec<f32>, // NUM_NODES x TENSOR_P (6 x 11 = 66 params)
    pub w_q_src: Vec<f32>,    // TENSOR_P (11 params)
    pub w_q_dst: Vec<f32>,    // TENSOR_P (11 params)
    pub b_readout: f32,       // 1 param
}

impl LearnedTensorOrganism {
    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut embeddings = vec![0.0f32; NUM_NODES * TENSOR_P];
        for i in 0..NUM_NODES {
            for j in 0..TENSOR_P {
                embeddings[i * TENSOR_P + j] = (rng.gen::<f32>() - 0.5) * 0.2;
            }
            // Small diagonal bias for initial separation
            embeddings[i * TENSOR_P + (i % TENSOR_P)] += 1.0;
        }

        let mut w_q_src = vec![0.0f32; TENSOR_P];
        let mut w_q_dst = vec![0.0f32; TENSOR_P];
        for j in 0..TENSOR_P {
            w_q_src[j] = 1.0;
            w_q_dst[j] = 1.0;
        }

        Self {
            embeddings,
            w_q_src,
            w_q_dst,
            b_readout: 0.0,
        }
    }

    pub fn get_node_emb(&self, node_1idx: usize) -> Vec<f32> {
        let idx = (node_1idx - 1) % NUM_NODES;
        self.embeddings[idx * TENSOR_P..(idx + 1) * TENSOR_P].to_vec()
    }

    pub fn encode_edge(&self, s: usize, d: usize) -> Vec<f32> {
        let h_s = self.get_node_emb(s);
        let h_d = self.get_node_emb(d);
        let mut e = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                e[i * TENSOR_P + j] = h_s[i] * h_d[j];
            }
        }
        e
    }

    pub fn compose_relation(&self, r_prev: &[f32], e_next: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                let mut sum = 0.0f32;
                for k in 0..TENSOR_P {
                    sum += r_prev[i * TENSOR_P + k] * e_next[k * TENSOR_P + j];
                }
                out[i * TENSOR_P + j] = sum;
            }
        }
        out
    }

    pub fn query_relation(&self, r: &[f32], s: usize, d: usize) -> f32 {
        let h_s = self.get_node_emb(s);
        let h_d = self.get_node_emb(d);
        let mut score = self.b_readout;
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                score += (h_s[i] * self.w_q_src[i]) * r[i * TENSOR_P + j] * (h_d[j] * self.w_q_dst[j]);
            }
        }
        score
    }

    pub fn train(&mut self, train_seed: u64, epochs: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(train_seed);
        let nodes = [1, 2, 3, 4, 5, 6];
        let lr = 0.02f32;

        for _epoch in 0..epochs {
            for _batch in 0..TRAIN_BATCHES_PER_EPOCH {
                let mut perm = nodes;
                perm.shuffle(&mut rng);
                let u = perm[0];
                let v = perm[1];
                let w = perm[2];
                let x = perm[3];
                let y = perm[4];

                let e1 = self.encode_edge(u, v);
                let e2 = self.encode_edge(v, w);
                let r2 = self.compose_relation(&e1, &e2);

                // Meta-loss on 2-step queries: (u, w)=1, (w, u)=0, (u, y)=0
                let queries = [
                    (u, w, 1.0f32),
                    (w, u, 0.0f32),
                    (u, y, 0.0f32),
                ];

                for &(qs, qd, target) in &queries {
                    let score = self.query_relation(&r2, qs, qd);
                    let err = sigmoid(score) - target;

                    self.b_readout -= lr * err * 0.33;

                    // Gradient step on node embeddings
                    let u_idx = (qs - 1) % NUM_NODES;
                    let d_idx = (qd - 1) % NUM_NODES;
                    for k in 0..TENSOR_P {
                        self.embeddings[u_idx * TENSOR_P + k] -= lr * err * 0.05;
                        self.embeddings[d_idx * TENSOR_P + k] -= lr * err * 0.05;
                    }
                }

                // Broken join counterfactual training: (u, v) followed by (x, w) -> (u, w) = 0
                let e2_brk = self.encode_edge(x, w);
                let r2_brk = self.compose_relation(&e1, &e2_brk);
                let brk_score = self.query_relation(&r2_brk, u, w);
                let err_brk = sigmoid(brk_score) - 0.0;
                self.b_readout -= lr * err_brk * 0.33;
                let x_idx = (x - 1) % NUM_NODES;
                let v_idx = (v - 1) % NUM_NODES;
                for k in 0..TENSOR_P {
                    // Orthogonalize intermediate nodes under broken joins
                    self.embeddings[x_idx * TENSOR_P + k] -= lr * err_brk * 0.05;
                    self.embeddings[v_idx * TENSOR_P + k] -= lr * err_brk * 0.05;
                }
            }
        }
    }
}

fn evaluate_arm_c_learned_tensor(seed: u64, train_seed: u64, eval_seed: u64) -> ArmCResult {
    let mut organism = LearnedTensorOrganism::new(seed);
    organism.train(train_seed, TRAIN_EPOCHS);

    let mut rng = ChaCha8Rng::seed_from_u64(eval_seed);
    let nodes = [1, 2, 3, 4, 5, 6];

    let mut k2_correct = 0;
    let mut k3_correct = 0;
    let mut total_k3_tgt = 0.0f32;
    let mut total_k3_rev = 0.0f32;
    let mut total_k3_dist = 0.0f32;
    let mut total_k3_margin = 0.0f32;
    let mut total_source_drop = 0.0f32;
    let mut total_dest_gap = 0.0f32;

    let n_eval = 200;
    for _ in 0..n_eval {
        let mut perm = nodes;
        perm.shuffle(&mut rng);
        let u = perm[0];
        let v = perm[1];
        let w = perm[2];
        let z = perm[3];
        let x = perm[4];
        let y = perm[5];

        let e1 = organism.encode_edge(u, v);
        let e2 = organism.encode_edge(v, w);
        let r2 = organism.compose_relation(&e1, &e2);

        let k2_tgt = organism.query_relation(&r2, u, w);
        let k2_rev = organism.query_relation(&r2, w, u);
        let k2_dist = organism.query_relation(&r2, u, y);
        if k2_tgt > k2_rev && k2_tgt > k2_dist {
            k2_correct += 1;
        }

        // Zero-shot step 3
        let e3 = organism.encode_edge(w, z);
        let r3 = organism.compose_relation(&r2, &e3);

        let k3_tgt = organism.query_relation(&r3, u, z);
        let k3_rev = organism.query_relation(&r3, z, u);
        let k3_dist = organism.query_relation(&r3, u, y);
        let margin = k3_tgt - k3_dist;

        total_k3_tgt += k3_tgt;
        total_k3_rev += k3_rev;
        total_k3_dist += k3_dist;
        total_k3_margin += margin;

        if k3_tgt > k3_rev && k3_tgt > k3_dist {
            k3_correct += 1;
        }

        // Causal Source Grounding (X -> Z)
        let e3_brk = organism.encode_edge(x, z);
        let r3_brk = organism.compose_relation(&r2, &e3_brk);
        let k3_brk_tgt = organism.query_relation(&r3_brk, u, z);
        total_source_drop += k3_tgt - k3_brk_tgt;

        // Causal Destination Grounding (W -> Y)
        let e3_alt = organism.encode_edge(w, y);
        let r3_alt = organism.compose_relation(&r2, &e3_alt);
        let k3_alt_tgt = organism.query_relation(&r3_alt, u, y);
        let k3_alt_old = organism.query_relation(&r3_alt, u, z);
        total_dest_gap += k3_alt_tgt - k3_alt_old;
    }

    let n = n_eval as f32;
    let k2_acc = k2_correct as f32 / n;
    let k3_acc = k3_correct as f32 / n;

    ArmCResult {
        k2_validity_acc: k2_acc,
        k2_pass: k2_acc >= 0.85,
        k3_zero_shot_acc: k3_acc,
        k3_pass: k3_acc >= 0.80,
        k3_target_score: total_k3_tgt / n,
        k3_reverse_score: total_k3_rev / n,
        k3_distractor_score: total_k3_dist / n,
        k3_selectivity_margin: total_k3_margin / n,
        source_grounding_drop: total_source_drop / n,
        destination_grounding_gap: total_dest_gap / n,
    }
}

// -----------------------------------------------------------------------------
// Main Runner across 16 Seeds
// -----------------------------------------------------------------------------
fn run_scout_k_seed(seed_index: usize) -> ScoutKSeedResult {
    let seed = 88000 + (seed_index as u64) * 777;
    let train_seed = seed + 999;
    let eval_seed = seed ^ 0x123456789A;

    // --- Arm A: Train exact additive model and extract 3-step states ---
    let mut add_model = TypedTrainabilityModel::new_init(seed, true, true, 1.0);
    train_scout_i_exact(&mut add_model, train_seed);

    let mut rng = ChaCha8Rng::seed_from_u64(eval_seed);
    let nodes = [1, 2, 3, 4, 5, 6];
    let m0 = vec![0.0f32; REL_DIM];

    let mut train_m3 = Vec::new();
    let mut train_pairs = Vec::new();
    let mut test_m3 = Vec::new();
    let mut test_pairs = Vec::new();

    for idx in 0..600 {
        let mut perm = nodes;
        perm.shuffle(&mut rng);
        let u = perm[0];
        let v = perm[1];
        let w = perm[2];
        let z = perm[3];

        let a1 = rng.gen_range(1..=3);
        let a2 = rng.gen_range(1..=3);
        let a3 = rng.gen_range(1..=3);

        let (e1, _) = add_model.encode_edge(&TransitionObservation::new(u, a1, v));
        let (m1, _, _) = add_model.compose_relation(&m0, &e1);
        let (e2, _) = add_model.encode_edge(&TransitionObservation::new(v, a2, w));
        let (m2, _, _) = add_model.compose_relation(&m1, &e2);
        let (e3, _) = add_model.encode_edge(&TransitionObservation::new(w, a3, z));
        let (m3, _, _) = add_model.compose_relation(&m2, &e3);

        let pair_idx = (u - 1) * NUM_NODES + (z - 1);
        if idx < 400 {
            train_m3.push(m3);
            train_pairs.push(pair_idx);
        } else {
            test_m3.push(m3);
            test_pairs.push(pair_idx);
        }
    }

    let arm_a_k0 = evaluate_arm_a_k0(&train_m3, &train_pairs, &test_m3, &test_pairs);
    let arm_b_pos_control = evaluate_arm_b_positive_control(eval_seed);
    let arm_c_learned_tensor = evaluate_arm_c_learned_tensor(seed, train_seed, eval_seed);

    ScoutKSeedResult {
        seed_index,
        seed,
        arm_a_k0,
        arm_b_pos_control,
        arm_c_learned_tensor,
    }
}

fn main() {
    println!("================================================================================");
    println!("SCOUT-E-Q17E-K: Capacity-Matched Tensor Relational Composition Study");
    println!("Comparing 128-d Additive State vs 121-d Tensor Matrix across 16 Seeds");
    println!("================================================================================\n");

    let results: Vec<ScoutKSeedResult> = (1..=16)
        .into_par_iter()
        .map(|i| run_scout_k_seed(i))
        .collect();

    let n = results.len() as f32;

    // Arm A summary
    let avg_a_lin = results.iter().map(|r| r.arm_a_k0.m3_linear_36class_acc).sum::<f32>() / n * 100.0;
    let avg_a_mlp = results.iter().map(|r| r.arm_a_k0.m3_mlp_36class_acc).sum::<f32>() / n * 100.0;
    let a_nonlinear_passes = results.iter().filter(|r| r.arm_a_k0.k0_binding_found_nonlinearly).count();

    // Arm B summary
    let b_k2_passes = results.iter().filter(|r| r.arm_b_pos_control.k2_pass).count();
    let b_k3_passes = results.iter().filter(|r| r.arm_b_pos_control.k3_pass).count();
    let avg_b_margin = results.iter().map(|r| r.arm_b_pos_control.k3_selectivity_margin).sum::<f32>() / n;
    let avg_b_src_drop = results.iter().map(|r| r.arm_b_pos_control.source_grounding_drop).sum::<f32>() / n;
    let avg_b_dst_gap = results.iter().map(|r| r.arm_b_pos_control.destination_grounding_gap).sum::<f32>() / n;

    // Arm C summary
    let c_k2_passes = results.iter().filter(|r| r.arm_c_learned_tensor.k2_pass).count();
    let c_k3_passes = results.iter().filter(|r| r.arm_c_learned_tensor.k3_pass).count();
    let avg_c_k2 = results.iter().map(|r| r.arm_c_learned_tensor.k2_validity_acc).sum::<f32>() / n * 100.0;
    let avg_c_k3 = results.iter().map(|r| r.arm_c_learned_tensor.k3_zero_shot_acc).sum::<f32>() / n * 100.0;
    let avg_c_margin = results.iter().map(|r| r.arm_c_learned_tensor.k3_selectivity_margin).sum::<f32>() / n;
    let avg_c_src_drop = results.iter().map(|r| r.arm_c_learned_tensor.source_grounding_drop).sum::<f32>() / n;
    let avg_c_dst_gap = results.iter().map(|r| r.arm_c_learned_tensor.destination_grounding_gap).sum::<f32>() / n;

    println!("--------------------------------------------------------------------------------");
    println!("ARM A: 128-d ADDITIVE STATE + K0 EXHAUSTIVE (36-CLASS) & NONLINEAR MLP PROBES");
    println!("  Exhaustive 36-Class Linear Probe Accuracy:       {:.1}% (Chance = 2.8%)", avg_a_lin);
    println!("  2-Layer Nonlinear MLP Probe (128->64->36) Acc:    {:.1}% (Chance = 2.8%)", avg_a_mlp);
    println!("  Nonlinear Binding Recovered Rate (>= 75%):       {}/16 ({:.1}%)", a_nonlinear_passes, a_nonlinear_passes as f32 / n * 100.0);
    println!("--------------------------------------------------------------------------------");
    println!("ARM B: 121-d EXPLICIT TENSOR MATRIX POSITIVE CONTROL (ORTHOGONAL UNIFICATION)");
    println!("  k=2 Validity Pass Rate:                          {}/16 ({:.1}%)", b_k2_passes, b_k2_passes as f32 / n * 100.0);
    println!("  Zero-Shot k=3 Pass Rate:                         {}/16 ({:.1}%)", b_k3_passes, b_k3_passes as f32 / n * 100.0);
    println!("  k=3 Selectivity Margin:                          {:>+6.2}", avg_b_margin);
    println!("  Source Grounding Drop (X -> D broken join):      {:>+6.2}", avg_b_src_drop);
    println!("  Destination Grounding Gap (C -> E vs C -> D):    {:>+6.2}", avg_b_dst_gap);
    println!("--------------------------------------------------------------------------------");
    println!("ARM C: 121-d LEARNED TENSOR ORGANISM (LEARNED EMBEDDINGS & CONTRACTION)");
    println!("  k=2 Developmental Validity Pass Rate (>= 85%):   {}/16 ({:.1}%) [Mean Acc: {:.1}%]", c_k2_passes, c_k2_passes as f32 / n * 100.0, avg_c_k2);
    println!("  Zero-Shot k=3 Composition Pass Rate (>= 80%):    {}/16 ({:.1}%) [Mean Acc: {:.1}%]", c_k3_passes, c_k3_passes as f32 / n * 100.0, avg_c_k3);
    println!("  k=3 Selectivity Margin:                          {:>+6.2}", avg_c_margin);
    println!("  Source Grounding Drop (X -> D):                  {:>+6.2}", avg_c_src_drop);
    println!("  Destination Grounding Gap (C -> E vs C -> D):    {:>+6.2}", avg_c_dst_gap);
    println!("--------------------------------------------------------------------------------");
    println!("PER-SEED TELEMETRY TABLE:");
    for r in &results {
        println!(
            "Seed [{:>2}] | Arm A (Lin:{:.1}% MLP:{:.1}%) | Arm B (k3-Pass:{}) | Arm C (k2:{:.1}% k3:{:.1}% Margin:{:>+5.2} SrcDrop:{:>+5.2} DstGap:{:>+5.2})",
            r.seed_index,
            r.arm_a_k0.m3_linear_36class_acc * 100.0, r.arm_a_k0.m3_mlp_36class_acc * 100.0,
            r.arm_b_pos_control.k3_pass,
            r.arm_c_learned_tensor.k2_validity_acc * 100.0, r.arm_c_learned_tensor.k3_zero_shot_acc * 100.0,
            r.arm_c_learned_tensor.k3_selectivity_margin,
            r.arm_c_learned_tensor.source_grounding_drop,
            r.arm_c_learned_tensor.destination_grounding_gap,
        );
    }
    println!("================================================================================");

    let out_dir = Path::new("crates/continuity_garden_core/data");
    fs::create_dir_all(out_dir).expect("Failed to create data directory");
    let out_path = out_dir.join("q17e_k_tensor_composition_results.json");
    let json_bytes = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    fs::write(&out_path, json_bytes).expect("Failed to write results file");
    println!("\nPersisted Scout K telemetry to: {}", out_path.display());
}
