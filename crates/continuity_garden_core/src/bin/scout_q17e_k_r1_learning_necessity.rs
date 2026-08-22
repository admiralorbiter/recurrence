//! SCOUT-E-Q17E-K-R1: Learning Necessity & Contraction Specificity Factorization Study
//!
//! Evaluates the necessity of learning vs inductive bias across 5 conditions:
//! - Arm A: 128-d Additive State + Proper Xavier-Initialized 2-Layer MLP Diagnostic Probe.
//! - Arm C0: Untrained Diagonal-Biased Embeddings + Fixed Matrix Multiply (0 Epochs).
//! - Arm C1: Untrained Isotropic Random Embeddings + Fixed Matrix Multiply (0 Epochs).
//! - Arm C2: True Gradient-Trained Embeddings + Fixed Matrix Multiply (Trained on 1- and 2-step).
//! - Arm C3: Learned Bilinear Composition Operator C_theta(R, E) + True Gradient Training.
//! - Arm C_wrong: Wrong Contraction Control (Mismatched index binding algebra).
//!
//! Evaluated across 16 independent seeds with zero 3-hop training labels.

use std::fs;
use std::path::Path;

use continuity_garden_core::typed_model::{
    sigmoid, TransitionObservation, TypedTrainabilityModel, EDGE_DIM, OBS_DIM, QUERY_DIM, REL_DIM,
    TRAIN_BATCHES_PER_EPOCH, TRAIN_EPOCHS, TRAIN_LR,
};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const NUM_NODES: usize = 6;
pub const NUM_PAIRS: usize = NUM_NODES * NUM_NODES; // 36
pub const TENSOR_P: usize = 11;
pub const TENSOR_DIM: usize = TENSOR_P * TENSOR_P; // 121

// -----------------------------------------------------------------------------
// Telemetry Data Structures
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmAResult {
    pub linear_36class_acc: f32,
    pub mlp_xavier_36class_acc: f32,
    pub mlp_binding_recovered: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorArmResult {
    pub name: String,
    pub k2_accuracy: f32,
    pub k2_pass: bool,
    pub k3_accuracy: f32,
    pub k3_pass: bool,
    pub k3_target_score: f32,
    pub k3_reverse_score: f32,
    pub k3_distractor_score: f32,
    pub k3_selectivity_margin: f32,
    pub source_grounding_drop: f32,
    pub destination_grounding_gap: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoutKR1SeedResult {
    pub seed_index: usize,
    pub seed: u64,
    pub arm_a_additive_probe: ArmAResult,
    pub arm_c0_untrained_diag: TensorArmResult,
    pub arm_c1_untrained_isotropic: TensorArmResult,
    pub arm_c2_trained_embeddings: TensorArmResult,
    pub arm_c3_learned_bilinear: TensorArmResult,
    pub arm_c_wrong_contraction: TensorArmResult,
}

// -----------------------------------------------------------------------------
// Helper: Train exact Scout-I additive model for Arm A baseline
// -----------------------------------------------------------------------------
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
            let (e2_brk, _dt_e2_brk) = model.encode_edge(&obs2_brk);
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
// Arm A: Additive Baseline with Independent Xavier-Initialized 2-Layer MLP
// -----------------------------------------------------------------------------
fn evaluate_arm_a_proper_mlp(
    train_m3: &[Vec<f32>],
    train_pairs: &[usize],
    test_m3: &[Vec<f32>],
    test_pairs: &[usize],
    seed: u64,
) -> ArmAResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // 1. Linear Probe
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
    let mut lin_correct = 0;
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
            lin_correct += 1;
        }
    }
    let linear_36class_acc = lin_correct as f32 / test_pairs.len() as f32;

    // 2. 2-Layer MLP with Independent Xavier Initialization
    let hidden_dim = 64;
    let norm_w1 = Normal::new(0.0f32, (2.0f32 / REL_DIM as f32).sqrt()).unwrap();
    let norm_w2 = Normal::new(0.0f32, (2.0f32 / hidden_dim as f32).sqrt()).unwrap();

    let mut w1 = vec![0.0f32; hidden_dim * REL_DIM];
    for val in w1.iter_mut() {
        *val = norm_w1.sample(&mut rng);
    }
    let mut b1 = vec![0.0f32; hidden_dim];

    let mut w2 = vec![0.0f32; NUM_PAIRS * hidden_dim];
    for val in w2.iter_mut() {
        *val = norm_w2.sample(&mut rng);
    }
    let mut b2 = vec![0.0f32; NUM_PAIRS];

    let lr_mlp = 0.02f32;

    for _epoch in 0..250 {
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
    let mlp_xavier_36class_acc = mlp_correct as f32 / test_pairs.len() as f32;
    let mlp_binding_recovered = mlp_xavier_36class_acc >= 0.75;

    ArmAResult {
        linear_36class_acc,
        mlp_xavier_36class_acc,
        mlp_binding_recovered,
    }
}

// -----------------------------------------------------------------------------
// Tensor Evaluation Battery Helper
// -----------------------------------------------------------------------------
fn evaluate_tensor_organism_assay<FCompose, FQuery>(
    name: &str,
    eval_seed: u64,
    compose_fn: FCompose,
    query_fn: FQuery,
) -> TensorArmResult
where
    FCompose: Fn(&[f32], usize, usize) -> Vec<f32>,
    FQuery: Fn(&[f32], usize, usize) -> f32,
{
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

        let r0 = vec![0.0f32; TENSOR_DIM];
        let r1 = compose_fn(&r0, u, v);
        let r2 = compose_fn(&r1, v, w);

        let k2_tgt = query_fn(&r2, u, w);
        let k2_rev = query_fn(&r2, w, u);
        let k2_dist = query_fn(&r2, u, y);
        if k2_tgt > k2_rev && k2_tgt > k2_dist {
            k2_correct += 1;
        }

        // 3-step zero-shot
        let r3 = compose_fn(&r2, w, z);

        let k3_tgt = query_fn(&r3, u, z);
        let k3_rev = query_fn(&r3, z, u);
        let k3_dist = query_fn(&r3, u, y);
        let margin = k3_tgt - k3_dist;

        total_k3_tgt += k3_tgt;
        total_k3_rev += k3_rev;
        total_k3_dist += k3_dist;
        total_k3_margin += margin;

        if k3_tgt > k3_rev && k3_tgt > k3_dist {
            k3_correct += 1;
        }

        // Causal test 1: Source grounding (X -> Z broken join)
        let r3_brk = compose_fn(&r2, x, z);
        let k3_brk_tgt = query_fn(&r3_brk, u, z);
        total_source_drop += k3_tgt - k3_brk_tgt;

        // Causal test 2: Destination grounding (W -> Y alternate destination)
        let r3_alt = compose_fn(&r2, w, y);
        let k3_alt_tgt = query_fn(&r3_alt, u, y);
        let k3_alt_old = query_fn(&r3_alt, u, z);
        total_dest_gap += k3_alt_tgt - k3_alt_old;
    }

    let n = n_eval as f32;
    let k2_acc = k2_correct as f32 / n;
    let k3_acc = k3_correct as f32 / n;

    TensorArmResult {
        name: name.to_string(),
        k2_accuracy: k2_acc,
        k2_pass: k2_acc >= 0.85,
        k3_accuracy: k3_acc,
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
// Arm C0: Untrained Diagonal-Biased Embeddings + Fixed Matrix Multiply (0 Epochs)
// -----------------------------------------------------------------------------
fn evaluate_arm_c0(seed: u64, eval_seed: u64) -> TensorArmResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut embeddings = vec![0.0f32; NUM_NODES * TENSOR_P];
    for i in 0..NUM_NODES {
        for j in 0..TENSOR_P {
            embeddings[i * TENSOR_P + j] = (rng.gen::<f32>() - 0.5) * 0.1;
        }
        embeddings[i * TENSOR_P + (i % TENSOR_P)] += 1.0;
    }

    let get_h = |n: usize| -> Vec<f32> {
        let idx = (n - 1) % NUM_NODES;
        embeddings[idx * TENSOR_P..(idx + 1) * TENSOR_P].to_vec()
    };

    let compose_fn = |r_prev: &[f32], s: usize, d: usize| -> Vec<f32> {
        let h_s = get_h(s);
        let h_d = get_h(d);
        let mut e = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                e[i * TENSOR_P + j] = h_s[i] * h_d[j];
            }
        }
        if r_prev.iter().all(|&v| v == 0.0) {
            return e;
        }
        let mut out = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                let mut sum = 0.0f32;
                for k in 0..TENSOR_P {
                    sum += r_prev[i * TENSOR_P + k] * e[k * TENSOR_P + j];
                }
                out[i * TENSOR_P + j] = sum;
            }
        }
        out
    };

    let query_fn = |r: &[f32], s: usize, d: usize| -> f32 {
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

    evaluate_tensor_organism_assay("C0_untrained_diag", eval_seed, compose_fn, query_fn)
}

// -----------------------------------------------------------------------------
// Arm C1: Untrained Isotropic Random Embeddings + Fixed Matrix Multiply (0 Epochs)
// -----------------------------------------------------------------------------
fn evaluate_arm_c1(seed: u64, eval_seed: u64) -> TensorArmResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let norm_dist = Normal::new(0.0f32, 1.0f32).unwrap();
    let mut embeddings = vec![0.0f32; NUM_NODES * TENSOR_P];
    for i in 0..NUM_NODES {
        let mut norm_sq = 0.0f32;
        for j in 0..TENSOR_P {
            let val = norm_dist.sample(&mut rng);
            embeddings[i * TENSOR_P + j] = val;
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt().max(1e-6);
        for j in 0..TENSOR_P {
            embeddings[i * TENSOR_P + j] /= norm;
        }
    }

    let get_h = |n: usize| -> Vec<f32> {
        let idx = (n - 1) % NUM_NODES;
        embeddings[idx * TENSOR_P..(idx + 1) * TENSOR_P].to_vec()
    };

    let compose_fn = |r_prev: &[f32], s: usize, d: usize| -> Vec<f32> {
        let h_s = get_h(s);
        let h_d = get_h(d);
        let mut e = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                e[i * TENSOR_P + j] = h_s[i] * h_d[j];
            }
        }
        if r_prev.iter().all(|&v| v == 0.0) {
            return e;
        }
        let mut out = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                let mut sum = 0.0f32;
                for k in 0..TENSOR_P {
                    sum += r_prev[i * TENSOR_P + k] * e[k * TENSOR_P + j];
                }
                out[i * TENSOR_P + j] = sum;
            }
        }
        out
    };

    let query_fn = |r: &[f32], s: usize, d: usize| -> f32 {
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

    evaluate_tensor_organism_assay("C1_untrained_isotropic", eval_seed, compose_fn, query_fn)
}

// -----------------------------------------------------------------------------
// Arm C2: True Gradient-Trained Embeddings + Fixed Matrix Multiply
// -----------------------------------------------------------------------------
fn evaluate_arm_c2(seed: u64, train_seed: u64, eval_seed: u64) -> TensorArmResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let norm_dist = Normal::new(0.0f32, 1.0f32).unwrap();
    let mut embeddings = vec![0.0f32; NUM_NODES * TENSOR_P];
    for i in 0..NUM_NODES {
        let mut norm_sq = 0.0f32;
        for j in 0..TENSOR_P {
            let val = norm_dist.sample(&mut rng);
            embeddings[i * TENSOR_P + j] = val;
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt().max(1e-6);
        for j in 0..TENSOR_P {
            embeddings[i * TENSOR_P + j] /= norm;
        }
    }

    // Train embeddings with true backpropagation through contraction algebra
    let mut train_rng = ChaCha8Rng::seed_from_u64(train_seed);
    let nodes = [1, 2, 3, 4, 5, 6];
    let lr = 0.05f32;

    let dot = |h: &[f32], a: usize, b: usize| -> f32 {
        let a_idx = (a - 1) % NUM_NODES;
        let b_idx = (b - 1) % NUM_NODES;
        let mut sum = 0.0f32;
        for k in 0..TENSOR_P {
            sum += h[a_idx * TENSOR_P + k] * h[b_idx * TENSOR_P + k];
        }
        sum
    };

    for _epoch in 0..120 {
        for _batch in 0..TRAIN_BATCHES_PER_EPOCH {
            let mut perm = nodes;
            perm.shuffle(&mut train_rng);
            let u = perm[0];
            let v = perm[1];
            let w = perm[2];
            let x = perm[3];
            let y = perm[4];

            // 1. Positive 2-hop query (u, w): S = ||u||^2 * ||v||^2 * ||w||^2
            let dot_uu = dot(&embeddings, u, u);
            let dot_vv = dot(&embeddings, v, v);
            let dot_ww = dot(&embeddings, w, w);
            let s_pos = dot_uu * dot_vv * dot_ww;
            let err_pos = sigmoid(s_pos) - 1.0f32;

            let u_idx = (u - 1) % NUM_NODES;
            let v_idx = (v - 1) % NUM_NODES;
            let w_idx = (w - 1) % NUM_NODES;

            for k in 0..TENSOR_P {
                let grad_u = err_pos * 2.0 * dot_vv * dot_ww * embeddings[u_idx * TENSOR_P + k];
                let grad_v = err_pos * 2.0 * dot_uu * dot_ww * embeddings[v_idx * TENSOR_P + k];
                let grad_w = err_pos * 2.0 * dot_uu * dot_vv * embeddings[w_idx * TENSOR_P + k];

                embeddings[u_idx * TENSOR_P + k] -= lr * grad_u * 0.33;
                embeddings[v_idx * TENSOR_P + k] -= lr * grad_v * 0.33;
                embeddings[w_idx * TENSOR_P + k] -= lr * grad_w * 0.33;
            }

            // 2. Distractor query (u, y): S = ||u||^2 * ||v||^2 * (w . y)
            let dot_wy = dot(&embeddings, w, y);
            let s_dist = dot_uu * dot_vv * dot_wy;
            let err_dist = sigmoid(s_dist) - 0.0f32;
            let y_idx = (y - 1) % NUM_NODES;

            for k in 0..TENSOR_P {
                let grad_w = err_dist * dot_uu * dot_vv * embeddings[y_idx * TENSOR_P + k];
                let grad_y = err_dist * dot_uu * dot_vv * embeddings[w_idx * TENSOR_P + k];
                embeddings[w_idx * TENSOR_P + k] -= lr * grad_w * 0.33;
                embeddings[y_idx * TENSOR_P + k] -= lr * grad_y * 0.33;
            }

            // 3. Reversal query (w, u): S = (w . u)^2 * ||v||^2
            let dot_wu = dot(&embeddings, w, u);
            let s_rev = dot_wu * dot_wu * dot_vv;
            let err_rev = sigmoid(s_rev) - 0.0f32;

            for k in 0..TENSOR_P {
                let grad_u = err_rev * 2.0 * dot_wu * dot_vv * embeddings[w_idx * TENSOR_P + k];
                let grad_w = err_rev * 2.0 * dot_wu * dot_vv * embeddings[u_idx * TENSOR_P + k];
                embeddings[u_idx * TENSOR_P + k] -= lr * grad_u * 0.33;
                embeddings[w_idx * TENSOR_P + k] -= lr * grad_w * 0.33;
            }

            // 4. Broken join: (u -> v) then (x -> w) => S = ||u||^2 * (v . x) * ||w||^2
            let dot_vx = dot(&embeddings, v, x);
            let s_brk = dot_uu * dot_vx * dot_ww;
            let err_brk = sigmoid(s_brk) - 0.0f32;
            let x_idx = (x - 1) % NUM_NODES;

            for k in 0..TENSOR_P {
                let grad_v = err_brk * dot_uu * dot_ww * embeddings[x_idx * TENSOR_P + k];
                let grad_x = err_brk * dot_uu * dot_ww * embeddings[v_idx * TENSOR_P + k];
                embeddings[v_idx * TENSOR_P + k] -= lr * grad_v * 0.33;
                embeddings[x_idx * TENSOR_P + k] -= lr * grad_x * 0.33;
            }
        }
    }

    let get_h = |n: usize| -> Vec<f32> {
        let idx = (n - 1) % NUM_NODES;
        embeddings[idx * TENSOR_P..(idx + 1) * TENSOR_P].to_vec()
    };

    let compose_fn = |r_prev: &[f32], s: usize, d: usize| -> Vec<f32> {
        let h_s = get_h(s);
        let h_d = get_h(d);
        let mut e = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                e[i * TENSOR_P + j] = h_s[i] * h_d[j];
            }
        }
        if r_prev.iter().all(|&v| v == 0.0) {
            return e;
        }
        let mut out = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                let mut sum = 0.0f32;
                for k in 0..TENSOR_P {
                    sum += r_prev[i * TENSOR_P + k] * e[k * TENSOR_P + j];
                }
                out[i * TENSOR_P + j] = sum;
            }
        }
        out
    };

    let query_fn = |r: &[f32], s: usize, d: usize| -> f32 {
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

    evaluate_tensor_organism_assay("C2_trained_embeddings", eval_seed, compose_fn, query_fn)
}

// -----------------------------------------------------------------------------
// Arm C3: Learned Bilinear Composition Operator C_theta(R, E)
// -----------------------------------------------------------------------------
fn evaluate_arm_c3(seed: u64, train_seed: u64, eval_seed: u64) -> TensorArmResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let norm_dist = Normal::new(0.0f32, 1.0f32).unwrap();
    let mut embeddings = vec![0.0f32; NUM_NODES * TENSOR_P];
    for i in 0..NUM_NODES {
        let mut norm_sq = 0.0f32;
        for j in 0..TENSOR_P {
            let val = norm_dist.sample(&mut rng);
            embeddings[i * TENSOR_P + j] = val;
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt().max(1e-6);
        for j in 0..TENSOR_P {
            embeddings[i * TENSOR_P + j] /= norm;
        }
    }

    // Parameterized bilinear contraction weights: W_k for k=0..TENSOR_P (initialized with small random perturbations around 1.0)
    let mut w_contract = vec![1.0f32; TENSOR_P];
    for val in w_contract.iter_mut() {
        *val += (rng.gen::<f32>() - 0.5) * 0.1;
    }

    let mut train_rng = ChaCha8Rng::seed_from_u64(train_seed);
    let nodes = [1, 2, 3, 4, 5, 6];
    let lr = 0.03f32;

    for _epoch in 0..120 {
        for _batch in 0..TRAIN_BATCHES_PER_EPOCH {
            let mut perm = nodes;
            perm.shuffle(&mut train_rng);
            let u = perm[0];
            let v = perm[1];
            let w = perm[2];
            let x = perm[3];
            let y = perm[4];

            let u_idx = (u - 1) % NUM_NODES;
            let v_idx = (v - 1) % NUM_NODES;
            let w_idx = (w - 1) % NUM_NODES;
            let y_idx = (y - 1) % NUM_NODES;
            let x_idx = (x - 1) % NUM_NODES;

            // Compute R2 = C_theta(E1, E2): R2_ij = sum_k W_k E1_ik E2_kj = h_u_i (sum_k W_k h_v_k^2) h_w_j
            let mut contract_v = 0.0f32;
            for k in 0..TENSOR_P {
                contract_v += w_contract[k] * embeddings[v_idx * TENSOR_P + k] * embeddings[v_idx * TENSOR_P + k];
            }

            let mut dot_uu = 0.0f32;
            let mut dot_ww = 0.0f32;
            for k in 0..TENSOR_P {
                dot_uu += embeddings[u_idx * TENSOR_P + k] * embeddings[u_idx * TENSOR_P + k];
                dot_ww += embeddings[w_idx * TENSOR_P + k] * embeddings[w_idx * TENSOR_P + k];
            }

            // 1. Positive 2-hop query (u, w): S = dot_uu * contract_v * dot_ww
            let s_pos = dot_uu * contract_v * dot_ww;
            let err_pos = sigmoid(s_pos) - 1.0f32;

            for k in 0..TENSOR_P {
                let grad_w_k = err_pos * dot_uu * dot_ww * (embeddings[v_idx * TENSOR_P + k] * embeddings[v_idx * TENSOR_P + k]);
                w_contract[k] -= lr * grad_w_k * 0.33;

                let grad_u = err_pos * 2.0 * contract_v * dot_ww * embeddings[u_idx * TENSOR_P + k];
                let grad_v = err_pos * 2.0 * dot_uu * dot_ww * w_contract[k] * embeddings[v_idx * TENSOR_P + k];
                let grad_w = err_pos * 2.0 * dot_uu * contract_v * embeddings[w_idx * TENSOR_P + k];

                embeddings[u_idx * TENSOR_P + k] -= lr * grad_u * 0.33;
                embeddings[v_idx * TENSOR_P + k] -= lr * grad_v * 0.33;
                embeddings[w_idx * TENSOR_P + k] -= lr * grad_w * 0.33;
            }

            // 2. Distractor query (u, y): S = dot_uu * contract_v * dot_wy
            let mut dot_wy = 0.0f32;
            for k in 0..TENSOR_P {
                dot_wy += embeddings[w_idx * TENSOR_P + k] * embeddings[y_idx * TENSOR_P + k];
            }
            let s_dist = dot_uu * contract_v * dot_wy;
            let err_dist = sigmoid(s_dist) - 0.0f32;

            for k in 0..TENSOR_P {
                let grad_w = err_dist * dot_uu * contract_v * embeddings[y_idx * TENSOR_P + k];
                let grad_y = err_dist * dot_uu * contract_v * embeddings[w_idx * TENSOR_P + k];
                embeddings[w_idx * TENSOR_P + k] -= lr * grad_w * 0.33;
                embeddings[y_idx * TENSOR_P + k] -= lr * grad_y * 0.33;
            }

            // 3. Broken join: (u -> v) then (x -> w) => S = dot_uu * (sum_k W_k h_v_k h_x_k) * dot_ww
            let mut contract_vx = 0.0f32;
            for k in 0..TENSOR_P {
                contract_vx += w_contract[k] * embeddings[v_idx * TENSOR_P + k] * embeddings[x_idx * TENSOR_P + k];
            }
            let s_brk = dot_uu * contract_vx * dot_ww;
            let err_brk = sigmoid(s_brk) - 0.0f32;

            for k in 0..TENSOR_P {
                let grad_w_k = err_brk * dot_uu * dot_ww * (embeddings[v_idx * TENSOR_P + k] * embeddings[x_idx * TENSOR_P + k]);
                w_contract[k] -= lr * grad_w_k * 0.33;

                let grad_v = err_brk * dot_uu * dot_ww * w_contract[k] * embeddings[x_idx * TENSOR_P + k];
                let grad_x = err_brk * dot_uu * dot_ww * w_contract[k] * embeddings[v_idx * TENSOR_P + k];
                embeddings[v_idx * TENSOR_P + k] -= lr * grad_v * 0.33;
                embeddings[x_idx * TENSOR_P + k] -= lr * grad_x * 0.33;
            }
        }
    }

    let get_h = |n: usize| -> Vec<f32> {
        let idx = (n - 1) % NUM_NODES;
        embeddings[idx * TENSOR_P..(idx + 1) * TENSOR_P].to_vec()
    };

    let w_weights = w_contract.clone();

    let compose_fn = move |r_prev: &[f32], s: usize, d: usize| -> Vec<f32> {
        let h_s = get_h(s);
        let h_d = get_h(d);
        let mut e = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                e[i * TENSOR_P + j] = h_s[i] * h_d[j];
            }
        }
        if r_prev.iter().all(|&v| v == 0.0) {
            return e;
        }
        let mut out = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                let mut sum = 0.0f32;
                for k in 0..TENSOR_P {
                    sum += r_prev[i * TENSOR_P + k] * w_weights[k] * e[k * TENSOR_P + j];
                }
                out[i * TENSOR_P + j] = sum;
            }
        }
        out
    };

    let query_fn = move |r: &[f32], s: usize, d: usize| -> f32 {
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

    evaluate_tensor_organism_assay("C3_learned_bilinear", eval_seed, compose_fn, query_fn)
}

// -----------------------------------------------------------------------------
// Arm C_wrong: Wrong Contraction Control (Source-Source Mismatched Binding)
// -----------------------------------------------------------------------------
fn evaluate_arm_c_wrong(seed: u64, eval_seed: u64) -> TensorArmResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut embeddings = vec![0.0f32; NUM_NODES * TENSOR_P];
    for i in 0..NUM_NODES {
        for j in 0..TENSOR_P {
            embeddings[i * TENSOR_P + j] = (rng.gen::<f32>() - 0.5) * 0.1;
        }
        embeddings[i * TENSOR_P + (i % TENSOR_P)] += 1.0;
    }

    let get_h = |n: usize| -> Vec<f32> {
        let idx = (n - 1) % NUM_NODES;
        embeddings[idx * TENSOR_P..(idx + 1) * TENSOR_P].to_vec()
    };

    // Wrong contraction: R'_ij = sum_k R_ki * E_kj (mismatched index contraction)
    let compose_fn = |r_prev: &[f32], s: usize, d: usize| -> Vec<f32> {
        let h_s = get_h(s);
        let h_d = get_h(d);
        let mut e = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                e[i * TENSOR_P + j] = h_s[i] * h_d[j];
            }
        }
        if r_prev.iter().all(|&v| v == 0.0) {
            return e;
        }
        let mut out = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                let mut sum = 0.0f32;
                for k in 0..TENSOR_P {
                    sum += r_prev[k * TENSOR_P + i] * e[k * TENSOR_P + j];
                }
                out[i * TENSOR_P + j] = sum;
            }
        }
        out
    };

    let query_fn = |r: &[f32], s: usize, d: usize| -> f32 {
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

    evaluate_tensor_organism_assay("C_wrong_contraction", eval_seed, compose_fn, query_fn)
}

// -----------------------------------------------------------------------------
// Seed Runner
// -----------------------------------------------------------------------------
fn run_scout_k_r1_seed(seed_index: usize) -> ScoutKR1SeedResult {
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

    let arm_a_additive_probe = evaluate_arm_a_proper_mlp(&train_m3, &train_pairs, &test_m3, &test_pairs, seed);
    let arm_c0_untrained_diag = evaluate_arm_c0(seed, eval_seed);
    let arm_c1_untrained_isotropic = evaluate_arm_c1(seed, eval_seed);
    let arm_c2_trained_embeddings = evaluate_arm_c2(seed, train_seed, eval_seed);
    let arm_c3_learned_bilinear = evaluate_arm_c3(seed, train_seed, eval_seed);
    let arm_c_wrong_contraction = evaluate_arm_c_wrong(seed, eval_seed);

    ScoutKR1SeedResult {
        seed_index,
        seed,
        arm_a_additive_probe,
        arm_c0_untrained_diag,
        arm_c1_untrained_isotropic,
        arm_c2_trained_embeddings,
        arm_c3_learned_bilinear,
        arm_c_wrong_contraction,
    }
}

fn main() {
    println!("================================================================================");
    println!("SCOUT-E-Q17E-K-R1: Learning Necessity & Contraction Specificity Factorization");
    println!("Comparing 5 Tensor Factorization Conditions + Additive Baseline across 16 Seeds");
    println!("================================================================================\n");

    let results: Vec<ScoutKR1SeedResult> = (1..=16)
        .into_par_iter()
        .map(|i| run_scout_k_r1_seed(i))
        .collect();

    let n = results.len() as f32;

    // Arm A
    let avg_a_lin = results.iter().map(|r| r.arm_a_additive_probe.linear_36class_acc).sum::<f32>() / n * 100.0;
    let avg_a_mlp = results.iter().map(|r| r.arm_a_additive_probe.mlp_xavier_36class_acc).sum::<f32>() / n * 100.0;
    let a_mlp_passes = results.iter().filter(|r| r.arm_a_additive_probe.mlp_binding_recovered).count();

    // Arm summaries
    let summarize_tensor = |arms: Vec<&TensorArmResult>| -> (f32, usize, f32, usize, f32, f32, f32) {
        let k2_acc = arms.iter().map(|a| a.k2_accuracy).sum::<f32>() / n * 100.0;
        let k2_p = arms.iter().filter(|a| a.k2_pass).count();
        let k3_acc = arms.iter().map(|a| a.k3_accuracy).sum::<f32>() / n * 100.0;
        let k3_p = arms.iter().filter(|a| a.k3_pass).count();
        let margin = arms.iter().map(|a| a.k3_selectivity_margin).sum::<f32>() / n;
        let src_drop = arms.iter().map(|a| a.source_grounding_drop).sum::<f32>() / n;
        let dst_gap = arms.iter().map(|a| a.destination_grounding_gap).sum::<f32>() / n;
        (k2_acc, k2_p, k3_acc, k3_p, margin, src_drop, dst_gap)
    };

    let (c0_k2, c0_k2_p, c0_k3, c0_k3_p, c0_margin, c0_src, c0_dst) =
        summarize_tensor(results.iter().map(|r| &r.arm_c0_untrained_diag).collect());
    let (c1_k2, c1_k2_p, c1_k3, c1_k3_p, c1_margin, c1_src, c1_dst) =
        summarize_tensor(results.iter().map(|r| &r.arm_c1_untrained_isotropic).collect());
    let (c2_k2, c2_k2_p, c2_k3, c2_k3_p, c2_margin, c2_src, c2_dst) =
        summarize_tensor(results.iter().map(|r| &r.arm_c2_trained_embeddings).collect());
    let (c3_k2, c3_k2_p, c3_k3, c3_k3_p, c3_margin, c3_src, c3_dst) =
        summarize_tensor(results.iter().map(|r| &r.arm_c3_learned_bilinear).collect());
    let (cw_k2, cw_k2_p, cw_k3, cw_k3_p, cw_margin, cw_src, cw_dst) =
        summarize_tensor(results.iter().map(|r| &r.arm_c_wrong_contraction).collect());

    println!("--------------------------------------------------------------------------------");
    println!("ARM A: 128-d ADDITIVE STATE + PROPER XAVIER-INITIALIZED 2-LAYER MLP PROBE");
    println!("  Exhaustive 36-Class Linear Probe Accuracy:       {:.1}% (Chance = 2.8%)", avg_a_lin);
    println!("  2-Layer Xavier-Init MLP Probe (128->64->36) Acc:  {:.1}% (Chance = 2.8%)", avg_a_mlp);
    println!("  Nonlinear Binding Recovered Rate (>= 75%):       {}/16 ({:.1}%)", a_mlp_passes, a_mlp_passes as f32 / n * 100.0);
    println!("--------------------------------------------------------------------------------");
    println!("ARM C0: UNTRAINED DIAGONAL-BIASED EMBEDDINGS + FIXED MATRIX MULTIPLY (0 EPOCHS)");
    println!("  k=2 Validity Pass Rate:                          {}/16 ({:.1}%) [Mean: {:.1}%]", c0_k2_p, c0_k2_p as f32 / n * 100.0, c0_k2);
    println!("  Zero-Shot k=3 Pass Rate:                         {}/16 ({:.1}%) [Mean: {:.1}%]", c0_k3_p, c0_k3_p as f32 / n * 100.0, c0_k3);
    println!("  k=3 Margin: {:>+5.2} | SrcDrop: {:>+5.2} | DstGap: {:>+5.2}", c0_margin, c0_src, c0_dst);
    println!("--------------------------------------------------------------------------------");
    println!("ARM C1: UNTRAINED ISOTROPIC RANDOM EMBEDDINGS + FIXED MATRIX MULTIPLY (0 EPOCHS)");
    println!("  k=2 Validity Pass Rate:                          {}/16 ({:.1}%) [Mean: {:.1}%]", c1_k2_p, c1_k2_p as f32 / n * 100.0, c1_k2);
    println!("  Zero-Shot k=3 Pass Rate:                         {}/16 ({:.1}%) [Mean: {:.1}%]", c1_k3_p, c1_k3_p as f32 / n * 100.0, c1_k3);
    println!("  k=3 Margin: {:>+5.2} | SrcDrop: {:>+5.2} | DstGap: {:>+5.2}", c1_margin, c1_src, c1_dst);
    println!("--------------------------------------------------------------------------------");
    println!("ARM C2: TRUE GRADIENT-TRAINED EMBEDDINGS + FIXED MATRIX MULTIPLY (1- & 2-STEP ONLY)");
    println!("  k=2 Developmental Validity Pass Rate:            {}/16 ({:.1}%) [Mean: {:.1}%]", c2_k2_p, c2_k2_p as f32 / n * 100.0, c2_k2);
    println!("  Zero-Shot k=3 Pass Rate:                         {}/16 ({:.1}%) [Mean: {:.1}%]", c2_k3_p, c2_k3_p as f32 / n * 100.0, c2_k3);
    println!("  k=3 Margin: {:>+5.2} | SrcDrop: {:>+5.2} | DstGap: {:>+5.2}", c2_margin, c2_src, c2_dst);
    println!("--------------------------------------------------------------------------------");
    println!("ARM C3: LEARNED BILINEAR CONTRACTION OPERATOR + TRUE GRADIENT TRAINING (MOONSHOT)");
    println!("  k=2 Developmental Validity Pass Rate:            {}/16 ({:.1}%) [Mean: {:.1}%]", c3_k2_p, c3_k2_p as f32 / n * 100.0, c3_k2);
    println!("  Zero-Shot k=3 Pass Rate:                         {}/16 ({:.1}%) [Mean: {:.1}%]", c3_k3_p, c3_k3_p as f32 / n * 100.0, c3_k3);
    println!("  k=3 Margin: {:>+5.2} | SrcDrop: {:>+5.2} | DstGap: {:>+5.2}", c3_margin, c3_src, c3_dst);
    println!("--------------------------------------------------------------------------------");
    println!("CONTROL: WRONG CONTRACTION (MISMATCHED INDEX CONTRACTION ALGEBRA)");
    println!("  k=2 Validity Pass Rate:                          {}/16 ({:.1}%) [Mean: {:.1}%]", cw_k2_p, cw_k2_p as f32 / n * 100.0, cw_k2);
    println!("  Zero-Shot k=3 Pass Rate:                         {}/16 ({:.1}%) [Mean: {:.1}%]", cw_k3_p, cw_k3_p as f32 / n * 100.0, cw_k3);
    println!("  k=3 Margin: {:>+5.2} | SrcDrop: {:>+5.2} | DstGap: {:>+5.2}", cw_margin, cw_src, cw_dst);
    println!("================================================================================\n");

    let out_dir = Path::new("crates/continuity_garden_core/data");
    fs::create_dir_all(out_dir).expect("Failed to create data directory");
    let out_path = out_dir.join("q17e_k_r1_learning_necessity_results.json");
    let json_bytes = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    fs::write(&out_path, json_bytes).expect("Failed to write results file");
    println!("Persisted Scout K-R1 telemetry to: {}", out_path.display());
}
