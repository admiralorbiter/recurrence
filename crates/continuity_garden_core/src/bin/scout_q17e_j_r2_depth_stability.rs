//! SCOUT-E-Q17E-J-R2: Depth-Stable Relational Semantics & 3-Hop Destination Probing
//!
//! Evaluates whether the 3-hop composition failure in Scout J-R1 is caused by:
//! 1. Destination not written to m3 (absence of incoming endpoint information).
//! 2. Destination present but unbound (fillers survive but relation cannot be addressed).
//! 3. Relational semantics drift with depth (relation exists in m3, but subspace rotates).
//!
//! Protocol:
//! - Exact frozen Scout-I organisms.
//! - Probe 1: Cross-depth zero-shot probe (trained on m2 -> w, tested on m3 -> z).
//! - Probe 2: In-depth terminal presence probe (trained on m3 -> z, tested on held-out m3 -> z).
//! - Probe 3: In-depth pair diagnostic decoder (trained on m3 -> (u, z), tested on held-out m3 -> (u, z)).

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
pub const NUM_PAIRS: usize = NUM_NODES * NUM_NODES;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthProbesResult {
    // Probe 1: Cross-Depth Endpoint Stability
    pub m2_trained_m2_test_acc: f32, // Baseline terminal decoding on m2
    pub m2_trained_m3_test_acc: f32, // Zero-shot transfer to m3
    // Probe 2: In-Depth Node Presence on m3
    pub m3_origin_acc: f32,       // u from m3
    pub m3_inter1_acc: f32,       // v from m3
    pub m3_inter2_acc: f32,       // w from m3
    pub m3_terminal_acc: f32,     // z from m3
    // Probe 3: In-Depth Pair Addressability on m3
    pub m3_pair_selectivity_pass: bool,
    pub m3_target_score: f32,     // r(m3, (u, z))
    pub m3_reverse_score: f32,    // r(m3, (z, u))
    pub m3_distractor_score: f32, // mean r(m3, (u, d'))
    pub m3_selectivity_margin: f32,
    // Mechanistic Diagnosis
    pub diagnosis: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JR2SeedResult {
    pub seed_index: usize,
    pub seed: u64,
    pub parameter_sha256: String,
    pub probes: DepthProbesResult,
}

fn train_scout_i_exact_model(model: &mut TypedTrainabilityModel, train_seed: u64, epochs: usize) {
    let mut rng = ChaCha8Rng::seed_from_u64(train_seed);
    let nodes = [1, 2, 3, 4, 5, 6];
    let lr = TRAIN_LR;

    for _epoch in 0..epochs {
        for _batch in 0..TRAIN_BATCHES_PER_EPOCH {
            let mut perm_nodes = nodes;
            perm_nodes.shuffle(&mut rng);
            let u = perm_nodes[0];
            let v = perm_nodes[1];
            let w = perm_nodes[2];
            let x = perm_nodes[3];
            let y = perm_nodes[4];

            let a1 = rng.gen_range(1..=3);
            let a2 = rng.gen_range(1..=3);

            let xi1 = (rng.gen::<f32>() - 0.5) * 0.02;
            let xi2 = (rng.gen::<f32>() - 0.5) * 0.02;

            let obs1 = TransitionObservation::with_noise(u, a1, v, xi1);
            let m0 = vec![0.0f32; REL_DIM];
            let (e1, dt_e1) = model.encode_edge(&obs1);
            let (m1, _, dt_m1) = model.compose_relation(&m0, &e1);

            let obs2 = TransitionObservation::with_noise(v, a2, w, xi2);
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

            let sensor_prob = model.query_sensor(&m2, 0.5);
            let sensor_err = sensor_prob - 0.95;
            model.b_sensor -= lr * sensor_err * 0.1;
            for i in 0..REL_DIM {
                model.w_sensor[i] -= lr * sensor_err * m2[i] * 0.01;
            }
        }
    }
}

fn train_linear_classifier(train_states: &[Vec<f32>], train_targets: &[usize], epochs: usize) -> Vec<f32> {
    let mut weights = vec![0.0f32; NUM_NODES * REL_DIM];
    let lr = 0.05f32;
    for _epoch in 0..epochs {
        for (m, &t) in train_states.iter().zip(train_targets.iter()) {
            let mut logits = vec![0.0f32; NUM_NODES];
            for k in 0..NUM_NODES {
                for i in 0..REL_DIM {
                    logits[k] += weights[k * REL_DIM + i] * m[i];
                }
            }
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
            for k in 0..NUM_NODES {
                let prob = (logits[k] - max_l).exp() / exp_sum;
                let grad = prob - if k == t { 1.0f32 } else { 0.0f32 };
                for i in 0..REL_DIM {
                    weights[k * REL_DIM + i] -= lr * grad * m[i];
                }
            }
        }
    }
    weights
}

fn eval_linear_classifier(weights: &[f32], test_states: &[Vec<f32>], test_targets: &[usize]) -> f32 {
    let mut correct = 0;
    for (m, &t) in test_states.iter().zip(test_targets.iter()) {
        let mut best_k = 0;
        let mut best_logit = f32::NEG_INFINITY;
        for k in 0..NUM_NODES {
            let mut logit = 0.0f32;
            for i in 0..REL_DIM {
                logit += weights[k * REL_DIM + i] * m[i];
            }
            if logit > best_logit {
                best_logit = logit;
                best_k = k;
            }
        }
        if best_k == t {
            correct += 1;
        }
    }
    correct as f32 / test_targets.len() as f32
}

fn run_scout_j_r2_seed(seed_index: usize) -> JR2SeedResult {
    let seed = 88000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;
    let eval_seed = seed ^ 0x9876543210;

    let mut model = TypedTrainabilityModel::new_init(seed, true, true, 1.0);
    train_scout_i_exact_model(&mut model, aux_train_seed, TRAIN_EPOCHS);
    let parameter_sha256 = model.compute_parameter_sha256();

    let mut rng = ChaCha8Rng::seed_from_u64(eval_seed);
    let nodes = [1, 2, 3, 4, 5, 6];
    let m0 = vec![0.0f32; REL_DIM];

    // Collect 400 train and 200 test trajectories for 2-step (m2) and 3-step (m3)
    let mut train_m2 = Vec::new();
    let mut train_m2_w = Vec::new(); // terminal node at step 2 (w)

    let mut test_m2 = Vec::new();
    let mut test_m2_w = Vec::new();

    let mut train_m3 = Vec::new();
    let mut train_m3_u = Vec::new(); // origin node (u)
    let mut train_m3_v = Vec::new(); // inter 1 (v)
    let mut train_m3_w = Vec::new(); // inter 2 (w)
    let mut train_m3_z = Vec::new(); // terminal node at step 3 (z)

    let mut test_m3 = Vec::new();
    let mut test_m3_u = Vec::new();
    let mut test_m3_v = Vec::new();
    let mut test_m3_w = Vec::new();
    let mut test_m3_z = Vec::new();

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

        let (e1, _) = model.encode_edge(&TransitionObservation::new(u, a1, v));
        let (m1, _, _) = model.compose_relation(&m0, &e1);
        let (e2, _) = model.encode_edge(&TransitionObservation::new(v, a2, w));
        let (m2, _, _) = model.compose_relation(&m1, &e2);
        let (e3, _) = model.encode_edge(&TransitionObservation::new(w, a3, z));
        let (m3, _, _) = model.compose_relation(&m2, &e3);

        if idx < 400 {
            train_m2.push(m2);
            train_m2_w.push(w - 1);

            train_m3.push(m3);
            train_m3_u.push(u - 1);
            train_m3_v.push(v - 1);
            train_m3_w.push(w - 1);
            train_m3_z.push(z - 1);
        } else {
            test_m2.push(m2);
            test_m2_w.push(w - 1);

            test_m3.push(m3);
            test_m3_u.push(u - 1);
            test_m3_v.push(v - 1);
            test_m3_w.push(w - 1);
            test_m3_z.push(z - 1);
        }
    }

    // -------------------------------------------------------------------------
    // Probe 1: Cross-Depth Endpoint Stability
    // -------------------------------------------------------------------------
    let w_probe_m2 = train_linear_classifier(&train_m2, &train_m2_w, 100);
    let m2_trained_m2_test_acc = eval_linear_classifier(&w_probe_m2, &test_m2, &test_m2_w);
    let m2_trained_m3_test_acc = eval_linear_classifier(&w_probe_m2, &test_m3, &test_m3_z);

    // -------------------------------------------------------------------------
    // Probe 2: In-Depth Node Presence on m3
    // -------------------------------------------------------------------------
    let w_probe_m3_u = train_linear_classifier(&train_m3, &train_m3_u, 100);
    let w_probe_m3_v = train_linear_classifier(&train_m3, &train_m3_v, 100);
    let w_probe_m3_w = train_linear_classifier(&train_m3, &train_m3_w, 100);
    let w_probe_m3_z = train_linear_classifier(&train_m3, &train_m3_z, 100);

    let m3_origin_acc = eval_linear_classifier(&w_probe_m3_u, &test_m3, &test_m3_u);
    let m3_inter1_acc = eval_linear_classifier(&w_probe_m3_v, &test_m3, &test_m3_v);
    let m3_inter2_acc = eval_linear_classifier(&w_probe_m3_w, &test_m3, &test_m3_w);
    let m3_terminal_acc = eval_linear_classifier(&w_probe_m3_z, &test_m3, &test_m3_z);

    // -------------------------------------------------------------------------
    // Probe 3: In-Depth Pair Diagnostic Decoder on m3 (u -> v -> w -> z)
    // -------------------------------------------------------------------------
    let mut w_pairs_m3 = vec![0.0f32; NUM_PAIRS * REL_DIM];
    let mut b_pairs_m3 = vec![0.0f32; NUM_PAIRS];
    let lr = 0.02f32;

    for _epoch in 0..100 {
        for (m3, (&u_idx, &z_idx)) in train_m3.iter().zip(train_m3_u.iter().zip(train_m3_z.iter())) {
            let u = u_idx + 1;
            let z = z_idx + 1;

            // Positive query: (u, z) = 1.0
            // Negative reverse: (z, u) = 0.0
            // Negative distractor: (u, d) = 0.0 for d != z
            let d_dist = ((z % NUM_NODES) + 1);

            let queries = [
                (u, z, 1.0f32, 1.0f32),
                (z, u, 0.0f32, 0.5f32),
                (u, d_dist, 0.0f32, 0.5f32),
            ];

            for &(s, d, target, weight) in &queries {
                let p_idx = (s - 1) * NUM_NODES + (d - 1);
                let mut logit = b_pairs_m3[p_idx];
                for i in 0..REL_DIM {
                    logit += w_pairs_m3[p_idx * REL_DIM + i] * m3[i];
                }
                let err = (sigmoid(logit) - target) * weight;
                b_pairs_m3[p_idx] -= lr * err;
                for i in 0..REL_DIM {
                    w_pairs_m3[p_idx * REL_DIM + i] -= lr * err * m3[i];
                }
            }
        }
    }

    // Evaluate pair selectivity on held-out test set
    let mut correct_pairs = 0;
    let mut total_target_score = 0.0f32;
    let mut total_reverse_score = 0.0f32;
    let mut total_distractor_score = 0.0f32;
    let mut total_margin = 0.0f32;

    let query_m3 = |m: &[f32], s: usize, d: usize| -> f32 {
        let p_idx = (s - 1) * NUM_NODES + (d - 1);
        let mut logit = b_pairs_m3[p_idx];
        for i in 0..REL_DIM {
            logit += w_pairs_m3[p_idx * REL_DIM + i] * m[i];
        }
        logit
    };

    for (m3, (&u_idx, &z_idx)) in test_m3.iter().zip(test_m3_u.iter().zip(test_m3_z.iter())) {
        let u = u_idx + 1;
        let z = z_idx + 1;

        let tgt_score = query_m3(m3, u, z);
        let rev_score = query_m3(m3, z, u);

        let mut max_dist = f32::NEG_INFINITY;
        let mut dist_sum = 0.0f32;
        let mut dist_count = 0;

        for d in 1..=NUM_NODES {
            if d != u && d != z {
                let s = query_m3(m3, u, d);
                dist_sum += s;
                dist_count += 1;
                if s > max_dist {
                    max_dist = s;
                }
            }
        }

        let mean_dist = dist_sum / dist_count as f32;
        let margin = tgt_score - max_dist;

        total_target_score += tgt_score;
        total_reverse_score += rev_score;
        total_distractor_score += mean_dist;
        total_margin += margin;

        if tgt_score > max_dist && tgt_score > rev_score {
            correct_pairs += 1;
        }
    }

    let n_test = test_m3.len() as f32;
    let m3_target_score = total_target_score / n_test;
    let m3_reverse_score = total_reverse_score / n_test;
    let m3_distractor_score = total_distractor_score / n_test;
    let m3_selectivity_margin = total_margin / n_test;
    let pair_pass_rate = correct_pairs as f32 / n_test;
    let m3_pair_selectivity_pass = pair_pass_rate >= 0.75;

    // Determine diagnosis based on the three probes
    let diagnosis = if m3_terminal_acc < 0.30 {
        "BRANCH_1: DESTINATION_NEVER_WRITTEN (Probe 2 fails -> Keyed destination writing needed)".to_string()
    } else if !m3_pair_selectivity_pass {
        "BRANCH_2: FILLERS_SURVIVE_BINDING_FAILS (Probe 2 passes, Probe 3 fails -> Tensor/unification binding needed)".to_string()
    } else if m2_trained_m3_test_acc < 0.40 {
        "BRANCH_3: RELATIONAL_REPRESENTATION_DRIFTS_WITH_DEPTH (Probes 2 & 3 pass, Probe 1 cross-depth fails -> Depth-invariance / closure needed)".to_string()
    } else {
        "BRANCH_4: FULL_DEPTH_TRANSFER_RETAINED".to_string()
    };

    let probes = DepthProbesResult {
        m2_trained_m2_test_acc,
        m2_trained_m3_test_acc,
        m3_origin_acc,
        m3_inter1_acc,
        m3_inter2_acc,
        m3_terminal_acc,
        m3_pair_selectivity_pass,
        m3_target_score,
        m3_reverse_score,
        m3_distractor_score,
        m3_selectivity_margin,
        diagnosis,
    };

    JR2SeedResult {
        seed_index,
        seed,
        parameter_sha256,
        probes,
    }
}

fn main() {
    println!("================================================================================");
    println!("SCOUT-E-Q17E-J-R2: Depth-Stable Relational Semantics & 3-Hop Probing");
    println!("Evaluating frozen Scout-I representations under 3 diagnostic probes");
    println!("================================================================================\n");

    let results: Vec<JR2SeedResult> = (1..=16)
        .into_par_iter()
        .map(|i| run_scout_j_r2_seed(i))
        .collect();

    let n = results.len() as f32;

    // Summary statistics
    let avg_m2_m2 = results.iter().map(|r| r.probes.m2_trained_m2_test_acc).sum::<f32>() / n * 100.0;
    let avg_m2_m3 = results.iter().map(|r| r.probes.m2_trained_m3_test_acc).sum::<f32>() / n * 100.0;

    let avg_m3_u = results.iter().map(|r| r.probes.m3_origin_acc).sum::<f32>() / n * 100.0;
    let avg_m3_v = results.iter().map(|r| r.probes.m3_inter1_acc).sum::<f32>() / n * 100.0;
    let avg_m3_w = results.iter().map(|r| r.probes.m3_inter2_acc).sum::<f32>() / n * 100.0;
    let avg_m3_z = results.iter().map(|r| r.probes.m3_terminal_acc).sum::<f32>() / n * 100.0;

    let avg_m3_tgt = results.iter().map(|r| r.probes.m3_target_score).sum::<f32>() / n;
    let avg_m3_rev = results.iter().map(|r| r.probes.m3_reverse_score).sum::<f32>() / n;
    let avg_m3_dist = results.iter().map(|r| r.probes.m3_distractor_score).sum::<f32>() / n;
    let avg_m3_margin = results.iter().map(|r| r.probes.m3_selectivity_margin).sum::<f32>() / n;
    let m3_pass_count = results.iter().filter(|r| r.probes.m3_pair_selectivity_pass).count();

    println!("--------------------------------------------------------------------------------");
    println!("PROBE 1: CROSS-DEPTH ENDPOINT COORDINATE STABILITY");
    println!("  m2-trained probe tested on m2 (baseline):        {:.1}%", avg_m2_m2);
    println!("  m2-trained probe tested zero-shot on m3 (z):    {:.1}%", avg_m2_m3);
    println!("  Cross-Depth Retention Ratio:                     {:.1}%", avg_m2_m3 / avg_m2_m2 * 100.0);
    println!("--------------------------------------------------------------------------------");
    println!("PROBE 2: IN-DEPTH NODE PRESENCE IN m3 (4-Node Chain u -> v -> w -> z)");
    println!("  Origin Node (u):                                {:.1}% (Chance = 16.7%)", avg_m3_u);
    println!("  Intermediate Node 1 (v):                        {:.1}% (Chance = 16.7%)", avg_m3_v);
    println!("  Intermediate Node 2 (w):                        {:.1}% (Chance = 16.7%)", avg_m3_w);
    println!("  New Terminal Node (z):                          {:.1}% (Chance = 16.7%)", avg_m3_z);
    println!("--------------------------------------------------------------------------------");
    println!("PROBE 3: IN-DEPTH PAIR-SPECIFIC DIAGNOSTIC DECODER ON m3");
    println!("  m3 Pair Selectivity Pass Rate (>= 75%):          {}/16 ({:.1}%)", m3_pass_count, m3_pass_count as f32 / n * 100.0);
    println!("  Target Score r(m3, (u, z)):                     {:>+6.2}", avg_m3_tgt);
    println!("  Reverse Score r(m3, (z, u)):                    {:>+6.2}", avg_m3_rev);
    println!("  Mean Distractor Score r(m3, (u, d')):           {:>+6.2}", avg_m3_dist);
    println!("  Mean Selectivity Margin:                        {:>+6.2}", avg_m3_margin);
    println!("--------------------------------------------------------------------------------");
    println!("OVERALL MECHANISTIC DIAGNOSES:");
    for r in &results {
        println!(
            "Seed [{:>2}] SHA:{:.8} | m2->m2:{:.1}% m2->m3:{:.1}% | m3(z):{:.1}% | m3-PairMargin:{:>+5.2} | {}",
            r.seed_index, r.parameter_sha256,
            r.probes.m2_trained_m2_test_acc * 100.0, r.probes.m2_trained_m3_test_acc * 100.0,
            r.probes.m3_terminal_acc * 100.0, r.probes.m3_selectivity_margin,
            r.probes.diagnosis
        );
    }
    println!("================================================================================");

    let out_dir = Path::new("crates/continuity_garden_core/data");
    fs::create_dir_all(out_dir).expect("Failed to create data directory");
    let out_path = out_dir.join("q17e_j_r2_depth_stability_results.json");
    let json_bytes = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    fs::write(&out_path, json_bytes).expect("Failed to write results file");
    println!("\nPersisted Scout J-R2 telemetry to: {}", out_path.display());
}
