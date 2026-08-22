//! SCOUT-E-Q17E-J-R1: Frozen-State Addressability Assay & Pair-Specific Diagnostic Readout
//!
//! Investigates whether the frozen Scout-I recurrent state (trained under broken joins)
//! already contains addressable endpoint representations when probed with a
//! pair-specific diagnostic decoder.
//!
//! Protocol:
//! 1. Regenerate & cryptographically hash exact Scout-I organisms.
//! 2. Part 1: Diagnostic linear probes (u, v, w) directly on exact Scout-I frozen m2 states.
//! 3. Part 2: Fit well-conditioned unconstrained pair-specific diagnostic readout W_pair in R^(36 x 128) on frozen m.
//! 4. Part 3: Require k=2 endpoint selectivity before evaluating k=3.
//! 5. Part 4: Evaluate zero-shot k=3 causal battery.

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
pub const NUM_PAIRS: usize = NUM_NODES * NUM_NODES; // 36 ordered pairs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactProbeResult {
    pub origin_accuracy: f32,       // Decodability of u from frozen m2
    pub intermediate_accuracy: f32, // Decodability of v from frozen m2
    pub terminal_accuracy: f32,     // Decodability of w from frozen m2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JR1SeedResult {
    pub seed_index: usize,
    pub seed: u64,
    pub parameter_sha256: String,
    pub exact_probe: ExactProbeResult,
    // k=2 Addressability
    pub k2_target_score: f32,          // r(m2, (A, C))
    pub k2_reverse_score: f32,         // r(m2, (C, A))
    pub k2_mean_distractor_score: f32, // mean r(m2, (A, D)) for D != C
    pub k2_endpoint_selective: bool,   // r(m2, A, C) > all r(m2, A, D) && r(m2, A, C) > r(m2, C, A)
    pub k2_selectivity_margin: f32,    // r(m2, A, C) - max_{D != C} r(m2, A, D)
    // Zero-Shot k=3 Causal Battery
    pub k3_pre_edge_score: f32,        // r(m2, (A, D))
    pub k3_intact_target_score: f32,   // r(m3, (A, D))
    pub k3_intact_reverse_score: f32,  // r(m3, (D, A))
    pub k3_intact_old_endpoint: f32,   // r(m3, (A, C))
    pub k3_wrong_src_score: f32,       // r(m3_ws, (A, D)) (X -> D)
    pub k3_wrong_dst_true_score: f32,  // r(m3_wd, (A, E)) (C -> E target)
    pub k3_wrong_dst_false_score: f32, // r(m3_wd, (A, D)) (C -> E false)
    pub k3_zero_edge_score: f32,       // r(m3_zero, (A, D))
    pub k3_donor_score: f32,           // r(m3_donor, (A, D))
    pub source_grounded: bool,         // Intact > WrongSrc
    pub destination_grounded: bool,    // (A, E) on C->E > (A, D) on C->E && Intact (A, D) > (A, D) on C->E
    pub full_causal_binding_pass: bool,
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

            // 1. Positive trajectory: u -> v -> w
            let obs1 = TransitionObservation::with_noise(u, a1, v, xi1);
            let m0 = vec![0.0f32; REL_DIM];
            let (e1, dt_e1) = model.encode_edge(&obs1);
            let (m1, _, dt_m1) = model.compose_relation(&m0, &e1);

            let obs2 = TransitionObservation::with_noise(v, a2, w, xi2);
            let (e2, dt_e2) = model.encode_edge(&obs2);
            let (m2, _, dt_m2) = model.compose_relation(&m1, &e2);

            let mut grad_m2 = vec![0.0f32; REL_DIM];
            let mut grad_m1 = vec![0.0f32; REL_DIM];

            // Terminal positive queries on m2
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

            // Shared prefix on m1
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

            // Backprop Step 2 & 1
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

            // Broken join negative
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

            // Sensor competence
            let sensor_prob = model.query_sensor(&m2, 0.5);
            let sensor_err = sensor_prob - 0.95;
            model.b_sensor -= lr * sensor_err * 0.1;
            for i in 0..REL_DIM {
                model.w_sensor[i] -= lr * sensor_err * m2[i] * 0.01;
            }
        }
    }
}

/// Evaluates linear decodability of (u, v, w) directly on exact Scout-I frozen m2 states
fn evaluate_exact_scout_i_probes(model: &TypedTrainabilityModel, probe_seed: u64) -> ExactProbeResult {
    let mut rng = ChaCha8Rng::seed_from_u64(probe_seed);
    let nodes = [1, 2, 3, 4, 5, 6];
    let m0 = vec![0.0f32; REL_DIM];

    let mut train_m2 = Vec::new();
    let mut train_u = Vec::new();
    let mut train_v = Vec::new();
    let mut train_w = Vec::new();

    let mut test_m2 = Vec::new();
    let mut test_u = Vec::new();
    let mut test_v = Vec::new();
    let mut test_w = Vec::new();

    for idx in 0..400 {
        let mut perm = nodes;
        perm.shuffle(&mut rng);
        let u = perm[0];
        let v = perm[1];
        let w = perm[2];
        let a1 = rng.gen_range(1..=3);
        let a2 = rng.gen_range(1..=3);

        let (e1, _) = model.encode_edge(&TransitionObservation::new(u, a1, v));
        let (m1, _, _) = model.compose_relation(&m0, &e1);
        let (e2, _) = model.encode_edge(&TransitionObservation::new(v, a2, w));
        let (m2, _, _) = model.compose_relation(&m1, &e2);

        if idx < 300 {
            train_m2.push(m2);
            train_u.push(u - 1);
            train_v.push(v - 1);
            train_w.push(w - 1);
        } else {
            test_m2.push(m2);
            test_u.push(u - 1);
            test_v.push(v - 1);
            test_w.push(w - 1);
        }
    }

    let train_classifier = |targets: &[usize]| -> Vec<f32> {
        let mut weights = vec![0.0f32; NUM_NODES * REL_DIM];
        let lr = 0.05f32;
        for _epoch in 0..100 {
            for (m, &t) in train_m2.iter().zip(targets.iter()) {
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
    };

    let eval_classifier = |weights: &[f32], test_targets: &[usize]| -> f32 {
        let mut correct = 0;
        for (m, &t) in test_m2.iter().zip(test_targets.iter()) {
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
    };

    let w_u = train_classifier(&train_u);
    let w_v = train_classifier(&train_v);
    let w_w = train_classifier(&train_w);

    ExactProbeResult {
        origin_accuracy: eval_classifier(&w_u, &test_u),
        intermediate_accuracy: eval_classifier(&w_v, &test_v),
        terminal_accuracy: eval_classifier(&w_w, &test_w),
    }
}

/// Unconstrained pair-specific diagnostic readout: maps each ordered pair (s, d) to an independent linear vector on m
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairSpecificDiagnosticDecoder {
    pub w_pairs: Vec<f32>, // NUM_PAIRS (36) x REL_DIM (128)
    pub b_pairs: Vec<f32>, // NUM_PAIRS (36)
}

impl PairSpecificDiagnosticDecoder {
    pub fn new() -> Self {
        Self {
            w_pairs: vec![0.0f32; NUM_PAIRS * REL_DIM],
            b_pairs: vec![0.0f32; NUM_PAIRS],
        }
    }

    #[inline]
    pub fn pair_index(s: usize, d: usize) -> usize {
        let s_i = (s - 1) % NUM_NODES;
        let d_i = (d - 1) % NUM_NODES;
        s_i * NUM_NODES + d_i
    }

    pub fn query(&self, m: &[f32], s: usize, d: usize) -> f32 {
        let idx = Self::pair_index(s, d);
        let mut logit = self.b_pairs[idx];
        for i in 0..REL_DIM {
            logit += self.w_pairs[idx * REL_DIM + i] * m[i];
        }
        logit
    }

    /// Fits the pair-specific readout on frozen recurrent states with balanced positive/negative examples
    pub fn fit_on_frozen_states(&mut self, model: &TypedTrainabilityModel, fit_seed: u64, epochs: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(fit_seed);
        let lr = 0.02f32;
        let nodes = [1, 2, 3, 4, 5, 6];
        let m0 = vec![0.0f32; REL_DIM];

        for _epoch in 0..epochs {
            for _batch in 0..TRAIN_BATCHES_PER_EPOCH {
                let mut perm = nodes;
                perm.shuffle(&mut rng);
                let u = perm[0];
                let v = perm[1];
                let w = perm[2];
                let x = perm[3]; // Broken join source
                let y = perm[4]; // Alternate destination

                let a1 = rng.gen_range(1..=3);
                let a2 = rng.gen_range(1..=3);
                let xi1 = (rng.gen::<f32>() - 0.5) * 0.02;
                let xi2 = (rng.gen::<f32>() - 0.5) * 0.02;

                // Intact forward trajectory: u -> v -> w
                let obs1 = TransitionObservation::with_noise(u, a1, v, xi1);
                let (e1, _) = model.encode_edge(&obs1);
                let (m1, _, _) = model.compose_relation(&m0, &e1);

                let obs2 = TransitionObservation::with_noise(v, a2, w, xi2);
                let (e2, _) = model.encode_edge(&obs2);
                let (m2, _, _) = model.compose_relation(&m1, &e2);

                // 1. Balanced Supervision on m2:
                // Positives: (u, w) = 1.0
                // Negatives: (w, u) = 0.0 (reverse), (u, y) = 0.0 (distractor dst), (x, w) = 0.0 (distractor src)
                let m2_queries = [
                    (u, w, 1.0f32, 1.0f32),  // Positive
                    (w, u, 0.0f32, 0.33f32), // Negative reverse
                    (u, y, 0.0f32, 0.33f32), // Negative distractor dst
                    (x, w, 0.0f32, 0.33f32), // Negative distractor src
                ];

                for &(s, d, target, weight) in &m2_queries {
                    let idx = Self::pair_index(s, d);
                    let logit = self.query(&m2, s, d);
                    let err = (sigmoid(logit) - target) * weight;

                    self.b_pairs[idx] -= lr * err;
                    for i in 0..REL_DIM {
                        self.w_pairs[idx * REL_DIM + i] -= lr * err * m2[i];
                    }
                }

                // 2. Balanced Supervision on m1 (Prefix reachability):
                // Positive: (u, v) = 1.0
                // Negative: (v, u) = 0.0, (u, w) = 0.0, (u, x) = 0.0
                let m1_queries = [
                    (u, v, 1.0f32, 1.0f32),
                    (v, u, 0.0f32, 0.33f32),
                    (u, w, 0.0f32, 0.33f32),
                    (u, x, 0.0f32, 0.33f32),
                ];

                for &(s, d, target, weight) in &m1_queries {
                    let idx = Self::pair_index(s, d);
                    let logit = self.query(&m1, s, d);
                    let err = (sigmoid(logit) - target) * weight;

                    self.b_pairs[idx] -= lr * err;
                    for i in 0..REL_DIM {
                        self.w_pairs[idx * REL_DIM + i] -= lr * err * m1[i];
                    }
                }

                // 3. Alternate destination counterfactual trajectory: u -> v, then v -> y (y != w)
                // Produces state m2_alt where (u, y) = 1.0 and (u, w) = 0.0!
                let obs2_alt = TransitionObservation::with_noise(v, a2, y, xi2);
                let (e2_alt, _) = model.encode_edge(&obs2_alt);
                let (m2_alt, _, _) = model.compose_relation(&m1, &e2_alt);

                let alt_queries = [
                    (u, y, 1.0f32, 1.0f32),
                    (u, w, 0.0f32, 1.0f32),
                ];

                for &(s, d, target, weight) in &alt_queries {
                    let idx = Self::pair_index(s, d);
                    let logit = self.query(&m2_alt, s, d);
                    let err = (sigmoid(logit) - target) * weight;

                    self.b_pairs[idx] -= lr * err;
                    for i in 0..REL_DIM {
                        self.w_pairs[idx * REL_DIM + i] -= lr * err * m2_alt[i];
                    }
                }

                // 4. Broken join trajectory: u -> v, then x -> w (x != v) -> (u, w) = 0.0
                let obs2_brk = TransitionObservation::with_noise(x, a2, w, xi2);
                let (e2_brk, _) = model.encode_edge(&obs2_brk);
                let (m2_brk, _, _) = model.compose_relation(&m1, &e2_brk);

                let idx_brk = Self::pair_index(u, w);
                let logit_brk = self.query(&m2_brk, u, w);
                let err_brk = sigmoid(logit_brk) - 0.0f32;
                self.b_pairs[idx_brk] -= lr * err_brk;
                for i in 0..REL_DIM {
                    self.w_pairs[idx_brk * REL_DIM + i] -= lr * err_brk * m2_brk[i];
                }
            }
        }
    }
}

fn evaluate_scout_j_r1_seed(seed_index: usize) -> JR1SeedResult {
    let seed = 88000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;
    let fit_decoder_seed = seed ^ 0xABCDEF01;

    // 1. Train exact Scout-I organism
    let mut model = TypedTrainabilityModel::new_init(seed, true, true, 1.0);
    train_scout_i_exact_model(&mut model, aux_train_seed, TRAIN_EPOCHS);
    let parameter_sha256 = model.compute_parameter_sha256();

    // 2. Part 1: Run exact diagnostic linear probes directly on exact Scout-I states
    let exact_probe = evaluate_exact_scout_i_probes(&model, seed ^ 0x5555AAAA);

    // 3. Part 2: Fit unconstrained pair-specific diagnostic decoder on frozen m states
    let mut decoder = PairSpecificDiagnosticDecoder::new();
    decoder.fit_on_frozen_states(&model, fit_decoder_seed, 100);

    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;
    let e = 5;
    let x = 6;

    let m0 = vec![0.0f32; REL_DIM];

    // 4. Evaluate k=2 Endpoint Addressability on A -> B -> C
    let (e1, _) = model.encode_edge(&TransitionObservation::new(a, 1, b));
    let (m1, _, _) = model.compose_relation(&m0, &e1);
    let (e2, _) = model.encode_edge(&TransitionObservation::new(b, 2, c));
    let (m2, _, _) = model.compose_relation(&m1, &e2);

    let k2_target_score = decoder.query(&m2, a, c);
    let k2_reverse_score = decoder.query(&m2, c, a);

    let mut distractor_scores = Vec::new();
    for node in 1..=NUM_NODES {
        if node != a && node != c {
            distractor_scores.push(decoder.query(&m2, a, node));
        }
    }
    let k2_mean_distractor_score = distractor_scores.iter().sum::<f32>() / distractor_scores.len() as f32;
    let max_distractor = distractor_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let k2_selectivity_margin = k2_target_score - max_distractor;
    let k2_endpoint_selective = (k2_target_score > max_distractor) && (k2_target_score > k2_reverse_score);

    // 5. Zero-Shot k=3 Causal Battery on A -> B -> C -> D
    // Pre-edge query for unseen endpoint (A, D) on m2
    let k3_pre_edge_score = decoder.query(&m2, a, d);

    // Step 3 Intact: C -> D
    let (e3_intact, _) = model.encode_edge(&TransitionObservation::new(c, 1, d));
    let (m3_intact, _, _) = model.compose_relation(&m2, &e3_intact);
    let k3_intact_target_score = decoder.query(&m3_intact, a, d);
    let k3_intact_reverse_score = decoder.query(&m3_intact, d, a);
    let k3_intact_old_endpoint = decoder.query(&m3_intact, a, c);

    // Wrong source: X -> D (6 -> 4)
    let (e3_wrong_src, _) = model.encode_edge(&TransitionObservation::new(x, 1, d));
    let (m3_wrong_src, _, _) = model.compose_relation(&m2, &e3_wrong_src);
    let k3_wrong_src_score = decoder.query(&m3_wrong_src, a, d);

    // Wrong destination: C -> E (3 -> 5)
    let (e3_wrong_dst, _) = model.encode_edge(&TransitionObservation::new(c, 1, e));
    let (m3_wrong_dst, _, _) = model.compose_relation(&m2, &e3_wrong_dst);
    let k3_wrong_dst_true_score = decoder.query(&m3_wrong_dst, a, e);
    let k3_wrong_dst_false_score = decoder.query(&m3_wrong_dst, a, d);

    // Zero final edge
    let (m3_zero, _, _) = model.compose_relation(&m2, &vec![0.0f32; EDGE_DIM]);
    let k3_zero_edge_score = decoder.query(&m3_zero, a, d);

    // Donor history transplant: D -> C -> B (donor) + intact C -> D
    let (e1_d, _) = model.encode_edge(&TransitionObservation::new(d, 1, c));
    let (m1_d, _, _) = model.compose_relation(&m0, &e1_d);
    let (e2_d, _) = model.encode_edge(&TransitionObservation::new(c, 2, b));
    let (m2_donor, _, _) = model.compose_relation(&m1_d, &e2_d);
    let (m3_donor, _, _) = model.compose_relation(&m2_donor, &e3_intact);
    let k3_donor_score = decoder.query(&m3_donor, a, d);

    let source_grounded = k3_intact_target_score > k3_wrong_src_score;
    let destination_grounded = (k3_wrong_dst_true_score > k3_wrong_dst_false_score)
        && (k3_intact_target_score > k3_wrong_dst_false_score);

    let full_causal_binding_pass = k2_endpoint_selective
        && (k3_intact_target_score > k3_intact_reverse_score)
        && (k3_intact_target_score > k3_pre_edge_score)
        && (k3_intact_target_score > k3_zero_edge_score)
        && source_grounded
        && destination_grounded
        && (k3_donor_score < 0.0);

    JR1SeedResult {
        seed_index,
        seed,
        parameter_sha256,
        exact_probe,
        k2_target_score,
        k2_reverse_score,
        k2_mean_distractor_score,
        k2_endpoint_selective,
        k2_selectivity_margin,
        k3_pre_edge_score,
        k3_intact_target_score,
        k3_intact_reverse_score,
        k3_intact_old_endpoint,
        k3_wrong_src_score,
        k3_wrong_dst_true_score,
        k3_wrong_dst_false_score,
        k3_zero_edge_score,
        k3_donor_score,
        source_grounded,
        destination_grounded,
        full_causal_binding_pass,
    }
}

fn main() {
    println!("================================================================================");
    println!("SCOUT-E-Q17E-J-R1: Frozen-State Relational Addressability & Diagnostic Decoder");
    println!("Evaluating exact Scout-I organisms with unconstrained pair-specific decoder");
    println!("================================================================================\n");

    let results: Vec<JR1SeedResult> = (1..=16)
        .into_par_iter()
        .map(|i| evaluate_scout_j_r1_seed(i))
        .collect();

    let n = results.len() as f32;

    // 1. Exact Probe Results on true Scout-I states
    let avg_origin = results.iter().map(|r| r.exact_probe.origin_accuracy).sum::<f32>() / n * 100.0;
    let avg_inter = results.iter().map(|r| r.exact_probe.intermediate_accuracy).sum::<f32>() / n * 100.0;
    let avg_term = results.iter().map(|r| r.exact_probe.terminal_accuracy).sum::<f32>() / n * 100.0;

    println!("--------------------------------------------------------------------------------");
    println!("1. DIAGNOSTIC LINEAR PROBES ON EXACT FROZEN SCOUT-I STATES (m2):");
    println!("   Origin Node Accuracy (u):        {:.1}% (Chance = 16.7%)", avg_origin);
    println!("   Intermediate Node Accuracy (v):  {:.1}% (Chance = 16.7%)", avg_inter);
    println!("   Terminal Node Accuracy (w):      {:.1}% (Chance = 16.7%)", avg_term);
    println!("--------------------------------------------------------------------------------");

    // 2. k=2 Addressability
    let k2_sel_count = results.iter().filter(|r| r.k2_endpoint_selective).count();
    let avg_k2_target = results.iter().map(|r| r.k2_target_score).sum::<f32>() / n;
    let avg_k2_rev = results.iter().map(|r| r.k2_reverse_score).sum::<f32>() / n;
    let avg_k2_dist = results.iter().map(|r| r.k2_mean_distractor_score).sum::<f32>() / n;
    let avg_k2_margin = results.iter().map(|r| r.k2_selectivity_margin).sum::<f32>() / n;

    println!("2. TWO-STEP ENDPOINT ADDRESSABILITY (A -> B -> C):");
    println!("   Endpoint Selectivity Pass Rate:  {}/16 ({:.1}%)", k2_sel_count, k2_sel_count as f32 / n * 100.0);
    println!("   Target Score r(m2, (A, C)):      {:>+6.2}", avg_k2_target);
    println!("   Reverse Score r(m2, (C, A)):     {:>+6.2}", avg_k2_rev);
    println!("   Mean Distractor r(m2, (A, D)):   {:>+6.2}", avg_k2_dist);
    println!("   Mean Selectivity Margin:         {:>+6.2}", avg_k2_margin);
    println!("--------------------------------------------------------------------------------");

    // 3. Zero-Shot k=3 Causal Battery
    let src_ground_count = results.iter().filter(|r| r.source_grounded).count();
    let dst_ground_count = results.iter().filter(|r| r.destination_grounded).count();
    let full_causal_count = results.iter().filter(|r| r.full_causal_binding_pass).count();

    let avg_k3_pre = results.iter().map(|r| r.k3_pre_edge_score).sum::<f32>() / n;
    let avg_k3_intact = results.iter().map(|r| r.k3_intact_target_score).sum::<f32>() / n;
    let avg_k3_ws = results.iter().map(|r| r.k3_wrong_src_score).sum::<f32>() / n;
    let avg_k3_wd_true = results.iter().map(|r| r.k3_wrong_dst_true_score).sum::<f32>() / n;
    let avg_k3_wd_false = results.iter().map(|r| r.k3_wrong_dst_false_score).sum::<f32>() / n;
    let avg_k3_zero = results.iter().map(|r| r.k3_zero_edge_score).sum::<f32>() / n;
    let avg_k3_donor = results.iter().map(|r| r.k3_donor_score).sum::<f32>() / n;

    println!("3. ZERO-SHOT 3-HOP CAUSAL BATTERY (A -> B -> C -> D):");
    println!("   Pre-Edge Score r(m2, (A, D)):              {:>+6.2}", avg_k3_pre);
    println!("   Intact Target Score r(m3, (A, D)):         {:>+6.2} (Gain over pre-edge: {:>+6.2})", avg_k3_intact, avg_k3_intact - avg_k3_pre);
    println!("   Wrong Source r(m3_ws, (A, D)) (X -> D):    {:>+6.2} -> Grounded: {}/16", avg_k3_ws, src_ground_count);
    println!("   Wrong Destination C -> E:");
    println!("     - True New Endpoint r(m3_wd, (A, E)):    {:>+6.2}", avg_k3_wd_true);
    println!("     - False Unreached Endpoint (A, D):       {:>+6.2} -> Grounded: {}/16", avg_k3_wd_false, dst_ground_count);
    println!("   Zero Final Edge r(m3_zero, (A, D)):        {:>+6.2}", avg_k3_zero);
    println!("   Donor History Transplant:                  {:>+6.2}", avg_k3_donor);
    println!("--------------------------------------------------------------------------------");
    println!("4. FULL COMPOSITIONAL BINDING VERDICT (All 6 criteria): {}/16 ({:.1}%)", full_causal_count, full_causal_count as f32 / n * 100.0);
    println!("================================================================================");

    for r in &results {
        println!(
            "Seed [{:>2}] SHA:{:.8} | Probe(w):{:.1}% | k2(AC):{:>+5.2} MaxDist:{:>+5.2} | m3(AD):{:>+5.2} Pre:{:>+5.2} | m3(AE|CE):{:>+5.2} m3(AD|CE):{:>+5.2} | Verdict: {}",
            r.seed_index, r.parameter_sha256, r.exact_probe.terminal_accuracy * 100.0, r.k2_target_score, r.k2_target_score - r.k2_selectivity_margin,
            r.k3_intact_target_score, r.k3_pre_edge_score, r.k3_wrong_dst_true_score, r.k3_wrong_dst_false_score,
            if r.full_causal_binding_pass { "PASS" } else { "FAIL" }
        );
    }

    let out_dir = Path::new("crates/continuity_garden_core/data");
    fs::create_dir_all(out_dir).expect("Failed to create data directory");
    let out_path = out_dir.join("q17e_j_r1_frozen_addressability_results.json");
    let json_bytes = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    fs::write(&out_path, json_bytes).expect("Failed to write results file");
    println!("\nPersisted Scout J-R1 telemetry to: {}", out_path.display());
}
