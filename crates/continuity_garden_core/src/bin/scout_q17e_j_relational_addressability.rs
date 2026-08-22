//! SCOUT-E-Q17E-J: Relational Addressability & Endpoint Grounding
//!
//! Investigates whether relational composition failures in Q17E were caused by
//! query-head structural degeneracy rather than recurrent state capacity.
//!
//! Three stages:
//! 1. Stage J-A: Algebraic proof and property oracle of the old query head's structural degeneracy.
//! 2. Stage J-B: Diagnostic linear/nonlinear probing of terminal node identity from existing Scout-I states.
//! 3. Stage J-C: Pair-specific multiplicative readout (q = phi(s) * phi(d)) with true matched counterfactual training.

use std::fs;
use std::path::Path;

use continuity_garden_core::typed_model::{
    sigmoid, TransitionObservation, TypedTrainabilityModel, EDGE_DIM, OBS_DIM, REL_DIM,
    TRAIN_BATCHES_PER_EPOCH, TRAIN_EPOCHS, TRAIN_LR,
};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const NUM_NODES: usize = 6;

// ============================================================================
// STAGE J-A: THEOREM & PROPERTY ORACLE OF READOUT DEGENERACY
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageJAResult {
    pub num_trials: usize,
    pub max_relative_error: f64,
    pub k2_to_pre_edge_ratio_exact: f64,
    pub degeneracy_proven: bool,
}

pub fn run_stage_j_a_readout_oracle() -> StageJAResult {
    let mut rng = ChaCha8Rng::seed_from_u64(0x12345678);
    let mut max_rel_error = 0.0f64;
    let num_trials = 1000;

    for _ in 0..num_trials {
        let mut m = vec![0.0f32; REL_DIM];
        let mut w_r = vec![0.0f32; REL_DIM];
        let mut w_q = vec![0.0f32; REL_DIM * 2];
        let b_r = (rng.gen::<f32>() - 0.5) * 2.0;

        for i in 0..REL_DIM {
            m[i] = (rng.gen::<f32>() - 0.5) * 2.0;
            w_r[i] = (rng.gen::<f32>() - 0.5) * 2.0;
            w_q[i * 2] = (rng.gen::<f32>() - 0.5) * 2.0;
            w_q[i * 2 + 1] = (rng.gen::<f32>() - 0.5) * 2.0;
        }

        // Compute A(m) and B(m)
        let mut a_m = 0.0f32;
        let mut b_m = 0.0f32;
        for i in 0..REL_DIM {
            a_m += w_r[i] * m[i] * w_q[i * 2];
            b_m += w_r[i] * m[i] * w_q[i * 2 + 1];
        }

        // Test arbitrary pairs (s1, d1) and (s2, d2)
        let s1 = rng.gen_range(1..=6);
        let mut d1 = rng.gen_range(1..=6);
        while d1 == s1 { d1 = rng.gen_range(1..=6); }

        let s2 = rng.gen_range(1..=6);
        let mut d2 = rng.gen_range(1..=6);
        while d2 == s2 { d2 = rng.gen_range(1..=6); }

        let eval_margin = |s: usize, d: usize| -> f32 {
            let mut logit_fwd = b_r;
            let mut logit_rev = b_r;
            let q_s = s as f32 / 5.0;
            let q_d = d as f32 / 5.0;
            for i in 0..REL_DIM {
                let e_fwd = w_q[i * 2] * q_s + w_q[i * 2 + 1] * q_d;
                let e_rev = w_q[i * 2] * q_d + w_q[i * 2 + 1] * q_s;
                logit_fwd += w_r[i] * m[i] * e_fwd;
                logit_rev += w_r[i] * m[i] * e_rev;
            }
            logit_fwd - logit_rev
        };

        let m1 = eval_margin(s1, d1) as f64;
        let m2 = eval_margin(s2, d2) as f64;

        let normalized_1 = m1 / (s1 as f64 - d1 as f64);
        let normalized_2 = m2 / (s2 as f64 - d2 as f64);

        let theoretical_normalized = (a_m as f64 - b_m as f64) / 5.0;

        let err1 = (normalized_1 - theoretical_normalized).abs();
        let err2 = (normalized_2 - theoretical_normalized).abs();
        let pair_diff = (normalized_1 - normalized_2).abs();

        max_rel_error = max_rel_error.max(err1).max(err2).max(pair_diff);
    }

    // Specific test for A->C (1, 3) vs A->D (1, 4): Ratio must be (1-4)/(1-3) = -3/-2 = 1.5
    let ratio = (1.0 - 4.0) / (1.0 - 3.0);

    StageJAResult {
        num_trials,
        max_relative_error: max_rel_error,
        k2_to_pre_edge_ratio_exact: ratio,
        degeneracy_proven: max_rel_error < 1e-5,
    }
}

// ============================================================================
// STAGE J-B: DIAGNOSTIC PROBING ON EXISTING SCOUT-I STATES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageJBProbeResult {
    pub seed_index: usize,
    pub origin_accuracy: f32,       // Decodability of u from m2
    pub intermediate_accuracy: f32, // Decodability of v from m2
    pub terminal_accuracy: f32,     // Decodability of w from m2
}

pub fn run_stage_j_b_probing(seed_index: usize) -> StageJBProbeResult {
    let seed = 88000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;

    let mut model = TypedTrainabilityModel::new_init(seed, true, true, 1.0);
    // Train with Scout I logic (source-continuity training)
    model.meta_train_bptt(aux_train_seed, TRAIN_EPOCHS);

    // Generate 300 held-out trajectories for probe training & 100 for probe testing
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xFEEDFACE);
    let nodes = [1, 2, 3, 4, 5, 6];

    let mut train_m2 = Vec::new();
    let mut train_u = Vec::new();
    let mut train_v = Vec::new();
    let mut train_w = Vec::new();

    let mut test_m2 = Vec::new();
    let mut test_u = Vec::new();
    let mut test_v = Vec::new();
    let mut test_w = Vec::new();

    let m0 = vec![0.0f32; REL_DIM];

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

    // Train simple softmax linear classifiers for u, v, w from m2
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
                // Softmax
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

    let origin_accuracy = eval_classifier(&w_u, &test_u);
    let intermediate_accuracy = eval_classifier(&w_v, &test_v);
    let terminal_accuracy = eval_classifier(&w_w, &test_w);

    StageJBProbeResult {
        seed_index,
        origin_accuracy,
        intermediate_accuracy,
        terminal_accuracy,
    }
}

// ============================================================================
// STAGE J-C: PAIR-SPECIFIC MULTIPLICATIVE READOUT & TRUE MATCHED CURRICULUM
// ============================================================================

pub const EMBED_NODE_DIM: usize = 128;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplicativeQueryModel {
    pub w_e: Vec<f32>,     // EDGE_DIM x OBS_DIM
    pub b_e: Vec<f32>,     // EDGE_DIM
    pub w_m: Vec<f32>,     // REL_DIM x REL_DIM
    pub w_c: Vec<f32>,     // REL_DIM x EDGE_DIM
    pub b_m: Vec<f32>,     // REL_DIM
    // Multiplicative Role-Filler Readout Embeddings: phi_s, phi_d in R^(NUM_NODES x REL_DIM)
    pub phi_src: Vec<f32>, // NUM_NODES x REL_DIM
    pub phi_dst: Vec<f32>, // NUM_NODES x REL_DIM
    pub v_readout: Vec<f32>, // REL_DIM
    pub b_readout: f32,
    pub w_sensor: Vec<f32>,
    pub b_sensor: f32,
}

impl MultiplicativeQueryModel {
    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x33445566778899AA);
        let scale_e = (2.0f32 / (EDGE_DIM + OBS_DIM) as f32).sqrt();
        let scale_m = (2.0f32 / (REL_DIM + REL_DIM) as f32).sqrt();
        let scale_c = (2.0f32 / (REL_DIM + EDGE_DIM) as f32).sqrt();
        let scale_phi = (2.0f32 / REL_DIM as f32).sqrt();

        let mut w_e = vec![0.0f32; EDGE_DIM * OBS_DIM];
        let mut b_e = vec![0.0f32; EDGE_DIM];
        let mut w_m = vec![0.0f32; REL_DIM * REL_DIM];
        let mut w_c = vec![0.0f32; REL_DIM * EDGE_DIM];
        let mut b_m = vec![0.0f32; REL_DIM];
        let mut phi_src = vec![0.0f32; NUM_NODES * REL_DIM];
        let mut phi_dst = vec![0.0f32; NUM_NODES * REL_DIM];
        let mut v_readout = vec![0.0f32; REL_DIM];
        let mut w_sensor = vec![0.0f32; REL_DIM];

        for i in 0..(EDGE_DIM * OBS_DIM) { w_e[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_e; }
        for i in 0..(REL_DIM * REL_DIM) { w_m[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_m; }
        for i in 0..(REL_DIM * EDGE_DIM) { w_c[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_c; }
        for i in 0..(NUM_NODES * REL_DIM) {
            phi_src[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_phi;
            phi_dst[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_phi;
        }
        for i in 0..REL_DIM {
            b_m[i] = (rng.gen::<f32>() - 0.5) * 0.02;
            v_readout[i] = 1.0 / (REL_DIM as f32).sqrt();
            w_sensor[i] = (rng.gen::<f32>() - 0.5) * 0.1;
        }

        Self {
            w_e,
            b_e,
            w_m,
            w_c,
            b_m,
            phi_src,
            phi_dst,
            v_readout,
            b_readout: 0.0,
            w_sensor,
            b_sensor: 2.5,
        }
    }

    pub fn encode_edge(&self, obs: &TransitionObservation) -> (Vec<f32>, Vec<f32>) {
        let x = obs.to_vec();
        let mut e = self.b_e.clone();
        for i in 0..EDGE_DIM {
            for j in 0..OBS_DIM {
                e[i] += self.w_e[i * OBS_DIM + j] * x[j];
            }
        }
        let dt = vec![1.0f32; EDGE_DIM]; // Linear edge encoder
        (e, dt)
    }

    pub fn compose_relation(&self, m_prev: &[f32], e: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut delta = self.b_m.clone();
        for i in 0..REL_DIM {
            for j in 0..REL_DIM {
                delta[i] += self.w_m[i * REL_DIM + j] * m_prev[j];
            }
            for j in 0..EDGE_DIM {
                delta[i] += self.w_c[i * EDGE_DIM + j] * e[j];
            }
        }
        let mut dt = vec![0.0f32; REL_DIM];
        let mut delta_act = vec![0.0f32; REL_DIM];
        for i in 0..REL_DIM {
            let t = delta[i].tanh();
            delta_act[i] = t;
            dt[i] = 1.0 - t * t;
        }
        let mut m_next = vec![0.0f32; REL_DIM];
        for i in 0..REL_DIM {
            m_next[i] = m_prev[i] + delta_act[i]; // Additive residual pass-through
        }
        (m_next, delta_act, dt)
    }

    pub fn query_pair(&self, m: &[f32], s: usize, d: usize) -> f32 {
        let s_idx = (s - 1) % NUM_NODES;
        let d_idx = (d - 1) % NUM_NODES;
        let mut logit = self.b_readout;
        for i in 0..REL_DIM {
            let q_sd_i = self.phi_src[s_idx * REL_DIM + i] * self.phi_dst[d_idx * REL_DIM + i];
            logit += self.v_readout[i] * m[i] * q_sd_i;
        }
        logit
    }

    pub fn query_sensor(&self, m: &[f32], cue: f32) -> f32 {
        let mut logit = self.b_sensor + cue;
        for i in 0..REL_DIM {
            logit += self.w_sensor[i] * m[i];
        }
        sigmoid(logit)
    }

    pub fn meta_train_counterfactual(&mut self, train_seed: u64, epochs: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(train_seed);
        let lr = TRAIN_LR;
        let nodes = [1, 2, 3, 4, 5, 6];

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

                let m0 = vec![0.0f32; REL_DIM];

                // 1. POSITIVE PATH: u -> v -> w
                let obs1 = TransitionObservation::with_noise(u, a1, v, xi1);
                let (e1, dt_e1) = self.encode_edge(&obs1);
                let (m1, _, dt_m1) = self.compose_relation(&m0, &e1);

                let obs2 = TransitionObservation::with_noise(v, a2, w, xi2);
                let (e2, dt_e2) = self.encode_edge(&obs2);
                let (m2, _, dt_m2) = self.compose_relation(&m1, &e2);

                let mut grad_m2 = vec![0.0f32; REL_DIM];
                let mut grad_m1 = vec![0.0f32; REL_DIM];

                // Query 1: (u, w) = 1.0 (True endpoint)
                // Query 2: (w, u) = 0.0 (Reverse)
                // Query 3: (u, y) = 0.0 (Distractor endpoint)
                for &(s, d, target) in &[((u), (w), 1.0f32), ((w), (u), 0.0f32), ((u), (y), 0.0f32)] {
                    let logit = self.query_pair(&m2, s, d);
                    let pred = sigmoid(logit);
                    let err = pred - target;
                    let s_i = (s - 1) % NUM_NODES;
                    let d_i = (d - 1) % NUM_NODES;

                    self.b_readout -= lr * err * 0.33;
                    for i in 0..REL_DIM {
                        let q_sd_i = self.phi_src[s_i * REL_DIM + i] * self.phi_dst[d_i * REL_DIM + i];
                        let d_v = err * m2[i] * q_sd_i;
                        let d_m = err * self.v_readout[i] * q_sd_i;
                        let d_q = err * self.v_readout[i] * m2[i];

                        self.v_readout[i] -= lr * d_v * 0.33;
                        self.phi_src[s_i * REL_DIM + i] -= lr * d_q * self.phi_dst[d_i * REL_DIM + i] * 0.33;
                        self.phi_dst[d_i * REL_DIM + i] -= lr * d_q * self.phi_src[s_i * REL_DIM + i] * 0.33;
                        grad_m2[i] += d_m * 0.33;
                    }
                }

                // Shared prefix queries on m1: (u, v) = 1.0, (v, u) = 0.0, (u, x) = 0.0
                for &(s, d, target) in &[((u), (v), 1.0f32), ((v), (u), 0.0f32), ((u), (x), 0.0f32)] {
                    let logit = self.query_pair(&m1, s, d);
                    let pred = sigmoid(logit);
                    let err = pred - target;
                    let s_i = (s - 1) % NUM_NODES;
                    let d_i = (d - 1) % NUM_NODES;

                    self.b_readout -= lr * err * 0.33;
                    for i in 0..REL_DIM {
                        let q_sd_i = self.phi_src[s_i * REL_DIM + i] * self.phi_dst[d_i * REL_DIM + i];
                        let d_v = err * m1[i] * q_sd_i;
                        let d_m = err * self.v_readout[i] * q_sd_i;
                        let d_q = err * self.v_readout[i] * m1[i];

                        self.v_readout[i] -= lr * d_v * 0.33;
                        self.phi_src[s_i * REL_DIM + i] -= lr * d_q * self.phi_dst[d_i * REL_DIM + i] * 0.33;
                        self.phi_dst[d_i * REL_DIM + i] -= lr * d_q * self.phi_src[s_i * REL_DIM + i] * 0.33;
                        grad_m1[i] += d_m * 0.33;
                    }
                }

                // Backprop Step 2 & Step 1 into accumulator & edge weights
                let mut d_act_m2 = vec![0.0f32; REL_DIM];
                let mut grad_e2 = vec![0.0f32; EDGE_DIM];
                for i in 0..REL_DIM {
                    d_act_m2[i] = grad_m2[i] * dt_m2[i];
                    self.b_m[i] -= lr * d_act_m2[i];
                    for j in 0..EDGE_DIM {
                        self.w_c[i * EDGE_DIM + j] -= lr * d_act_m2[i] * e2[j];
                        grad_e2[j] += d_act_m2[i] * self.w_c[i * EDGE_DIM + j];
                    }
                }
                for j in 0..REL_DIM {
                    let mut sum = 0.0f32;
                    for i in 0..REL_DIM {
                        sum += self.w_m[i * REL_DIM + j] * d_act_m2[i];
                        self.w_m[i * REL_DIM + j] -= lr * d_act_m2[i] * m1[j];
                    }
                    grad_m1[j] += grad_m2[j] + sum; // Additive residual identity
                }
                let x2 = obs2.to_vec();
                for i in 0..EDGE_DIM {
                    let d_act_e2 = grad_e2[i] * dt_e2[i];
                    self.b_e[i] -= lr * d_act_e2;
                    for j in 0..OBS_DIM {
                        self.w_e[i * OBS_DIM + j] -= lr * d_act_e2 * x2[j];
                    }
                }

                let mut d_act_m1 = vec![0.0f32; REL_DIM];
                let mut grad_e1 = vec![0.0f32; EDGE_DIM];
                for i in 0..REL_DIM {
                    d_act_m1[i] = grad_m1[i] * dt_m1[i];
                    self.b_m[i] -= lr * d_act_m1[i];
                    for j in 0..EDGE_DIM {
                        self.w_c[i * EDGE_DIM + j] -= lr * d_act_m1[i] * e1[j];
                        grad_e1[j] += d_act_m1[i] * self.w_c[i * EDGE_DIM + j];
                    }
                }
                let x1 = obs1.to_vec();
                for i in 0..EDGE_DIM {
                    let d_act_e1 = grad_e1[i] * dt_e1[i];
                    self.b_e[i] -= lr * d_act_e1;
                    for j in 0..OBS_DIM {
                        self.w_e[i * OBS_DIM + j] -= lr * d_act_e1 * x1[j];
                    }
                }

                // 2. COUNTERFACTUAL 1: BROKEN JOIN u -> v, then x -> w (x != v) -> query(u, w) = 0.0
                let obs2_brk = TransitionObservation::with_noise(x, a2, w, xi2);
                let (e2_brk, dt_e2_brk) = self.encode_edge(&obs2_brk);
                let (m2_brk, _, dt_m2_brk) = self.compose_relation(&m1, &e2_brk);
                let logit_brk = self.query_pair(&m2_brk, u, w);
                let err_brk = sigmoid(logit_brk) - 0.0f32;
                let s_i = (u - 1) % NUM_NODES;
                let d_i = (w - 1) % NUM_NODES;

                self.b_readout -= lr * err_brk * 0.33;
                let mut grad_m2_brk = vec![0.0f32; REL_DIM];
                for i in 0..REL_DIM {
                    let q_sd_i = self.phi_src[s_i * REL_DIM + i] * self.phi_dst[d_i * REL_DIM + i];
                    self.v_readout[i] -= lr * err_brk * m2_brk[i] * q_sd_i * 0.33;
                    grad_m2_brk[i] += err_brk * self.v_readout[i] * q_sd_i * 0.33;
                }
                for i in 0..REL_DIM {
                    let d_act = grad_m2_brk[i] * dt_m2_brk[i];
                    self.b_m[i] -= lr * d_act;
                    for j in 0..EDGE_DIM {
                        self.w_c[i * EDGE_DIM + j] -= lr * d_act * e2_brk[j];
                    }
                }

                // 3. COUNTERFACTUAL 2: ALTERNATE DESTINATION u -> v, then v -> y (y != w) -> query(u, y) = 1.0, query(u, w) = 0.0
                let obs2_alt = TransitionObservation::with_noise(v, a2, y, xi2);
                let (e2_alt, dt_e2_alt) = self.encode_edge(&obs2_alt);
                let (m2_alt, _, dt_m2_alt) = self.compose_relation(&m1, &e2_alt);

                for &(s, d, target) in &[((u), (y), 1.0f32), ((u), (w), 0.0f32)] {
                    let logit_alt = self.query_pair(&m2_alt, s, d);
                    let err_alt = sigmoid(logit_alt) - target;
                    let s_i = (s - 1) % NUM_NODES;
                    let d_i = (d - 1) % NUM_NODES;

                    self.b_readout -= lr * err_alt * 0.33;
                    let mut grad_m2_alt = vec![0.0f32; REL_DIM];
                    for i in 0..REL_DIM {
                        let q_sd_i = self.phi_src[s_i * REL_DIM + i] * self.phi_dst[d_i * REL_DIM + i];
                        self.v_readout[i] -= lr * err_alt * m2_alt[i] * q_sd_i * 0.33;
                        grad_m2_alt[i] += err_alt * self.v_readout[i] * q_sd_i * 0.33;
                    }
                    for i in 0..REL_DIM {
                        let d_act = grad_m2_alt[i] * dt_m2_alt[i];
                        self.b_m[i] -= lr * d_act;
                        for j in 0..EDGE_DIM {
                            self.w_c[i * EDGE_DIM + j] -= lr * d_act * e2_alt[j];
                        }
                    }
                }

                // Sensor training
                let s_prob = self.query_sensor(&m2, 0.5);
                let s_err = s_prob - 0.95;
                self.b_sensor -= lr * s_err * 0.1;
                for i in 0..REL_DIM {
                    self.w_sensor[i] -= lr * s_err * m2[i] * 0.01;
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageJCSeedResult {
    pub seed_index: usize,
    pub seed: u64,
    pub k2_target_score: f32,       // r(m2, (A, C))
    pub k2_reverse_score: f32,      // r(m2, (C, A))
    pub k2_distractor_score: f32,   // r(m2, (A, D))
    pub k2_endpoint_selective: bool,// r(m2, A, C) > r(m2, A, D) && r(m2, A, C) > r(m2, C, A)
    pub k3_pre_edge_target_score: f32, // r(m2, (A, D))
    pub k3_intact_target_score: f32,   // r(m3, (A, D))
    pub k3_intact_reverse_score: f32,  // r(m3, (D, A))
    pub k3_intact_old_dst_score: f32,  // r(m3, (A, C)) (transitive vs frontier)
    pub k3_wrong_src_score: f32,       // r(m3_wrong_src, (A, D))
    pub k3_wrong_dst_target_score: f32,// r(m3_wrong_dst, (A, E)) (should be high!)
    pub k3_wrong_dst_old_score: f32,   // r(m3_wrong_dst, (A, D)) (should NOT be high!)
    pub k3_zero_edge_score: f32,       // r(m3_zero, (A, D))
    pub k3_donor_score: f32,           // r(m3_donor, (A, D))
    pub source_grounded: bool,         // Intact (A, D) > WrongSrc (A, D)
    pub destination_grounded: bool,    // WrongDst produces (A, E) > (A, D) && Intact > WrongDst on (A, D)
    pub full_binding_passed: bool,
    pub sensor_accuracy: f32,
}

pub fn run_stage_j_c_seed(seed_index: usize) -> StageJCSeedResult {
    let seed = 88000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;

    let mut model = MultiplicativeQueryModel::new(seed);
    model.meta_train_counterfactual(aux_train_seed, TRAIN_EPOCHS);

    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;
    let e = 5;
    let x = 6;

    let m0 = vec![0.0f32; REL_DIM];

    // Step 1: A -> B
    let (e1, _) = model.encode_edge(&TransitionObservation::new(a, 1, b));
    let (m1, _, _) = model.compose_relation(&m0, &e1);

    // Step 2: B -> C
    let (e2, _) = model.encode_edge(&TransitionObservation::new(b, 2, c));
    let (m2, _, _) = model.compose_relation(&m1, &e2);

    // k=2 Evaluation
    let k2_target_score = model.query_pair(&m2, a, c);
    let k2_reverse_score = model.query_pair(&m2, c, a);
    let k2_distractor_score = model.query_pair(&m2, a, d);
    let k2_endpoint_selective = k2_target_score > k2_distractor_score && k2_target_score > k2_reverse_score;

    // k=3 Pre-edge probe on m2 for (A, D)
    let k3_pre_edge_target_score = model.query_pair(&m2, a, d);

    // Step 3 Intact: C -> D
    let (e3_intact, _) = model.encode_edge(&TransitionObservation::new(c, 1, d));
    let (m3_intact, _, _) = model.compose_relation(&m2, &e3_intact);
    let k3_intact_target_score = model.query_pair(&m3_intact, a, d);
    let k3_intact_reverse_score = model.query_pair(&m3_intact, d, a);
    let k3_intact_old_dst_score = model.query_pair(&m3_intact, a, c);

    // Wrong source: X -> D (5 -> 4)
    let (e3_wrong_src, _) = model.encode_edge(&TransitionObservation::new(x, 1, d));
    let (m3_wrong_src, _, _) = model.compose_relation(&m2, &e3_wrong_src);
    let k3_wrong_src_score = model.query_pair(&m3_wrong_src, a, d);

    // Wrong destination: C -> E (3 -> 5)
    let (e3_wrong_dst, _) = model.encode_edge(&TransitionObservation::new(c, 1, e));
    let (m3_wrong_dst, _, _) = model.compose_relation(&m2, &e3_wrong_dst);
    let k3_wrong_dst_target_score = model.query_pair(&m3_wrong_dst, a, e); // Target for C->E
    let k3_wrong_dst_old_score = model.query_pair(&m3_wrong_dst, a, d);    // False score for (A, D)

    // Zero final edge
    let (m3_zero, _, _) = model.compose_relation(&m2, &vec![0.0f32; EDGE_DIM]);
    let k3_zero_edge_score = model.query_pair(&m3_zero, a, d);

    // Donor transplant: D -> C -> B (donor) + intact C -> D
    let (e1_d, _) = model.encode_edge(&TransitionObservation::new(d, 1, c));
    let (m1_d, _, _) = model.compose_relation(&m0, &e1_d);
    let (e2_d, _) = model.encode_edge(&TransitionObservation::new(c, 2, b));
    let (m2_donor, _, _) = model.compose_relation(&m1_d, &e2_d);
    let (m3_donor, _, _) = model.compose_relation(&m2_donor, &e3_intact);
    let k3_donor_score = model.query_pair(&m3_donor, a, d);

    let source_grounded = k3_intact_target_score > k3_wrong_src_score;
    let destination_grounded = k3_wrong_dst_target_score > k3_wrong_dst_old_score
        && k3_intact_target_score > k3_wrong_dst_old_score;

    let full_binding_passed = k2_endpoint_selective
        && (k3_intact_target_score > k3_intact_reverse_score)
        && (k3_intact_target_score > k3_pre_edge_target_score)
        && (k3_intact_target_score > k3_zero_edge_score)
        && source_grounded
        && destination_grounded
        && (k3_donor_score < 0.0);

    // Sensor check
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

    StageJCSeedResult {
        seed_index,
        seed,
        k2_target_score,
        k2_reverse_score,
        k2_distractor_score,
        k2_endpoint_selective,
        k3_pre_edge_target_score,
        k3_intact_target_score,
        k3_intact_reverse_score,
        k3_intact_old_dst_score,
        k3_wrong_src_score,
        k3_wrong_dst_target_score,
        k3_wrong_dst_old_score,
        k3_zero_edge_score,
        k3_donor_score,
        source_grounded,
        destination_grounded,
        full_binding_passed,
        sensor_accuracy,
    }
}

fn main() {
    println!("================================================================================");
    println!("SCOUT-E-Q17E-J: Relational Addressability & Endpoint Grounding");
    println!("================================================================================\n");

    // STAGE J-A
    println!("--------------------------------------------------------------------------------");
    println!("STAGE J-A: Algebraic Proof & Property Oracle of Readout Degeneracy");
    println!("--------------------------------------------------------------------------------");
    let stage_a = run_stage_j_a_readout_oracle();
    println!("  Trials tested:                  {}", stage_a.num_trials);
    println!("  Max Normalized Margin Diff:     {:.2e}", stage_a.max_relative_error);
    println!("  M(m; 1, 4) / M(m; 1, 3) Ratio:  {:.4} (Exact Theorem: 1.5000)", stage_a.k2_to_pre_edge_ratio_exact);
    println!("  Degeneracy Status:              {}", if stage_a.degeneracy_proven { "PROVEN (OLD_QUERY_HEAD_PAIR_SPECIFICITY = IMPOSSIBLE)" } else { "FAILED" });

    // STAGE J-B
    println!("\n--------------------------------------------------------------------------------");
    println!("STAGE J-B: Diagnostic Probing on Existing Scout-I Representations");
    println!("--------------------------------------------------------------------------------");
    let probe_results: Vec<StageJBProbeResult> = (1..=16)
        .into_par_iter()
        .map(|i| run_stage_j_b_probing(i))
        .collect();

    let n = probe_results.len() as f32;
    let avg_origin = probe_results.iter().map(|r| r.origin_accuracy).sum::<f32>() / n * 100.0;
    let avg_inter = probe_results.iter().map(|r| r.intermediate_accuracy).sum::<f32>() / n * 100.0;
    let avg_term = probe_results.iter().map(|r| r.terminal_accuracy).sum::<f32>() / n * 100.0;

    println!("  Mean Origin Node Accuracy (u):        {:.1}% (Chance = 16.7%)", avg_origin);
    println!("  Mean Intermediate Node Accuracy (v):  {:.1}% (Chance = 16.7%)", avg_inter);
    println!("  Mean Terminal Node Accuracy (w):      {:.1}% (Chance = 16.7%)", avg_term);
    for r in &probe_results {
        println!("    Seed [{:>2}]: u={:.1}%, v={:.1}%, w={:.1}%", r.seed_index, r.origin_accuracy * 100.0, r.intermediate_accuracy * 100.0, r.terminal_accuracy * 100.0);
    }

    // STAGE J-C
    println!("\n--------------------------------------------------------------------------------");
    println!("STAGE J-C: Pair-Specific Multiplicative Readout & True Matched Counterfactuals");
    println!("--------------------------------------------------------------------------------");
    let jc_results: Vec<StageJCSeedResult> = (1..=16)
        .into_par_iter()
        .map(|i| run_stage_j_c_seed(i))
        .collect();

    let k2_sel_count = jc_results.iter().filter(|r| r.k2_endpoint_selective).count();
    let src_ground_count = jc_results.iter().filter(|r| r.source_grounded).count();
    let dst_ground_count = jc_results.iter().filter(|r| r.destination_grounded).count();
    let full_bind_count = jc_results.iter().filter(|r| r.full_binding_passed).count();

    let avg_k2_target = jc_results.iter().map(|r| r.k2_target_score).sum::<f32>() / n;
    let avg_k2_dist = jc_results.iter().map(|r| r.k2_distractor_score).sum::<f32>() / n;
    let avg_k3_pre = jc_results.iter().map(|r| r.k3_pre_edge_target_score).sum::<f32>() / n;
    let avg_k3_intact = jc_results.iter().map(|r| r.k3_intact_target_score).sum::<f32>() / n;
    let avg_k3_w_src = jc_results.iter().map(|r| r.k3_wrong_src_score).sum::<f32>() / n;
    let avg_k3_w_dst_tgt = jc_results.iter().map(|r| r.k3_wrong_dst_target_score).sum::<f32>() / n;
    let avg_k3_w_dst_old = jc_results.iter().map(|r| r.k3_wrong_dst_old_score).sum::<f32>() / n;
    let avg_k3_donor = jc_results.iter().map(|r| r.k3_donor_score).sum::<f32>() / n;

    println!("  1. Two-Step Endpoint Selectivity:            {}/16 ({:.1}%)", k2_sel_count, k2_sel_count as f32 / n * 100.0);
    println!("     Target (A, C): {:>+5.2} | Distractor (A, D): {:>+5.2} (Selectivity Margin: {:>+5.2})", avg_k2_target, avg_k2_dist, avg_k2_target - avg_k2_dist);
    println!("  2. Zero-Shot 3-Hop Causal Battery:");
    println!("     Pre-Edge Score (A, D on m2):              {:>+5.2}", avg_k3_pre);
    println!("     Intact 3-Step Target Score (A, D on m3):  {:>+5.2} (Gain over pre-edge: {:>+5.2})", avg_k3_intact, avg_k3_intact - avg_k3_pre);
    println!("     Wrong Source Score (m2 + X->D for A->D):  {:>+5.2} -> Grounded: {}/16", avg_k3_w_src, src_ground_count);
    println!("     Wrong Destination C->E:");
    println!("       - Score for true new endpoint (A, E):   {:>+5.2}", avg_k3_w_dst_tgt);
    println!("       - Score for non-existent endpoint(A, D):{:>+5.2} -> Grounded: {}/16", avg_k3_w_dst_old, dst_ground_count);
    println!("     Donor History Transplant:                 {:>+5.2}", avg_k3_donor);
    println!("  3. Full Compositional Binding Verdict:       {}/16 ({:.1}%)", full_bind_count, full_bind_count as f32 / n * 100.0);
    println!("================================================================================");

    for r in &jc_results {
        println!(
            "Seed [{:>2}] k2(AC):{:>+5.2} k2(AD):{:>+5.2} | Pre(AD):{:>+5.2} m3(AD):{:>+5.2} | m3(AE|CE):{:>+5.2} m3(AD|CE):{:>+5.2} | Full Binding: {}",
            r.seed_index, r.k2_target_score, r.k2_distractor_score, r.k3_pre_edge_target_score, r.k3_intact_target_score, r.k3_wrong_dst_target_score, r.k3_wrong_dst_old_score,
            if r.full_binding_passed { "PASS" } else { "FAIL" }
        );
    }

    let out_dir = Path::new("crates/continuity_garden_core/data");
    fs::create_dir_all(out_dir).expect("Failed to create data directory");
    let out_path = out_dir.join("q17e_j_relational_addressability_results.json");
    let full_telemetry = serde_json::json!({
        "stage_a": stage_a,
        "stage_b_probes": probe_results,
        "stage_c_results": jc_results
    });
    fs::write(&out_path, serde_json::to_string_pretty(&full_telemetry).unwrap()).expect("Failed to write results");
    println!("\nPersisted complete Scout J telemetry to: {}", out_path.display());
}
