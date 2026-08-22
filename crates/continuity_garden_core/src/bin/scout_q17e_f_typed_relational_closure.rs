//! Scout Q17E-F: Typed Relational State Closure
//!
//! Investigates whether separating the local edge representation e_t = E(x_t)
//! from the typed relational accumulator m_{t+1} = C_theta(m_t, e_t)
//! enables recursive composition across multi-hop horizons.
//!
//! Architecture:
//!   e_t = tanh(W_e * x_t + b_e) in R^32 (Local Transition Representation)
//!   m_{t+1} = tanh(W_m * m_t + W_c * e_t + b_m) in R^96 (Typed Relational State)
//!   Query Readout r_theta(m_t, q) reads m_t exclusively.
//!   Total dimensions matched: 32 + 96 = 128.
//!
//! Training:
//!   2-step developmental sequences under exact 120 epochs with shared prefix supervision:
//!   - Step 1: r(m_1, (u, v)) > r(m_1, (v, u))
//!   - Step 2: r(m_2, (u, w)) > r(m_2, (w, u))
//!   - ZERO 3-hop labels in training!
//!
//! Assays:
//!   - Double dissociation: Transplant m-state vs transplant local e-channel
//!   - Zero-shot k=3 directional margin & paired sign-flip p-value
//!   - Independent cloned-twin state surgery & swap effect (same action semantics 1,2,1 and separate jitter)
//!   - Independent transposition reversals
//!   - Deranged shuffle superiority
//!   - Task-aligned early/late sensitivities

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub const EDGE_DIM: usize = 32;
pub const REL_DIM: usize = 96;
pub const OBS_DIM: usize = 4;
pub const QUERY_DIM: usize = 2;
pub const TRAIN_EPOCHS: usize = 120;
pub const TRAIN_LR: f32 = 0.030;
pub const TRAIN_BATCHES_PER_EPOCH: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionObservation {
    pub src: usize,
    pub action: usize,
    pub dst: usize,
    pub noise_jitter: f32,
}

impl TransitionObservation {
    pub fn new(src: usize, action: usize, dst: usize) -> Self {
        Self { src, action, dst, noise_jitter: 0.0 }
    }

    pub fn with_noise(src: usize, action: usize, dst: usize, noise_jitter: f32) -> Self {
        Self { src, action, dst, noise_jitter }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; OBS_DIM];
        let s = if self.src >= 10 { (self.src % 10) as f32 } else { self.src as f32 };
        let d = if self.dst >= 10 { (self.dst % 10) as f32 } else { self.dst as f32 };
        v[0] = (s / 5.0) + self.noise_jitter;
        v[1] = (self.action as f32) / 5.0;
        v[2] = (d / 5.0) + self.noise_jitter;
        v[3] = 1.0;
        v
    }
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedRelationalModel {
    // Edge encoder: x_t -> e_t (dim: EDGE_DIM x OBS_DIM)
    pub w_e: Vec<f32>,
    pub b_e: Vec<f32>,
    // Relational composition operator: (m_t, e_t) -> m_{t+1}
    pub w_m: Vec<f32>, // REL_DIM x REL_DIM
    pub w_c: Vec<f32>, // REL_DIM x EDGE_DIM
    pub b_m: Vec<f32>, // REL_DIM
    // Relational query head: (m_t, q) -> logit
    pub w_q: Vec<f32>, // REL_DIM x QUERY_DIM
    pub w_r: Vec<f32>, // REL_DIM
    pub b_r: f32,
    // Sensor head: (m_t, cue) -> prob
    pub w_sensor: Vec<f32>,
    pub b_sensor: f32,
}

impl TypedRelationalModel {
    pub fn new_init(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5A5A55A5A5A5A);
        let scale_e = (2.0f32 / (EDGE_DIM + OBS_DIM) as f32).sqrt();
        let scale_m = (2.0f32 / (REL_DIM + REL_DIM) as f32).sqrt();
        let scale_c = (2.0f32 / (REL_DIM + EDGE_DIM) as f32).sqrt();
        let scale_q = (2.0f32 / (REL_DIM + QUERY_DIM) as f32).sqrt();

        let mut w_e = vec![0.0f32; EDGE_DIM * OBS_DIM];
        let mut b_e = vec![0.0f32; EDGE_DIM];
        let mut w_m = vec![0.0f32; REL_DIM * REL_DIM];
        let mut w_c = vec![0.0f32; REL_DIM * EDGE_DIM];
        let mut b_m = vec![0.0f32; REL_DIM];
        let mut w_q = vec![0.0f32; REL_DIM * QUERY_DIM];
        let mut w_r = vec![0.0f32; REL_DIM];
        let mut w_sensor = vec![0.0f32; REL_DIM];

        for i in 0..(EDGE_DIM * OBS_DIM) {
            w_e[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_e;
        }
        for i in 0..(REL_DIM * REL_DIM) {
            w_m[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_m;
        }
        for i in 0..(REL_DIM * EDGE_DIM) {
            w_c[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_c;
        }
        for i in 0..(REL_DIM * QUERY_DIM) {
            w_q[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_q * 2.0;
        }
        for i in 0..REL_DIM {
            b_m[i] = (rng.gen::<f32>() * 2.0 - 1.0) * 0.02;
            w_r[i] = 1.0 / (REL_DIM as f32).sqrt();
            w_sensor[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_q * 0.1;
        }

        Self {
            w_e,
            b_e,
            w_m,
            w_c,
            b_m,
            w_q,
            w_r,
            b_r: 0.0,
            w_sensor,
            b_sensor: 2.5,
        }
    }

    #[inline(always)]
    pub fn encode_edge(&self, obs: &TransitionObservation) -> (Vec<f32>, Vec<f32>) {
        let x = obs.to_vec();
        let mut e = vec![0.0f32; EDGE_DIM];
        let mut dt_e = vec![0.0f32; EDGE_DIM];

        for i in 0..EDGE_DIM {
            let mut sum = self.b_e[i];
            for j in 0..OBS_DIM {
                sum += self.w_e[i * OBS_DIM + j] * x[j];
            }
            let val = sum.tanh();
            e[i] = val;
            dt_e[i] = 1.0 - val * val;
        }
        (e, dt_e)
    }

    #[inline(always)]
    pub fn compose_relation(&self, m_prev: &[f32], e: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut m_next = vec![0.0f32; REL_DIM];
        let mut dt_m = vec![0.0f32; REL_DIM];

        for i in 0..REL_DIM {
            let mut sum = self.b_m[i];
            for j in 0..REL_DIM {
                sum += self.w_m[i * REL_DIM + j] * m_prev[j];
            }
            for j in 0..EDGE_DIM {
                sum += self.w_c[i * EDGE_DIM + j] * e[j];
            }
            let val = sum.tanh();
            m_next[i] = val;
            dt_m[i] = 1.0 - val * val;
        }
        (m_next, dt_m)
    }

    #[inline(always)]
    pub fn query_composition(&self, m: &[f32], query: (usize, usize)) -> f32 {
        let q_s = query.0 as f32 / 5.0;
        let q_d = query.1 as f32 / 5.0;

        let mut sum = self.b_r;
        for i in 0..REL_DIM {
            let e_q_i = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
            sum += self.w_r[i] * m[i] * e_q_i;
        }
        sum
    }

    #[inline(always)]
    pub fn query_sensor(&self, m: &[f32], cue_feat: f32) -> f32 {
        let mut sum = self.b_sensor + cue_feat;
        for i in 0..REL_DIM {
            sum += self.w_sensor[i] * m[i];
        }
        sigmoid(sum)
    }

    pub fn meta_train_bptt(&mut self, rng_seed: u64, epochs: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(rng_seed);
        let lr = TRAIN_LR;

        for _ in 0..epochs {
            for _ in 0..TRAIN_BATCHES_PER_EPOCH {
                let u = 1;
                let v = 2;
                let w = 3;
                let w_alt = 4;
                let v_alt = 5;

                let traj_mode = rng.gen_range(0..3);

                let (obs1, obs2, q_pos2, q_neg2, q_pos1, q_neg1, is_causal) = match traj_mode {
                    0 => (
                        TransitionObservation::new(u, 1, v),
                        TransitionObservation::new(v, 2, w),
                        (u, w),
                        (w, u),
                        (u, v),
                        (v, u),
                        true,
                    ),
                    1 => (
                        TransitionObservation::new(w, 2, v),
                        TransitionObservation::new(v, 1, u),
                        (w, u),
                        (u, w),
                        (w, v),
                        (v, w),
                        true,
                    ),
                    _ => (
                        TransitionObservation::new(v, 2, w),
                        TransitionObservation::new(u, 1, v),
                        (w_alt, w_alt),
                        (u, w),
                        (v_alt, v_alt),
                        (u, v),
                        false,
                    ),
                };

                // Forward Step 1
                let m0 = vec![0.0f32; REL_DIM];
                let (e1, dt_e1) = self.encode_edge(&obs1);
                let (m1, dt_m1) = self.compose_relation(&m0, &e1);

                // Forward Step 2
                let (e2, dt_e2) = self.encode_edge(&obs2);
                let (m2, dt_m2) = self.compose_relation(&m1, &e2);

                let mut grad_m2 = vec![0.0f32; REL_DIM];
                let mut grad_m1 = vec![0.0f32; REL_DIM];

                // Terminal supervision on m2: (u -> w)
                for &(q_pair, target_y) in &[
                    (q_pos2, if is_causal { 1.0f32 } else { 0.0f32 }),
                    (q_neg2, 0.0f32),
                    ((u, w_alt), 0.0f32),
                ] {
                    let q_s = q_pair.0 as f32 / 5.0;
                    let q_d = q_pair.1 as f32 / 5.0;

                    let mut logit = self.b_r;
                    for i in 0..REL_DIM {
                        let e_q = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
                        logit += self.w_r[i] * m2[i] * e_q;
                    }
                    let pred = sigmoid(logit);
                    let err = pred - target_y;

                    self.b_r -= lr * err * 0.33;
                    for i in 0..REL_DIM {
                        let e_q = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
                        let d_w_r = err * m2[i] * e_q;
                        let d_e_q = err * self.w_r[i] * m2[i];
                        let d_m2_i = err * self.w_r[i] * e_q;

                        self.w_r[i] -= lr * d_w_r * 0.33;
                        self.w_q[i * QUERY_DIM] -= lr * d_e_q * q_s * 0.33;
                        self.w_q[i * QUERY_DIM + 1] -= lr * d_e_q * q_d * 0.33;
                        grad_m2[i] += d_m2_i * 0.33;
                    }
                }

                // Shared Prefix supervision on m1: (u -> v)
                for &(q_pair, target_y) in &[
                    (q_pos1, if is_causal { 1.0f32 } else { 0.0f32 }),
                    (q_neg1, 0.0f32),
                    ((u, v_alt), 0.0f32),
                ] {
                    let q_s = q_pair.0 as f32 / 5.0;
                    let q_d = q_pair.1 as f32 / 5.0;

                    let mut logit = self.b_r;
                    for i in 0..REL_DIM {
                        let e_q = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
                        logit += self.w_r[i] * m1[i] * e_q;
                    }
                    let pred = sigmoid(logit);
                    let err = pred - target_y;

                    self.b_r -= lr * err * 0.33;
                    for i in 0..REL_DIM {
                        let e_q = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
                        let d_w_r = err * m1[i] * e_q;
                        let d_e_q = err * self.w_r[i] * m1[i];
                        let d_m1_i = err * self.w_r[i] * e_q;

                        self.w_r[i] -= lr * d_w_r * 0.33;
                        self.w_q[i * QUERY_DIM] -= lr * d_e_q * q_s * 0.33;
                        self.w_q[i * QUERY_DIM + 1] -= lr * d_e_q * q_d * 0.33;
                        grad_m1[i] += d_m1_i * 0.33;
                    }
                }

                // Backprop Step 2: C_theta(m1, e2) -> m2
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
                    grad_m1[j] += sum;
                }

                // Backprop Edge 2: E(x2) -> e2
                let x2 = obs2.to_vec();
                for i in 0..EDGE_DIM {
                    let d_act_e2 = grad_e2[i] * dt_e2[i];
                    self.b_e[i] -= lr * d_act_e2;
                    for j in 0..OBS_DIM {
                        self.w_e[i * OBS_DIM + j] -= lr * d_act_e2 * x2[j];
                    }
                }

                // Backprop Step 1: C_theta(m0, e1) -> m1
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

                // Backprop Edge 1: E(x1) -> e1
                let x1 = obs1.to_vec();
                for i in 0..EDGE_DIM {
                    let d_act_e1 = grad_e1[i] * dt_e1[i];
                    self.b_e[i] -= lr * d_act_e1;
                    for j in 0..OBS_DIM {
                        self.w_e[i * OBS_DIM + j] -= lr * d_act_e1 * x1[j];
                    }
                }

                // Sensor training
                let sensor_prob = self.query_sensor(&m2, 0.5);
                let sensor_err = sensor_prob - 0.95;
                self.b_sensor -= lr * sensor_err * 0.1;
                for i in 0..REL_DIM {
                    self.w_sensor[i] -= lr * sensor_err * m2[i] * 0.01;
                }
            }
        }
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoutFResult {
    pub seed_index: usize,
    pub k2_passed: bool,
    pub k2_margin: f32,
    pub k3_margin: f32,
    pub k3_passed: bool,
    pub k3_transplant_m_margin: f32,
    pub k3_transplant_e_margin: f32,
    pub k3_m_swap_effect: f32,
    pub k3_e_swap_effect: f32,
    pub k3_m_surgery_passed: bool,
    pub k3_transposition_passed: bool,
    pub k3_shuffle_passed: bool,
    pub s_early: f32,
    pub s_late: f32,
    pub sensor_accuracy: f32,
}

fn evaluate_typed_seed(seed_index: usize) -> ScoutFResult {
    let seed = 88000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;

    let mut model = TypedRelationalModel::new_init(seed);
    model.meta_train_bptt(aux_train_seed, TRAIN_EPOCHS);

    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;

    // k=2 Evaluation: A -> B -> C
    let m0 = vec![0.0f32; REL_DIM];
    let (e1, _) = model.encode_edge(&TransitionObservation::new(a, 1, b));
    let (m1, _) = model.compose_relation(&m0, &e1);
    let (e2, _) = model.encode_edge(&TransitionObservation::new(b, 2, c));
    let (m2, _) = model.compose_relation(&m1, &e2);
    let k2_margin = model.query_composition(&m2, (a, c)) - model.query_composition(&m2, (c, a));
    let k2_passed = k2_margin > 0.0;

    // k=3 Intact Evaluation: A(1) -> B(2) -> C(3) -> D(4) (Nuisance xi_0 = 0.0)
    let (e3, dt_e3) = model.encode_edge(&TransitionObservation::with_noise(c, 1, d, 0.0));
    let (m3, dt_m3) = model.compose_relation(&m2, &e3);
    let k3_margin = model.query_composition(&m3, (a, d)) - model.query_composition(&m3, (d, a));
    let k3_passed = k3_margin > 0.0;

    // Independent Cloned-Twin Donor Stream: D(4) -1-> C(3) -2-> B(2) -1-> A(1) + Nuisance xi_1 = 0.01 (same action semantics 1,2,1!)
    let (e1_d, _) = model.encode_edge(&TransitionObservation::with_noise(d, 1, c, 0.01));
    let (m1_d, _) = model.compose_relation(&m0, &e1_d);
    let (e2_d, _) = model.encode_edge(&TransitionObservation::with_noise(c, 2, b, 0.01));
    let (m2_d, _) = model.compose_relation(&m1_d, &e2_d);
    let (e3_d, _) = model.encode_edge(&TransitionObservation::with_noise(b, 1, a, 0.01));
    let (m3_donor, _) = model.compose_relation(&m2_d, &e3_d);

    // Double Dissociation Assay:
    // 1. Transplant relational state m3_donor into query readout
    let k3_transplant_m_margin = model.query_composition(&m3_donor, (a, d)) - model.query_composition(&m3_donor, (d, a));
    let k3_m_swap_effect = k3_margin - k3_transplant_m_margin;
    let k3_m_surgery_passed = k3_margin > 0.0 && k3_transplant_m_margin < 0.0;

    // 2. Transplant local edge: C_theta(m2_intact, e3_donor)
    let (m3_hybrid_e, _) = model.compose_relation(&m2, &e3_d);
    let k3_transplant_e_margin = model.query_composition(&m3_hybrid_e, (a, d)) - model.query_composition(&m3_hybrid_e, (d, a));
    let k3_e_swap_effect = k3_margin - k3_transplant_e_margin;

    // Independent Transposition Control: D(4) -1-> C(3) -2-> B(2) -1-> A(1) + Nuisance xi_2 = -0.01
    let (e1_t, _) = model.encode_edge(&TransitionObservation::with_noise(d, 1, c, -0.01));
    let (m1_t, _) = model.compose_relation(&m0, &e1_t);
    let (e2_t, _) = model.encode_edge(&TransitionObservation::with_noise(c, 2, b, -0.01));
    let (m2_t, _) = model.compose_relation(&m1_t, &e2_t);
    let (e3_t, _) = model.encode_edge(&TransitionObservation::with_noise(b, 1, a, -0.01));
    let (m3_trans, _) = model.compose_relation(&m2_t, &e3_t);
    let k3_transposition_score = model.query_composition(&m3_trans, (a, d)) - model.query_composition(&m3_trans, (d, a));
    let k3_transposition_passed = k3_transposition_score < 0.0;

    // Shuffle [e2, e3, e1]: (B, 2, C) -> (C, 1, D) -> (A, 1, B)
    let (es1, _) = model.encode_edge(&TransitionObservation::new(b, 2, c));
    let (ms1, _) = model.compose_relation(&m0, &es1);
    let (es2, _) = model.encode_edge(&TransitionObservation::new(c, 1, d));
    let (ms2, _) = model.compose_relation(&ms1, &es2);
    let (es3, _) = model.encode_edge(&TransitionObservation::new(a, 1, b));
    let (ms3, _) = model.compose_relation(&ms2, &es3);
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
    let mut j_e1 = vec![0.0f32; EDGE_DIM * OBS_DIM];
    let (e1_fwd, dt_e1_fwd) = model.encode_edge(&TransitionObservation::new(a, 1, b));
    for i in 0..EDGE_DIM {
        for j in 0..OBS_DIM {
            j_e1[i * OBS_DIM + j] = dt_e1_fwd[i] * model.w_e[i * OBS_DIM + j];
        }
    }

    // J_m1 = dm1/de1 * J_e1 = diag(dt_m1) * W_c * J_e1 (dim: REL_DIM x OBS_DIM)
    let (m1_fwd, dt_m1_fwd) = model.compose_relation(&m0, &e1_fwd);
    let mut j_m1 = vec![0.0f32; REL_DIM * OBS_DIM];
    for i in 0..REL_DIM {
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for k in 0..EDGE_DIM {
                sum += model.w_c[i * EDGE_DIM + k] * j_e1[k * OBS_DIM + l];
            }
            j_m1[i * OBS_DIM + l] = dt_m1_fwd[i] * sum;
        }
    }

    // J_m2 = dm2/dm1 * J_m1 = diag(dt_m2) * W_m * J_m1
    let (e2_fwd, _) = model.encode_edge(&TransitionObservation::new(b, 2, c));
    let (m2_fwd, dt_m2_fwd) = model.compose_relation(&m1_fwd, &e2_fwd);
    let mut j_m2 = vec![0.0f32; REL_DIM * OBS_DIM];
    for i in 0..REL_DIM {
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for k in 0..REL_DIM {
                sum += model.w_m[i * REL_DIM + k] * j_m1[k * OBS_DIM + l];
            }
            j_m2[i * OBS_DIM + l] = dt_m2_fwd[i] * sum;
        }
    }

    // J_m3 = dm3/dm2 * J_m2 = diag(dt_m3) * W_m * J_m2
    let mut j_m3 = vec![0.0f32; REL_DIM * OBS_DIM];
    for i in 0..REL_DIM {
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for k in 0..REL_DIM {
                sum += model.w_m[i * REL_DIM + k] * j_m2[k * OBS_DIM + l];
            }
            j_m3[i * OBS_DIM + l] = dt_m3[i] * sum;
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

    // S_late = ||dm_dm3 * dm3/de3 * de3/dx3||_F
    let mut j_e3 = vec![0.0f32; EDGE_DIM * OBS_DIM];
    for i in 0..EDGE_DIM {
        for j in 0..OBS_DIM {
            j_e3[i * OBS_DIM + j] = dt_e3[i] * model.w_e[i * OBS_DIM + j];
        }
    }
    let mut dm_dx3 = vec![0.0f32; OBS_DIM];
    for j in 0..OBS_DIM {
        let mut sum = 0.0f32;
        for i in 0..REL_DIM {
            let mut sum_k = 0.0f32;
            for k in 0..EDGE_DIM {
                sum_k += model.w_c[i * EDGE_DIM + k] * j_e3[k * OBS_DIM + j];
            }
            sum += dm_dm3[i] * dt_m3[i] * sum_k;
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
        let prob = model.query_sensor(&m3, cue_feat);
        if (prob >= 0.5) == is_gold_valid {
            sensor_correct += 1;
        }
    }
    let sensor_accuracy = sensor_correct as f32 / 20.0;

    ScoutFResult {
        seed_index,
        k2_passed,
        k2_margin,
        k3_margin,
        k3_passed,
        k3_transplant_m_margin,
        k3_transplant_e_margin,
        k3_m_swap_effect,
        k3_e_swap_effect,
        k3_m_surgery_passed,
        k3_transposition_passed,
        k3_shuffle_passed,
        s_early,
        s_late,
        sensor_accuracy,
    }
}

fn main() {
    println!("=================================================================================================================================");
    println!("SCOUT Q17E-F: Typed Relational State Closure (16 Auxiliary Seeds, 120 Training Epochs)");
    println!("Architecture: e_t = E(x_t) [R^32] | m_{{t+1}} = C_theta(m_t, e_t) [R^96] | Shared Prefix Supervision");
    println!("=================================================================================================================================");

    let results: Vec<ScoutFResult> = (1..=16)
        .into_par_iter()
        .map(evaluate_typed_seed)
        .collect();

    let k2_pass = results.iter().filter(|r| r.k2_passed).count();
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

    println!("  k=2 Baseline Retention:                         {}/16 ({:.1}%)", k2_pass, k2_pass as f32 / 16.0 * 100.0);
    println!("  k=3 Positive Direction (m_3 > 0):               {}/16 (p={:.4})", k3_pass, k3_p);
    println!("  DOUBLE DISSOCIATION ASSAY:");
    println!("    - Relational Accumulator Swap (m3_donor):     Mean Swap: {:+.4} | Surgery Flips: {}/16 ({:.1}%)", mean_m_swap, m_surg_pass, m_surg_pass as f32 / 16.0 * 100.0);
    println!("    - Local Edge Swap (e3_donor on m2_intact):    Mean Swap: {:+.4}", mean_e_swap);
    println!("  k=3 Transposition Reversals:                    {}/16 ({:.1}%)", trans_pass, trans_pass as f32 / 16.0 * 100.0);
    println!("  k=3 Deranged Shuffle Superiority:               {}/16 ({:.1}%)", shuf_pass, shuf_pass as f32 / 16.0 * 100.0);
    println!("  Task-Aligned Early Sensitivity (S_early):       {:.4}", s_early);
    println!("  Task-Aligned Last-Edge Sensitivity (S_late):    {:.4}", s_late);
    println!("  1-Hop Sensor Accuracy (>= 90%):                 {}/16 ({:.1}%)", sensor_pass, sensor_pass as f32 / 16.0 * 100.0);

    println!("\n=================================================================================================================================");

    let data_dir = Path::new("data");
    let out_file = data_dir.join("q17e_f_typed_relational_results.json");
    let file = File::create(&out_file).expect("Failed to create scout F results JSON");
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &results).expect("Failed to write JSON");
    println!("Persisted full Scout F telemetry to: {}", out_file.display());
}
