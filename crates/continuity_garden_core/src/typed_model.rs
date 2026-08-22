//! Shared Typed Relational Organism & Exact Training Implementation
//!
//! Provides the canonical TypedTrainabilityModel, parameter hashing, and exact serialization
//! so that Scout G and downstream interrogations (H, H-R1, I) evaluate identical parameter sets.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const EDGE_DIM: usize = 32;
pub const REL_DIM: usize = 128;
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

    #[inline(always)]
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
pub struct TypedTrainabilityModel {
    pub w_e: Vec<f32>, // EDGE_DIM x OBS_DIM
    pub b_e: Vec<f32>, // EDGE_DIM
    pub w_m: Vec<f32>, // REL_DIM x REL_DIM
    pub w_c: Vec<f32>, // REL_DIM x EDGE_DIM
    pub b_m: Vec<f32>, // REL_DIM
    pub w_q: Vec<f32>, // REL_DIM x QUERY_DIM
    pub w_r: Vec<f32>, // REL_DIM
    pub b_r: f32,
    pub w_sensor: Vec<f32>,
    pub b_sensor: f32,
    pub is_linear_edge: bool,
    pub is_residual_accumulator: bool,
    pub eta_residual: f32,
}

impl TypedTrainabilityModel {
    pub fn new_init(seed: u64, is_linear_edge: bool, is_residual_accumulator: bool, eta: f32) -> Self {
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
            is_linear_edge,
            is_residual_accumulator,
            eta_residual: eta,
        }
    }

    pub fn compute_parameter_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&[if self.is_linear_edge { 1 } else { 0 }]);
        hasher.update(&[if self.is_residual_accumulator { 1 } else { 0 }]);
        hasher.update(&self.eta_residual.to_le_bytes());

        for val in &self.w_e { hasher.update(&val.to_le_bytes()); }
        for val in &self.b_e { hasher.update(&val.to_le_bytes()); }
        for val in &self.w_m { hasher.update(&val.to_le_bytes()); }
        for val in &self.w_c { hasher.update(&val.to_le_bytes()); }
        for val in &self.b_m { hasher.update(&val.to_le_bytes()); }
        for val in &self.w_q { hasher.update(&val.to_le_bytes()); }
        for val in &self.w_r { hasher.update(&val.to_le_bytes()); }
        hasher.update(&self.b_r.to_le_bytes());
        for val in &self.w_sensor { hasher.update(&val.to_le_bytes()); }
        hasher.update(&self.b_sensor.to_le_bytes());

        let result = hasher.finalize();
        format!("{:x}", result)
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
            if self.is_linear_edge {
                e[i] = sum;
                dt_e[i] = 1.0;
            } else {
                let val = sum.tanh();
                e[i] = val;
                dt_e[i] = 1.0 - val * val;
            }
        }
        (e, dt_e)
    }

    #[inline(always)]
    pub fn compose_relation(&self, m_prev: &[f32], e: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut m_tilde = vec![0.0f32; REL_DIM];
        let mut dt_m = vec![0.0f32; REL_DIM];
        let mut m_next = vec![0.0f32; REL_DIM];

        for i in 0..REL_DIM {
            let mut sum = self.b_m[i];
            for j in 0..REL_DIM {
                sum += self.w_m[i * REL_DIM + j] * m_prev[j];
            }
            for j in 0..EDGE_DIM {
                sum += self.w_c[i * EDGE_DIM + j] * e[j];
            }
            let val = sum.tanh();
            m_tilde[i] = val;
            dt_m[i] = 1.0 - val * val;

            if self.is_residual_accumulator {
                m_next[i] = m_prev[i] + self.eta_residual * val;
            } else {
                m_next[i] = val;
            }
        }
        (m_next, m_tilde, dt_m)
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
                let (m1, _, dt_m1) = self.compose_relation(&m0, &e1);

                // Forward Step 2
                let (e2, dt_e2) = self.encode_edge(&obs2);
                let (m2, _, dt_m2) = self.compose_relation(&m1, &e2);

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
                    let scale = if self.is_residual_accumulator { self.eta_residual } else { 1.0 };
                    d_act_m2[i] = grad_m2[i] * scale * dt_m2[i];
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
                    if self.is_residual_accumulator {
                        grad_m1[j] += grad_m2[j] + sum; // Identity residual pass-through!
                    } else {
                        grad_m1[j] += sum;
                    }
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
                    let scale = if self.is_residual_accumulator { self.eta_residual } else { 1.0 };
                    d_act_m1[i] = grad_m1[i] * scale * dt_m1[i];
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedScoutGOrganism {
    pub seed_index: usize,
    pub seed: u64,
    pub aux_train_seed: u64,
    pub parameter_sha256: String,
    pub k2_margin: f32,
    pub k3_margin: f32,
    pub model: TypedTrainabilityModel,
}
