//! Scout Q17E-C: Minimal Adaptive Temporal Gating for Recursive Composition
//! Compares:
//!   A. Adaptive Scalar Gate: g_t = sigmoid(w_z^T z_t + w_x^T x_t + b_g) in R
//!   B. Adaptive Vector Update Gate (UGRNN): g_t = sigmoid(U_g z_t + V_g x_t + b_g) in R^d
//!   where z_{t+1} = g_t * z_t + (1 - g_t) * tanh(W_z z_t + W_x x_t + b_z)
//! Meta-trained on 2-step sequences ONLY under exact 120-epoch training baseline.
//!
//! Evaluates across 16 fresh auxiliary seeds measuring:
//! - k=2 baseline retention
//! - k=3 directional margin & paired sign-flip p-value
//! - Independent cloned-twin donor-state transplant assay & continuous swap effect (with separate nuisance realization)
//! - Independent transposition reversal pass rate (with separate nuisance realization)
//! - Deranged shuffle superiority pass rate
//! - Task-aligned early-edge Jacobian sensitivity S_early = ||(dm_3/dz_3) * (dz_3/dx_1)||_F
//! - Task-aligned last-edge Jacobian sensitivity S_late = ||(dm_3/dz_3) * (dz_3/dx_3)||_F
//! - Adaptive vs Mean-Clamped Gate Ablation: dynamic g(z, x) vs static clamped g_bar
//! - Gate trajectory telemetry (g_1, g_2, g_3) across conditions
//! - 20-trial 1-hop sensor competence

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub const HIDDEN_DIM: usize = 128;
pub const OBS_DIM: usize = 4;
pub const QUERY_DIM: usize = 2;
pub const TRAIN_EPOCHS: usize = 120;
pub const TRAIN_LR: f32 = 0.030;
pub const TRAIN_BATCHES_PER_EPOCH: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecurrentState {
    pub z: Vec<f32>,
}

impl RecurrentState {
    pub fn zero() -> Self {
        Self {
            z: vec![0.0f32; HIDDEN_DIM],
        }
    }
}

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

// ============================================================================
// Model A: Adaptive Scalar Gate RNN
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarGatedRNN {
    pub w_z: Vec<f32>,
    pub w_x: Vec<f32>,
    pub b_z: Vec<f32>,
    pub w_zg: Vec<f32>,
    pub w_xg: Vec<f32>,
    pub b_g: f32,
    pub w_q: Vec<f32>,
    pub w_r: Vec<f32>,
    pub b_r: f32,
    pub w_sensor: Vec<f32>,
    pub b_sensor: f32,
}

impl ScalarGatedRNN {
    pub fn new_init(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x1111111122222222);
        let scale_z = (2.0f32 / (HIDDEN_DIM + HIDDEN_DIM) as f32).sqrt();
        let scale_x = (2.0f32 / (HIDDEN_DIM + OBS_DIM) as f32).sqrt();
        let scale_r = (2.0f32 / (HIDDEN_DIM + QUERY_DIM) as f32).sqrt();

        let mut w_z = vec![0.0f32; HIDDEN_DIM * HIDDEN_DIM];
        let mut w_x = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
        let mut b_z = vec![0.0f32; HIDDEN_DIM];
        let mut w_zg = vec![0.0f32; HIDDEN_DIM];
        let mut w_xg = vec![0.0f32; OBS_DIM];
        let mut w_q = vec![0.0f32; HIDDEN_DIM * QUERY_DIM];
        let mut w_r = vec![0.0f32; HIDDEN_DIM];
        let mut w_sensor = vec![0.0f32; HIDDEN_DIM];

        for i in 0..(HIDDEN_DIM * HIDDEN_DIM) {
            w_z[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_z;
        }
        for i in 0..(HIDDEN_DIM * OBS_DIM) {
            w_x[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_x;
        }
        for i in 0..(HIDDEN_DIM * QUERY_DIM) {
            w_q[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_r * 2.0;
        }
        for i in 0..HIDDEN_DIM {
            b_z[i] = (rng.gen::<f32>() * 2.0 - 1.0) * 0.02;
            w_zg[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_z * 0.5;
            w_r[i] = 1.0 / (HIDDEN_DIM as f32).sqrt();
            w_sensor[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_r * 0.1;
        }
        for j in 0..OBS_DIM {
            w_xg[j] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_x * 0.5;
        }

        Self {
            w_z,
            w_x,
            b_z,
            w_zg,
            w_xg,
            b_g: -1.0, // initial bias toward new update
            w_q,
            w_r,
            b_r: 0.0,
            w_sensor,
            b_sensor: 2.5,
        }
    }

    #[inline(always)]
    pub fn step(&self, state: &RecurrentState, obs: &TransitionObservation, clamped_gate: Option<f32>) -> (RecurrentState, f32, Vec<f32>) {
        let x = obs.to_vec();
        let mut z_tilde = vec![0.0f32; HIDDEN_DIM];

        for i in 0..HIDDEN_DIM {
            let mut sum = self.b_z[i];
            for j in 0..HIDDEN_DIM {
                sum += self.w_z[i * HIDDEN_DIM + j] * state.z[j];
            }
            for j in 0..OBS_DIM {
                sum += self.w_x[i * OBS_DIM + j] * x[j];
            }
            z_tilde[i] = sum.tanh();
        }

        let g = match clamped_gate {
            Some(fixed_g) => fixed_g,
            None => {
                let mut sum_g = self.b_g;
                for j in 0..HIDDEN_DIM {
                    sum_g += self.w_zg[j] * state.z[j];
                }
                for j in 0..OBS_DIM {
                    sum_g += self.w_xg[j] * x[j];
                }
                sigmoid(sum_g)
            }
        };

        let mut next_z = vec![0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            next_z[i] = g * state.z[i] + (1.0 - g) * z_tilde[i];
        }

        (RecurrentState { z: next_z }, g, z_tilde)
    }

    #[inline(always)]
    pub fn query_composition(&self, state: &RecurrentState, query: (usize, usize)) -> f32 {
        let q_s = query.0 as f32 / 5.0;
        let q_d = query.1 as f32 / 5.0;

        let mut sum = self.b_r;
        for i in 0..HIDDEN_DIM {
            let e_q_i = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
            sum += self.w_r[i] * state.z[i] * e_q_i;
        }
        sum
    }

    #[inline(always)]
    pub fn query_sensor_trial(&self, state: &RecurrentState, cue_feat: f32) -> f32 {
        let mut sum = self.b_sensor + cue_feat;
        for i in 0..HIDDEN_DIM {
            sum += self.w_sensor[i] * state.z[i];
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

                let traj_mode = rng.gen_range(0..3);

                let (obs1, obs2, q_pos, q_neg, is_causal) = match traj_mode {
                    0 => (
                        TransitionObservation::new(u, 1, v),
                        TransitionObservation::new(v, 2, w),
                        (u, w),
                        (w, u),
                        true,
                    ),
                    1 => (
                        TransitionObservation::new(w, 2, v),
                        TransitionObservation::new(v, 1, u),
                        (w, u),
                        (u, w),
                        true,
                    ),
                    _ => (
                        TransitionObservation::new(v, 2, w),
                        TransitionObservation::new(u, 1, v),
                        (w_alt, w_alt),
                        (u, w),
                        false,
                    ),
                };

                let x1 = obs1.to_vec();
                let x2 = obs2.to_vec();

                // Step 1: z_0 = 0
                let z0 = RecurrentState::zero();
                let (z1, g1, z_tilde1) = self.step(&z0, &obs1, None);

                // Step 2: z1 -> z2
                let (z2, g2, z_tilde2) = self.step(&z1, &obs2, None);

                let mut grad_z2 = vec![0.0f32; HIDDEN_DIM];

                for &(q_pair, target_y) in &[
                    (q_pos, if is_causal { 1.0f32 } else { 0.0f32 }),
                    (q_neg, 0.0f32),
                    ((u, w_alt), 0.0f32),
                ] {
                    let q_s = q_pair.0 as f32 / 5.0;
                    let q_d = q_pair.1 as f32 / 5.0;

                    let mut logit = self.b_r;
                    for i in 0..HIDDEN_DIM {
                        let e_q_i = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
                        logit += self.w_r[i] * z2.z[i] * e_q_i;
                    }
                    let pred = sigmoid(logit);
                    let err = pred - target_y;

                    self.b_r -= lr * err * 0.33;
                    for i in 0..HIDDEN_DIM {
                        let e_q_i = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
                        let d_w_r = err * z2.z[i] * e_q_i;
                        let d_e_q_i = err * self.w_r[i] * z2.z[i];
                        let d_z2_i = err * self.w_r[i] * e_q_i;

                        self.w_r[i] -= lr * d_w_r * 0.33;
                        self.w_q[i * QUERY_DIM] -= lr * d_e_q_i * q_s * 0.33;
                        self.w_q[i * QUERY_DIM + 1] -= lr * d_e_q_i * q_d * 0.33;
                        grad_z2[i] += d_z2_i * 0.33;
                    }
                }

                // Backprop Step 2:
                // dz2/dz_tilde2 = (1 - g2)
                // dz2/dg2 = z1 - z_tilde2
                let mut d_g2 = 0.0f32;
                for i in 0..HIDDEN_DIM {
                    d_g2 += grad_z2[i] * (z1.z[i] - z_tilde2[i]);
                }
                let d_logit_g2 = d_g2 * g2 * (1.0 - g2);
                self.b_g -= lr * d_logit_g2;
                for j in 0..OBS_DIM {
                    self.w_xg[j] -= lr * d_logit_g2 * x2[j];
                }

                let mut d_a2 = vec![0.0f32; HIDDEN_DIM];
                for i in 0..HIDDEN_DIM {
                    let dt = 1.0 - z_tilde2[i] * z_tilde2[i];
                    d_a2[i] = grad_z2[i] * (1.0 - g2) * dt;
                    self.b_z[i] -= lr * d_a2[i];
                    for j in 0..OBS_DIM {
                        self.w_x[i * OBS_DIM + j] -= lr * d_a2[i] * x2[j];
                    }
                }

                let mut grad_z1 = vec![0.0f32; HIDDEN_DIM];
                for j in 0..HIDDEN_DIM {
                    let mut sum = 0.0f32;
                    for i in 0..HIDDEN_DIM {
                        sum += self.w_z[i * HIDDEN_DIM + j] * d_a2[i];
                    }
                    grad_z1[j] = grad_z2[j] * g2 + d_logit_g2 * self.w_zg[j] + sum;
                    self.w_zg[j] -= lr * d_logit_g2 * z1.z[j];
                }

                for i in 0..HIDDEN_DIM {
                    for j in 0..HIDDEN_DIM {
                        self.w_z[i * HIDDEN_DIM + j] -= lr * d_a2[i] * z1.z[j];
                    }
                }

                // Backprop Step 1:
                let mut d_g1 = 0.0f32;
                for i in 0..HIDDEN_DIM {
                    d_g1 += grad_z1[i] * (0.0 - z_tilde1[i]);
                }
                let d_logit_g1 = d_g1 * g1 * (1.0 - g1);
                self.b_g -= lr * d_logit_g1;
                for j in 0..OBS_DIM {
                    self.w_xg[j] -= lr * d_logit_g1 * x1[j];
                }

                for i in 0..HIDDEN_DIM {
                    let dt = 1.0 - z_tilde1[i] * z_tilde1[i];
                    let d_a1_i = grad_z1[i] * (1.0 - g1) * dt;
                    self.b_z[i] -= lr * d_a1_i;
                    for j in 0..OBS_DIM {
                        self.w_x[i * OBS_DIM + j] -= lr * d_a1_i * x1[j];
                    }
                }

                // Sensor training
                let sensor_prob = self.query_sensor_trial(&z2, 0.5);
                let sensor_err = sensor_prob - 0.95;
                self.b_sensor -= lr * sensor_err * 0.1;
                for i in 0..HIDDEN_DIM {
                    self.w_sensor[i] -= lr * sensor_err * z2.z[i] * 0.01;
                }
            }
        }
    }
}

// ============================================================================
// Model B: Adaptive Vector Update Gate RNN (UGRNN)
// ============================================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorGatedRNN {
    pub w_z: Vec<f32>,
    pub w_x: Vec<f32>,
    pub b_z: Vec<f32>,
    pub u_g: Vec<f32>,
    pub v_g: Vec<f32>,
    pub b_g: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_r: Vec<f32>,
    pub b_r: f32,
    pub w_sensor: Vec<f32>,
    pub b_sensor: f32,
}

impl VectorGatedRNN {
    pub fn new_init(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x3333333344444444);
        let scale_z = (2.0f32 / (HIDDEN_DIM + HIDDEN_DIM) as f32).sqrt();
        let scale_x = (2.0f32 / (HIDDEN_DIM + OBS_DIM) as f32).sqrt();
        let scale_r = (2.0f32 / (HIDDEN_DIM + QUERY_DIM) as f32).sqrt();

        let mut w_z = vec![0.0f32; HIDDEN_DIM * HIDDEN_DIM];
        let mut w_x = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
        let mut b_z = vec![0.0f32; HIDDEN_DIM];
        let mut u_g = vec![0.0f32; HIDDEN_DIM * HIDDEN_DIM];
        let mut v_g = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
        let mut b_g = vec![0.0f32; HIDDEN_DIM];
        let mut w_q = vec![0.0f32; HIDDEN_DIM * QUERY_DIM];
        let mut w_r = vec![0.0f32; HIDDEN_DIM];
        let mut w_sensor = vec![0.0f32; HIDDEN_DIM];

        for i in 0..(HIDDEN_DIM * HIDDEN_DIM) {
            w_z[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_z;
            u_g[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_z * 0.5;
        }
        for i in 0..(HIDDEN_DIM * OBS_DIM) {
            w_x[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_x;
            v_g[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_x * 0.5;
        }
        for i in 0..(HIDDEN_DIM * QUERY_DIM) {
            w_q[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_r * 2.0;
        }
        for i in 0..HIDDEN_DIM {
            b_z[i] = (rng.gen::<f32>() * 2.0 - 1.0) * 0.02;
            b_g[i] = -1.0; // initial bias toward new update
            w_r[i] = 1.0 / (HIDDEN_DIM as f32).sqrt();
            w_sensor[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_r * 0.1;
        }

        Self {
            w_z,
            w_x,
            b_z,
            u_g,
            v_g,
            b_g,
            w_q,
            w_r,
            b_r: 0.0,
            w_sensor,
            b_sensor: 2.5,
        }
    }

    #[inline(always)]
    pub fn step(&self, state: &RecurrentState, obs: &TransitionObservation, clamped_gate: Option<&[f32]>) -> (RecurrentState, Vec<f32>, Vec<f32>) {
        let x = obs.to_vec();
        let mut z_tilde = vec![0.0f32; HIDDEN_DIM];

        for i in 0..HIDDEN_DIM {
            let mut sum = self.b_z[i];
            for j in 0..HIDDEN_DIM {
                sum += self.w_z[i * HIDDEN_DIM + j] * state.z[j];
            }
            for j in 0..OBS_DIM {
                sum += self.w_x[i * OBS_DIM + j] * x[j];
            }
            z_tilde[i] = sum.tanh();
        }

        let mut g = vec![0.0f32; HIDDEN_DIM];
        match clamped_gate {
            Some(fixed_g) => {
                g.copy_from_slice(fixed_g);
            }
            None => {
                for i in 0..HIDDEN_DIM {
                    let mut sum_g = self.b_g[i];
                    for j in 0..HIDDEN_DIM {
                        sum_g += self.u_g[i * HIDDEN_DIM + j] * state.z[j];
                    }
                    for j in 0..OBS_DIM {
                        sum_g += self.v_g[i * OBS_DIM + j] * x[j];
                    }
                    g[i] = sigmoid(sum_g);
                }
            }
        }

        let mut next_z = vec![0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            next_z[i] = g[i] * state.z[i] + (1.0 - g[i]) * z_tilde[i];
        }

        (RecurrentState { z: next_z }, g, z_tilde)
    }

    #[inline(always)]
    pub fn query_composition(&self, state: &RecurrentState, query: (usize, usize)) -> f32 {
        let q_s = query.0 as f32 / 5.0;
        let q_d = query.1 as f32 / 5.0;

        let mut sum = self.b_r;
        for i in 0..HIDDEN_DIM {
            let e_q_i = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
            sum += self.w_r[i] * state.z[i] * e_q_i;
        }
        sum
    }

    #[inline(always)]
    pub fn query_sensor_trial(&self, state: &RecurrentState, cue_feat: f32) -> f32 {
        let mut sum = self.b_sensor + cue_feat;
        for i in 0..HIDDEN_DIM {
            sum += self.w_sensor[i] * state.z[i];
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

                let traj_mode = rng.gen_range(0..3);

                let (obs1, obs2, q_pos, q_neg, is_causal) = match traj_mode {
                    0 => (
                        TransitionObservation::new(u, 1, v),
                        TransitionObservation::new(v, 2, w),
                        (u, w),
                        (w, u),
                        true,
                    ),
                    1 => (
                        TransitionObservation::new(w, 2, v),
                        TransitionObservation::new(v, 1, u),
                        (w, u),
                        (u, w),
                        true,
                    ),
                    _ => (
                        TransitionObservation::new(v, 2, w),
                        TransitionObservation::new(u, 1, v),
                        (w_alt, w_alt),
                        (u, w),
                        false,
                    ),
                };

                let x1 = obs1.to_vec();
                let x2 = obs2.to_vec();

                // Step 1: z_0 = 0
                let z0 = RecurrentState::zero();
                let (z1, g1, z_tilde1) = self.step(&z0, &obs1, None);

                // Step 2: z1 -> z2
                let (z2, g2, z_tilde2) = self.step(&z1, &obs2, None);

                let mut grad_z2 = vec![0.0f32; HIDDEN_DIM];

                for &(q_pair, target_y) in &[
                    (q_pos, if is_causal { 1.0f32 } else { 0.0f32 }),
                    (q_neg, 0.0f32),
                    ((u, w_alt), 0.0f32),
                ] {
                    let q_s = q_pair.0 as f32 / 5.0;
                    let q_d = q_pair.1 as f32 / 5.0;

                    let mut logit = self.b_r;
                    for i in 0..HIDDEN_DIM {
                        let e_q_i = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
                        logit += self.w_r[i] * z2.z[i] * e_q_i;
                    }
                    let pred = sigmoid(logit);
                    let err = pred - target_y;

                    self.b_r -= lr * err * 0.33;
                    for i in 0..HIDDEN_DIM {
                        let e_q_i = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
                        let d_w_r = err * z2.z[i] * e_q_i;
                        let d_e_q_i = err * self.w_r[i] * z2.z[i];
                        let d_z2_i = err * self.w_r[i] * e_q_i;

                        self.w_r[i] -= lr * d_w_r * 0.33;
                        self.w_q[i * QUERY_DIM] -= lr * d_e_q_i * q_s * 0.33;
                        self.w_q[i * QUERY_DIM + 1] -= lr * d_e_q_i * q_d * 0.33;
                        grad_z2[i] += d_z2_i * 0.33;
                    }
                }

                // Backprop Step 2:
                // dz2/dz_tilde2_i = (1 - g2_i)
                // dz2/dg2_i = z1_i - z_tilde2_i
                let mut d_logit_g2 = vec![0.0f32; HIDDEN_DIM];
                for i in 0..HIDDEN_DIM {
                    let dg_i = grad_z2[i] * (z1.z[i] - z_tilde2[i]);
                    d_logit_g2[i] = dg_i * g2[i] * (1.0 - g2[i]);
                    self.b_g[i] -= lr * d_logit_g2[i];
                    for j in 0..OBS_DIM {
                        self.v_g[i * OBS_DIM + j] -= lr * d_logit_g2[i] * x2[j];
                    }
                }

                let mut d_a2 = vec![0.0f32; HIDDEN_DIM];
                for i in 0..HIDDEN_DIM {
                    let dt = 1.0 - z_tilde2[i] * z_tilde2[i];
                    d_a2[i] = grad_z2[i] * (1.0 - g2[i]) * dt;
                    self.b_z[i] -= lr * d_a2[i];
                    for j in 0..OBS_DIM {
                        self.w_x[i * OBS_DIM + j] -= lr * d_a2[i] * x2[j];
                    }
                }

                let mut grad_z1 = vec![0.0f32; HIDDEN_DIM];
                for j in 0..HIDDEN_DIM {
                    let mut sum_a = 0.0f32;
                    let mut sum_g = 0.0f32;
                    for i in 0..HIDDEN_DIM {
                        sum_a += self.w_z[i * HIDDEN_DIM + j] * d_a2[i];
                        sum_g += self.u_g[i * HIDDEN_DIM + j] * d_logit_g2[i];
                    }
                    grad_z1[j] = grad_z2[j] * g2[j] + sum_a + sum_g;
                }

                for i in 0..HIDDEN_DIM {
                    for j in 0..HIDDEN_DIM {
                        self.w_z[i * HIDDEN_DIM + j] -= lr * d_a2[i] * z1.z[j];
                        self.u_g[i * HIDDEN_DIM + j] -= lr * d_logit_g2[i] * z1.z[j];
                    }
                }

                // Backprop Step 1:
                let mut d_logit_g1 = vec![0.0f32; HIDDEN_DIM];
                for i in 0..HIDDEN_DIM {
                    let dg_i = grad_z1[i] * (0.0 - z_tilde1[i]);
                    d_logit_g1[i] = dg_i * g1[i] * (1.0 - g1[i]);
                    self.b_g[i] -= lr * d_logit_g1[i];
                    for j in 0..OBS_DIM {
                        self.v_g[i * OBS_DIM + j] -= lr * d_logit_g1[i] * x1[j];
                    }
                }

                for i in 0..HIDDEN_DIM {
                    let dt = 1.0 - z_tilde1[i] * z_tilde1[i];
                    let d_a1_i = grad_z1[i] * (1.0 - g1[i]) * dt;
                    self.b_z[i] -= lr * d_a1_i;
                    for j in 0..OBS_DIM {
                        self.w_x[i * OBS_DIM + j] -= lr * d_a1_i * x1[j];
                    }
                }

                // Sensor training
                let sensor_prob = self.query_sensor_trial(&z2, 0.5);
                let sensor_err = sensor_prob - 0.95;
                self.b_sensor -= lr * sensor_err * 0.1;
                for i in 0..HIDDEN_DIM {
                    self.w_sensor[i] -= lr * sensor_err * z2.z[i] * 0.01;
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
pub struct GatedEvalResult {
    pub seed_index: usize,
    pub model_type: String, // "SCALAR_GATE" or "VECTOR_UGRNN"
    pub k2_passed: bool,
    pub k2_margin: f32,
    pub k3_margin: f32,
    pub k3_passed: bool,
    pub k3_transplant_margin: f32,
    pub k3_swap_effect: f32,
    pub k3_surgery_transferred: bool,
    pub k3_transposition_score: f32,
    pub k3_transposition_passed: bool,
    pub k3_shuffle_passed: bool,
    pub s_early: f32,
    pub s_late: f32,
    pub k3_margin_clamped: f32,
    pub k3_passed_clamped: bool,
    pub k3_surgery_clamped: bool,
    pub gate_trajectory_intact: Vec<f32>,
    pub gate_trajectory_reversed: Vec<f32>,
    pub sensor_accuracy: f32,
}

fn evaluate_scalar_seed(seed_index: usize) -> GatedEvalResult {
    let seed = 88000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;

    let mut model = ScalarGatedRNN::new_init(seed);
    model.meta_train_bptt(aux_train_seed, TRAIN_EPOCHS);

    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;

    // k=2 Evaluation: A -> B -> C
    let z0 = RecurrentState::zero();
    let (z1, _, _) = model.step(&z0, &TransitionObservation::new(a, 1, b), None);
    let (z2, _, _) = model.step(&z1, &TransitionObservation::new(b, 2, c), None);
    let k2_margin = model.query_composition(&z2, (a, c)) - model.query_composition(&z2, (c, a));
    let k2_passed = k2_margin > 0.0;

    // k=3 Intact Evaluation: A(1) -> B(2) -> C(3) -> D(4) (Nuisance realization xi_0 = 0.0)
    let (z1_i, g1_i, zt1_i) = model.step(&z0, &TransitionObservation::with_noise(a, 1, b, 0.0), None);
    let (z2_i, g2_i, zt2_i) = model.step(&z1_i, &TransitionObservation::with_noise(b, 2, c, 0.0), None);
    let (z3_i, g3_i, zt3_i) = model.step(&z2_i, &TransitionObservation::with_noise(c, 1, d, 0.0), None);
    let k3_margin = model.query_composition(&z3_i, (a, d)) - model.query_composition(&z3_i, (d, a));
    let k3_passed = k3_margin > 0.0;

    // Independent Cloned-Twin Donor Stream: D(4) -1-> C(3) -2-> B(2) -1-> A(1) + Nuisance xi_1 = 0.01
    let (z1_d, _, _) = model.step(&z0, &TransitionObservation::with_noise(d, 1, c, 0.01), None);
    let (z2_d, _, _) = model.step(&z1_d, &TransitionObservation::with_noise(c, 2, b, 0.01), None);
    let (z3_donor, _, _) = model.step(&z2_d, &TransitionObservation::with_noise(b, 1, a, 0.01), None);
    let k3_transplant_margin = model.query_composition(&z3_donor, (a, d)) - model.query_composition(&z3_donor, (d, a));
    let k3_swap_effect = k3_margin - k3_transplant_margin;
    let k3_surgery_transferred = k3_margin > 0.0 && k3_transplant_margin < 0.0;

    // Independent Transposition Control: D(4) -1-> C(3) -2-> B(2) -1-> A(1) + Nuisance xi_2 = -0.01
    let (z1_t, g1_rev, _) = model.step(&z0, &TransitionObservation::with_noise(d, 1, c, -0.01), None);
    let (z2_t, g2_rev, _) = model.step(&z1_t, &TransitionObservation::with_noise(c, 2, b, -0.01), None);
    let (z3_trans, g3_rev, _) = model.step(&z2_t, &TransitionObservation::with_noise(b, 1, a, -0.01), None);
    let k3_transposition_score = model.query_composition(&z3_trans, (a, d)) - model.query_composition(&z3_trans, (d, a));
    let k3_transposition_passed = k3_transposition_score < 0.0;

    // Shuffle [e2, e3, e1]: (B, 2, C) -> (C, 1, D) -> (A, 1, B)
    let (z1_s, _, _) = model.step(&z0, &TransitionObservation::new(b, 2, c), None);
    let (z2_s, _, _) = model.step(&z1_s, &TransitionObservation::new(c, 1, d), None);
    let (z3_shuf, _, _) = model.step(&z2_s, &TransitionObservation::new(a, 1, b), None);
    let shuf_score = model.query_composition(&z3_shuf, (a, d)) - model.query_composition(&z3_shuf, (d, a));
    let k3_shuffle_passed = k3_margin > shuf_score;

    // Task-Aligned Jacobian Sensitivity:
    // dm_3/dz_3:
    let q_ad_s = a as f32 / 5.0;
    let q_ad_d = d as f32 / 5.0;
    let mut dm_dz3 = vec![0.0f32; HIDDEN_DIM];
    for i in 0..HIDDEN_DIM {
        let e_fwd = model.w_q[i * QUERY_DIM] * q_ad_s + model.w_q[i * QUERY_DIM + 1] * q_ad_d;
        let e_rev = model.w_q[i * QUERY_DIM] * q_ad_d + model.w_q[i * QUERY_DIM + 1] * q_ad_s;
        dm_dz3[i] = model.w_r[i] * (e_fwd - e_rev);
    }

    // A_t = g_t * I + (1 - g_t) * diag(1 - z_tilde_t^2) * W_z
    // dz1/dx1 = (1 - g1) * diag(1 - z_tilde1^2) * W_x
    let mut j1 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = (1.0 - g1_i) * (1.0 - zt1_i[i] * zt1_i[i]);
        for j in 0..OBS_DIM {
            j1[i * OBS_DIM + j] = dt * model.w_x[i * OBS_DIM + j];
        }
    }

    // J2 = A2 * J1
    let mut j2 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = (1.0 - g2_i) * (1.0 - zt2_i[i] * zt2_i[i]);
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j1[m * OBS_DIM + l];
            }
            j2[i * OBS_DIM + l] = g2_i * j1[i * OBS_DIM + l] + dt * sum;
        }
    }

    // J3 = A3 * J2
    let mut j3 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = (1.0 - g3_i) * (1.0 - zt3_i[i] * zt3_i[i]);
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j2[m * OBS_DIM + l];
            }
            j3[i * OBS_DIM + l] = g3_i * j2[i * OBS_DIM + l] + dt * sum;
        }
    }

    // S_early = ||dm_dz3 * J3||_F
    let mut dm_dx1 = vec![0.0f32; OBS_DIM];
    for j in 0..OBS_DIM {
        let mut sum = 0.0f32;
        for i in 0..HIDDEN_DIM {
            sum += dm_dz3[i] * j3[i * OBS_DIM + j];
        }
        dm_dx1[j] = sum;
    }
    let s_early = dm_dx1.iter().map(|&v| v * v).sum::<f32>().sqrt();

    // S_late = ||dm_dz3 * dz3/dx3||_F = ||dm_dz3 * (1 - g3) * diag(1 - zt3^2) * W_x||_F
    let mut dm_dx3 = vec![0.0f32; OBS_DIM];
    for j in 0..OBS_DIM {
        let mut sum = 0.0f32;
        for i in 0..HIDDEN_DIM {
            let dt = (1.0 - g3_i) * (1.0 - zt3_i[i] * zt3_i[i]);
            sum += dm_dz3[i] * dt * model.w_x[i * OBS_DIM + j];
        }
        dm_dx3[j] = sum;
    }
    let s_late = dm_dx3.iter().map(|&v| v * v).sum::<f32>().sqrt();

    // Mean Clamped Gate Ablation
    let g_bar = (g1_i + g2_i + g3_i) / 3.0;
    let (z1_c, _, _) = model.step(&z0, &TransitionObservation::new(a, 1, b), Some(g_bar));
    let (z2_c, _, _) = model.step(&z1_c, &TransitionObservation::new(b, 2, c), Some(g_bar));
    let (z3_c, _, _) = model.step(&z2_c, &TransitionObservation::new(c, 1, d), Some(g_bar));
    let k3_margin_clamped = model.query_composition(&z3_c, (a, d)) - model.query_composition(&z3_c, (d, a));
    let k3_passed_clamped = k3_margin_clamped > 0.0;

    let (z1_dc, _, _) = model.step(&z0, &TransitionObservation::with_noise(d, 1, c, 0.01), Some(g_bar));
    let (z2_dc, _, _) = model.step(&z1_dc, &TransitionObservation::with_noise(c, 2, b, 0.01), Some(g_bar));
    let (z3_dc, _, _) = model.step(&z2_dc, &TransitionObservation::with_noise(b, 1, a, 0.01), Some(g_bar));
    let k3_transplant_clamped = model.query_composition(&z3_dc, (a, d)) - model.query_composition(&z3_dc, (d, a));
    let k3_surgery_clamped = k3_margin_clamped > 0.0 && k3_transplant_clamped < 0.0;

    // Sensor task
    let mut rng_sensor = ChaCha8Rng::seed_from_u64(seed ^ 0x66666666);
    let mut sensor_correct = 0;
    for trial_id in 0..20 {
        let is_gold_valid = trial_id < 10;
        let cue_feat = if is_gold_valid {
            0.4 + rng_sensor.gen::<f32>() * 0.2
        } else {
            -3.5 - rng_sensor.gen::<f32>() * 0.5
        };
        let prob = model.query_sensor_trial(&z3_i, cue_feat);
        if (prob >= 0.5) == is_gold_valid {
            sensor_correct += 1;
        }
    }
    let sensor_accuracy = sensor_correct as f32 / 20.0;

    GatedEvalResult {
        seed_index,
        model_type: "SCALAR_GATE".to_string(),
        k2_passed,
        k2_margin,
        k3_margin,
        k3_passed,
        k3_transplant_margin,
        k3_swap_effect,
        k3_surgery_transferred,
        k3_transposition_score,
        k3_transposition_passed,
        k3_shuffle_passed,
        s_early,
        s_late,
        k3_margin_clamped,
        k3_passed_clamped,
        k3_surgery_clamped,
        gate_trajectory_intact: vec![g1_i, g2_i, g3_i],
        gate_trajectory_reversed: vec![g1_rev, g2_rev, g3_rev],
        sensor_accuracy,
    }
}

fn evaluate_vector_seed(seed_index: usize) -> GatedEvalResult {
    let seed = 88000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;

    let mut model = VectorGatedRNN::new_init(seed);
    model.meta_train_bptt(aux_train_seed, TRAIN_EPOCHS);

    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;

    // k=2 Evaluation: A -> B -> C
    let z0 = RecurrentState::zero();
    let (z1, _, _) = model.step(&z0, &TransitionObservation::new(a, 1, b), None);
    let (z2, _, _) = model.step(&z1, &TransitionObservation::new(b, 2, c), None);
    let k2_margin = model.query_composition(&z2, (a, c)) - model.query_composition(&z2, (c, a));
    let k2_passed = k2_margin > 0.0;

    // k=3 Intact Evaluation: A(1) -> B(2) -> C(3) -> D(4) (Nuisance realization xi_0 = 0.0)
    let (z1_i, g1_i, zt1_i) = model.step(&z0, &TransitionObservation::with_noise(a, 1, b, 0.0), None);
    let (z2_i, g2_i, zt2_i) = model.step(&z1_i, &TransitionObservation::with_noise(b, 2, c, 0.0), None);
    let (z3_i, g3_i, zt3_i) = model.step(&z2_i, &TransitionObservation::with_noise(c, 1, d, 0.0), None);
    let k3_margin = model.query_composition(&z3_i, (a, d)) - model.query_composition(&z3_i, (d, a));
    let k3_passed = k3_margin > 0.0;

    // Independent Cloned-Twin Donor Stream: D(4) -1-> C(3) -2-> B(2) -1-> A(1) + Nuisance xi_1 = 0.01
    let (z1_d, _, _) = model.step(&z0, &TransitionObservation::with_noise(d, 1, c, 0.01), None);
    let (z2_d, _, _) = model.step(&z1_d, &TransitionObservation::with_noise(c, 2, b, 0.01), None);
    let (z3_donor, _, _) = model.step(&z2_d, &TransitionObservation::with_noise(b, 1, a, 0.01), None);
    let k3_transplant_margin = model.query_composition(&z3_donor, (a, d)) - model.query_composition(&z3_donor, (d, a));
    let k3_swap_effect = k3_margin - k3_transplant_margin;
    let k3_surgery_transferred = k3_margin > 0.0 && k3_transplant_margin < 0.0;

    // Independent Transposition Control: D(4) -1-> C(3) -2-> B(2) -1-> A(1) + Nuisance xi_2 = -0.01
    let (z1_t, g1_rev, _) = model.step(&z0, &TransitionObservation::with_noise(d, 1, c, -0.01), None);
    let (z2_t, g2_rev, _) = model.step(&z1_t, &TransitionObservation::with_noise(c, 2, b, -0.01), None);
    let (z3_trans, g3_rev, _) = model.step(&z2_t, &TransitionObservation::with_noise(b, 1, a, -0.01), None);
    let k3_transposition_score = model.query_composition(&z3_trans, (a, d)) - model.query_composition(&z3_trans, (d, a));
    let k3_transposition_passed = k3_transposition_score < 0.0;

    // Shuffle [e2, e3, e1]: (B, 2, C) -> (C, 1, D) -> (A, 1, B)
    let (z1_s, _, _) = model.step(&z0, &TransitionObservation::new(b, 2, c), None);
    let (z2_s, _, _) = model.step(&z1_s, &TransitionObservation::new(c, 1, d), None);
    let (z3_shuf, _, _) = model.step(&z2_s, &TransitionObservation::new(a, 1, b), None);
    let shuf_score = model.query_composition(&z3_shuf, (a, d)) - model.query_composition(&z3_shuf, (d, a));
    let k3_shuffle_passed = k3_margin > shuf_score;

    // Task-Aligned Jacobian Sensitivity
    let q_ad_s = a as f32 / 5.0;
    let q_ad_d = d as f32 / 5.0;
    let mut dm_dz3 = vec![0.0f32; HIDDEN_DIM];
    for i in 0..HIDDEN_DIM {
        let e_fwd = model.w_q[i * QUERY_DIM] * q_ad_s + model.w_q[i * QUERY_DIM + 1] * q_ad_d;
        let e_rev = model.w_q[i * QUERY_DIM] * q_ad_d + model.w_q[i * QUERY_DIM + 1] * q_ad_s;
        dm_dz3[i] = model.w_r[i] * (e_fwd - e_rev);
    }

    // Vector Gate J1 = diag(1 - g1) * diag(1 - zt1^2) * W_x
    let mut j1 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = (1.0 - g1_i[i]) * (1.0 - zt1_i[i] * zt1_i[i]);
        for j in 0..OBS_DIM {
            j1[i * OBS_DIM + j] = dt * model.w_x[i * OBS_DIM + j];
        }
    }

    // J2 = A2 * J1
    let mut j2 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = (1.0 - g2_i[i]) * (1.0 - zt2_i[i] * zt2_i[i]);
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j1[m * OBS_DIM + l];
            }
            j2[i * OBS_DIM + l] = g2_i[i] * j1[i * OBS_DIM + l] + dt * sum;
        }
    }

    // J3 = A3 * J2
    let mut j3 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = (1.0 - g3_i[i]) * (1.0 - zt3_i[i] * zt3_i[i]);
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j2[m * OBS_DIM + l];
            }
            j3[i * OBS_DIM + l] = g3_i[i] * j2[i * OBS_DIM + l] + dt * sum;
        }
    }

    let mut dm_dx1 = vec![0.0f32; OBS_DIM];
    for j in 0..OBS_DIM {
        let mut sum = 0.0f32;
        for i in 0..HIDDEN_DIM {
            sum += dm_dz3[i] * j3[i * OBS_DIM + j];
        }
        dm_dx1[j] = sum;
    }
    let s_early = dm_dx1.iter().map(|&v| v * v).sum::<f32>().sqrt();

    let mut dm_dx3 = vec![0.0f32; OBS_DIM];
    for j in 0..OBS_DIM {
        let mut sum = 0.0f32;
        for i in 0..HIDDEN_DIM {
            let dt = (1.0 - g3_i[i]) * (1.0 - zt3_i[i] * zt3_i[i]);
            sum += dm_dz3[i] * dt * model.w_x[i * OBS_DIM + j];
        }
        dm_dx3[j] = sum;
    }
    let s_late = dm_dx3.iter().map(|&v| v * v).sum::<f32>().sqrt();

    // Mean Clamped Gate Ablation
    let mut g_bar = vec![0.0f32; HIDDEN_DIM];
    for i in 0..HIDDEN_DIM {
        g_bar[i] = (g1_i[i] + g2_i[i] + g3_i[i]) / 3.0;
    }
    let (z1_c, _, _) = model.step(&z0, &TransitionObservation::new(a, 1, b), Some(&g_bar));
    let (z2_c, _, _) = model.step(&z1_c, &TransitionObservation::new(b, 2, c), Some(&g_bar));
    let (z3_c, _, _) = model.step(&z2_c, &TransitionObservation::new(c, 1, d), Some(&g_bar));
    let k3_margin_clamped = model.query_composition(&z3_c, (a, d)) - model.query_composition(&z3_c, (d, a));
    let k3_passed_clamped = k3_margin_clamped > 0.0;

    let (z1_dc, _, _) = model.step(&z0, &TransitionObservation::with_noise(d, 1, c, 0.01), Some(&g_bar));
    let (z2_dc, _, _) = model.step(&z1_dc, &TransitionObservation::with_noise(c, 2, b, 0.01), Some(&g_bar));
    let (z3_dc, _, _) = model.step(&z2_dc, &TransitionObservation::with_noise(b, 1, a, 0.01), Some(&g_bar));
    let k3_transplant_clamped = model.query_composition(&z3_dc, (a, d)) - model.query_composition(&z3_dc, (d, a));
    let k3_surgery_clamped = k3_margin_clamped > 0.0 && k3_transplant_clamped < 0.0;

    // Sensor task
    let mut rng_sensor = ChaCha8Rng::seed_from_u64(seed ^ 0x66666666);
    let mut sensor_correct = 0;
    for trial_id in 0..20 {
        let is_gold_valid = trial_id < 10;
        let cue_feat = if is_gold_valid {
            0.4 + rng_sensor.gen::<f32>() * 0.2
        } else {
            -3.5 - rng_sensor.gen::<f32>() * 0.5
        };
        let prob = model.query_sensor_trial(&z3_i, cue_feat);
        if (prob >= 0.5) == is_gold_valid {
            sensor_correct += 1;
        }
    }
    let sensor_accuracy = sensor_correct as f32 / 20.0;

    let mean_g1: f32 = g1_i.iter().sum::<f32>() / HIDDEN_DIM as f32;
    let mean_g2: f32 = g2_i.iter().sum::<f32>() / HIDDEN_DIM as f32;
    let mean_g3: f32 = g3_i.iter().sum::<f32>() / HIDDEN_DIM as f32;
    let mean_g1_rev: f32 = g1_rev.iter().sum::<f32>() / HIDDEN_DIM as f32;
    let mean_g2_rev: f32 = g2_rev.iter().sum::<f32>() / HIDDEN_DIM as f32;
    let mean_g3_rev: f32 = g3_rev.iter().sum::<f32>() / HIDDEN_DIM as f32;

    GatedEvalResult {
        seed_index,
        model_type: "VECTOR_UGRNN".to_string(),
        k2_passed,
        k2_margin,
        k3_margin,
        k3_passed,
        k3_transplant_margin,
        k3_swap_effect,
        k3_surgery_transferred,
        k3_transposition_score,
        k3_transposition_passed,
        k3_shuffle_passed,
        s_early,
        s_late,
        k3_margin_clamped,
        k3_passed_clamped,
        k3_surgery_clamped,
        gate_trajectory_intact: vec![mean_g1, mean_g2, mean_g3],
        gate_trajectory_reversed: vec![mean_g1_rev, mean_g2_rev, mean_g3_rev],
        sensor_accuracy,
    }
}

fn main() {
    println!("=================================================================================================================================");
    println!("SCOUT Q17E-C: Minimal Adaptive Gating Comparison (16 Auxiliary Seeds, Exact 120-Epoch Training)");
    println!("Testing: A. Adaptive Scalar Gate vs B. Adaptive Vector Update Gate (UGRNN)");
    println!("=================================================================================================================================");

    let scalar_results: Vec<GatedEvalResult> = (1..=16)
        .into_par_iter()
        .map(evaluate_scalar_seed)
        .collect();

    let vector_results: Vec<GatedEvalResult> = (1..=16)
        .into_par_iter()
        .map(evaluate_vector_seed)
        .collect();

    for (label, results) in &[("A. ADAPTIVE SCALAR GATE", &scalar_results), ("B. ADAPTIVE VECTOR UGRNN", &vector_results)] {
        println!("\n--- {} ---", label);
        let k2_pass_count = results.iter().filter(|r| r.k2_passed).count();
        let k3_pass_count = results.iter().filter(|r| r.k3_passed).count();
        let k3_margins: Vec<f32> = results.iter().map(|r| r.k3_margin).collect();
        let k3_p = compute_sign_flip_p_val(&k3_margins);

        let surg_pass_count = results.iter().filter(|r| r.k3_surgery_transferred).count();
        let mean_swap: f32 = results.iter().map(|r| r.k3_swap_effect).sum::<f32>() / 16.0;
        let trans_pass_count = results.iter().filter(|r| r.k3_transposition_passed).count();
        let shuf_pass_count = results.iter().filter(|r| r.k3_shuffle_passed).count();

        let mean_s_early: f32 = results.iter().map(|r| r.s_early).sum::<f32>() / 16.0;
        let mean_s_late: f32 = results.iter().map(|r| r.s_late).sum::<f32>() / 16.0;

        let clamped_k3_pass = results.iter().filter(|r| r.k3_passed_clamped).count();
        let clamped_surg_pass = results.iter().filter(|r| r.k3_surgery_clamped).count();

        let sensor_pass_count = results.iter().filter(|r| r.sensor_accuracy >= 0.90).count();

        println!("  k=2 Baseline Retention:                   {}/16 ({:.1}%)", k2_pass_count, k2_pass_count as f32 / 16.0 * 100.0);
        println!("  k=3 Positive Direction (m_3 > 0):         {}/16 (p={:.4})", k3_pass_count, k3_p);
        println!("  k=3 Independent Cloned-Twin State Surgery: {}/16 ({:.1}%) [Mean Swap Effect: {:+.4}]", surg_pass_count, surg_pass_count as f32 / 16.0 * 100.0, mean_swap);
        println!("  k=3 Transposition Reversals:              {}/16 ({:.1}%)", trans_pass_count, trans_pass_count as f32 / 16.0 * 100.0);
        println!("  k=3 Deranged Shuffle Superiority:         {}/16 ({:.1}%)", shuf_pass_count, shuf_pass_count as f32 / 16.0 * 100.0);
        println!("  Task-Aligned Early Sensitivity (S_early): {:.4}", mean_s_early);
        println!("  Task-Aligned Last-Edge Sensitivity (S_late): {:.4}", mean_s_late);
        println!("  ABLATION: Clamped Static Gate k=3 Direction: {}/16 (Dynamic: {}/16)", clamped_k3_pass, k3_pass_count);
        println!("  ABLATION: Clamped Static Gate State Surgery: {}/16 (Dynamic: {}/16)", clamped_surg_pass, surg_pass_count);
        println!("  1-Hop Sensor Accuracy (>= 90%):           {}/16 ({:.1}%)", sensor_pass_count, sensor_pass_count as f32 / 16.0 * 100.0);

        let mean_g1: f32 = results.iter().map(|r| r.gate_trajectory_intact[0]).sum::<f32>() / 16.0;
        let mean_g2: f32 = results.iter().map(|r| r.gate_trajectory_intact[1]).sum::<f32>() / 16.0;
        let mean_g3: f32 = results.iter().map(|r| r.gate_trajectory_intact[2]).sum::<f32>() / 16.0;
        let mean_g1_r: f32 = results.iter().map(|r| r.gate_trajectory_reversed[0]).sum::<f32>() / 16.0;
        let mean_g2_r: f32 = results.iter().map(|r| r.gate_trajectory_reversed[1]).sum::<f32>() / 16.0;
        let mean_g3_r: f32 = results.iter().map(|r| r.gate_trajectory_reversed[2]).sum::<f32>() / 16.0;
        println!("  Mean Gate Trajectory (Intact Stream):     [g1: {:.3}, g2: {:.3}, g3: {:.3}]", mean_g1, mean_g2, mean_g3);
        println!("  Mean Gate Trajectory (Reversed Stream):   [g1: {:.3}, g2: {:.3}, g3: {:.3}]", mean_g1_r, mean_g2_r, mean_g3_r);
    }
    println!("\n=================================================================================================================================");

    let data_dir = Path::new("data");
    let out_file = data_dir.join("q17e_c_gating_scout_results.json");
    let mut all_results = Vec::new();
    all_results.extend(scalar_results);
    all_results.extend(vector_results);

    let file = File::create(&out_file).expect("Failed to create gating scout results JSON");
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &all_results).expect("Failed to write JSON");
    println!("Persisted full Scout C gating telemetry to: {}", out_file.display());
}
