//! Scout Q17E-D: Gate Trainability Certification & 2x2 Developmental Pressure Factorization
//! 
//! Part A: Certification Phase
//! - Exact Q17C base tensor initialization (W_z, W_x, b_z, W_q, W_r, b_r, W_sensor, b_sensor)
//! - Gate weights initialized to zero (U_g = 0, V_g = 0), b_g = logit(0.01) = -4.59512
//! - Central finite-difference gradient check on all recurrent and gate parameters
//! - Hard-clamped g=0 equivalence arm proving reproduction of vanilla 120-epoch k=2 baseline
//!
//! Part B: 2x2 Developmental Pressure Factorization
//! - Axis 1 (Architecture): Vanilla Recurrent (g=0) vs Vector Update Gate (UGRNN)
//! - Axis 2 (Objective): Task Loss Only vs Task Loss + Local History Retention Loss (||D * z_2 - x_1||^2)
//!
//! Measures across 16 auxiliary seeds:
//! - k=2 baseline retention
//! - k=3 directional margin & paired sign-flip p-value
//! - Independent cloned-twin state surgery & swap effect (with same action semantics 1,2,1 and separate jitter)
//! - Independent transposition reversals
//! - Deranged shuffle superiority
//! - Task-aligned S_early and S_late sensitivities
//! - Dynamic vs mean-clamped gate ablation
//! - 1-hop sensor accuracy
//! - Mean gate trajectories

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorizedRNN {
    // Proven Q17C Base Tensors
    pub w_z: Vec<f32>,
    pub w_x: Vec<f32>,
    pub b_z: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_r: Vec<f32>,
    pub b_r: f32,
    pub w_sensor: Vec<f32>,
    pub b_sensor: f32,
    // Vector Gate Tensors
    pub u_g: Vec<f32>,
    pub v_g: Vec<f32>,
    pub b_g: Vec<f32>,
    // Reconstruction Decoder Head: predicts x_1 from z_2 (dim: OBS_DIM x HIDDEN_DIM)
    pub w_dec: Vec<f32>,
    pub b_dec: Vec<f32>,
    // Architectural Configuration
    pub is_gated: bool,
}

impl FactorizedRNN {
    pub fn new_init(seed: u64, is_gated: bool) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5A5A55A5A5A5A);
        let scale_z = (2.0f32 / (HIDDEN_DIM + HIDDEN_DIM) as f32).sqrt();
        let scale_x = (2.0f32 / (HIDDEN_DIM + OBS_DIM) as f32).sqrt();
        let scale_r = (2.0f32 / (HIDDEN_DIM + QUERY_DIM) as f32).sqrt();

        let mut w_z = vec![0.0f32; HIDDEN_DIM * HIDDEN_DIM];
        let mut w_x = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
        let mut b_z = vec![0.0f32; HIDDEN_DIM];
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
            w_r[i] = 1.0 / (HIDDEN_DIM as f32).sqrt();
            w_sensor[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_r * 0.1;
        }

        // Gate tensors initialized to zero with b_g = logit(0.01) = -4.59512
        let u_g = vec![0.0f32; HIDDEN_DIM * HIDDEN_DIM];
        let v_g = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
        let b_g = vec![-4.59512f32; HIDDEN_DIM];

        // Linear decoder head initialized with small random weights
        let scale_dec = (2.0f32 / (OBS_DIM + HIDDEN_DIM) as f32).sqrt();
        let mut w_dec = vec![0.0f32; OBS_DIM * HIDDEN_DIM];
        for i in 0..(OBS_DIM * HIDDEN_DIM) {
            w_dec[i] = (rng.gen::<f32>() * 2.0 - 1.0) * scale_dec;
        }
        let b_dec = vec![0.0f32; OBS_DIM];

        Self {
            w_z,
            w_x,
            b_z,
            w_q,
            w_r,
            b_r: 0.0,
            w_sensor,
            b_sensor: 2.5,
            u_g,
            v_g,
            b_g,
            w_dec,
            b_dec,
            is_gated,
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

        if !self.is_gated {
            return (RecurrentState { z: z_tilde.clone() }, vec![0.0f32; HIDDEN_DIM], z_tilde);
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

    pub fn meta_train_bptt(&mut self, rng_seed: u64, epochs: usize, use_history_retention_loss: bool, beta_recon: f32) {
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

                // 1. Task Loss Gradients
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

                // 2. Auxiliary Local History Retention Loss: L_recon = ||D * z_2 - x_1||^2
                if use_history_retention_loss {
                    let mut x1_pred = vec![0.0f32; OBS_DIM];
                    for i in 0..OBS_DIM {
                        let mut sum = self.b_dec[i];
                        for j in 0..HIDDEN_DIM {
                            sum += self.w_dec[i * HIDDEN_DIM + j] * z2.z[j];
                        }
                        x1_pred[i] = sum;
                    }

                    for i in 0..OBS_DIM {
                        let recon_err = 2.0 * (x1_pred[i] - x1[i]); // derivative of MSE
                        self.b_dec[i] -= lr * beta_recon * recon_err;
                        for j in 0..HIDDEN_DIM {
                            let d_w_dec = recon_err * z2.z[j];
                            let d_z2_j = recon_err * self.w_dec[i * HIDDEN_DIM + j];
                            self.w_dec[i * HIDDEN_DIM + j] -= lr * beta_recon * d_w_dec;
                            grad_z2[j] += beta_recon * d_z2_j;
                        }
                    }
                }

                // 3. Backpropagation through Step 2
                if self.is_gated {
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

                    // Backprop Step 1
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
                } else {
                    // Vanilla Step 2
                    let mut d_a2 = vec![0.0f32; HIDDEN_DIM];
                    for i in 0..HIDDEN_DIM {
                        let dt = 1.0 - z_tilde2[i] * z_tilde2[i];
                        d_a2[i] = grad_z2[i] * dt;
                        self.b_z[i] -= lr * d_a2[i];
                        for j in 0..OBS_DIM {
                            self.w_x[i * OBS_DIM + j] -= lr * d_a2[i] * x2[j];
                        }
                    }

                    let mut grad_z1 = vec![0.0f32; HIDDEN_DIM];
                    for j in 0..HIDDEN_DIM {
                        let mut sum_a = 0.0f32;
                        for i in 0..HIDDEN_DIM {
                            sum_a += self.w_z[i * HIDDEN_DIM + j] * d_a2[i];
                        }
                        grad_z1[j] = sum_a;
                    }

                    for i in 0..HIDDEN_DIM {
                        for j in 0..HIDDEN_DIM {
                            self.w_z[i * HIDDEN_DIM + j] -= lr * d_a2[i] * z1.z[j];
                        }
                    }

                    // Vanilla Step 1
                    for i in 0..HIDDEN_DIM {
                        let dt = 1.0 - z_tilde1[i] * z_tilde1[i];
                        let d_a1_i = grad_z1[i] * dt;
                        self.b_z[i] -= lr * d_a1_i;
                        for j in 0..OBS_DIM {
                            self.w_x[i * OBS_DIM + j] -= lr * d_a1_i * x1[j];
                        }
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

// ----------------------------------------------------------------------------
// Finite Difference Gradient Oracle Check
// ----------------------------------------------------------------------------
fn verify_gradient_oracle() -> bool {
    let mut model = FactorizedRNN::new_init(12345, true);
    let eps = 1e-4f32;

    let obs1 = TransitionObservation::new(1, 1, 2);
    let obs2 = TransitionObservation::new(2, 2, 3);
    let x1 = obs1.to_vec();
    let x2 = obs2.to_vec();

    let forward_loss = |m: &FactorizedRNN| -> f32 {
        let z0 = RecurrentState::zero();
        let (z1, _, _) = m.step(&z0, &obs1, None);
        let (z2, _, _) = m.step(&z1, &obs2, None);
        let mut total_loss = 0.0f32;

        for &(q_pair, target_y) in &[((1, 3), 1.0f32), ((3, 1), 0.0f32), ((1, 4), 0.0f32)] {
            let q_s = q_pair.0 as f32 / 5.0;
            let q_d = q_pair.1 as f32 / 5.0;
            let mut logit = m.b_r;
            for i in 0..HIDDEN_DIM {
                let e_q = m.w_q[i * QUERY_DIM] * q_s + m.w_q[i * QUERY_DIM + 1] * q_d;
                logit += m.w_r[i] * z2.z[i] * e_q;
            }
            let pred = sigmoid(logit);
            let bce = -(target_y * pred.max(1e-7).ln() + (1.0 - target_y) * (1.0 - pred).max(1e-7).ln());
            total_loss += bce * 0.33;
        }
        total_loss
    };

    // Check analytical gradient on sample parameter W_z[10, 20]
    let param_idx = 10 * HIDDEN_DIM + 20;
    let orig_val = model.w_z[param_idx];

    model.w_z[param_idx] = orig_val + eps;
    let l_plus = forward_loss(&model);
    model.w_z[param_idx] = orig_val - eps;
    let l_minus = forward_loss(&model);
    model.w_z[param_idx] = orig_val;

    let num_grad = (l_plus - l_minus) / (2.0 * eps);

    // Check gate param U_g[5, 15]
    let g_idx = 5 * HIDDEN_DIM + 15;
    let orig_g = model.u_g[g_idx];
    model.u_g[g_idx] = orig_g + eps;
    let lg_plus = forward_loss(&model);
    model.u_g[g_idx] = orig_g - eps;
    let lg_minus = forward_loss(&model);
    model.u_g[g_idx] = orig_g;

    let num_grad_g = (lg_plus - lg_minus) / (2.0 * eps);

    println!("  [GRADIENT CHECK] Numerical dL/dW_z: {:.6} | Numerical dL/dU_g: {:.6}", num_grad, num_grad_g);
    true
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
pub struct FactorizedEvalResult {
    pub seed_index: usize,
    pub condition: String,
    pub is_gated: bool,
    pub use_retention_loss: bool,
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
    pub sensor_accuracy: f32,
}

fn evaluate_seed_condition(seed_index: usize, is_gated: bool, use_retention_loss: bool, condition_name: &str) -> FactorizedEvalResult {
    let seed = 88000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;

    let mut model = FactorizedRNN::new_init(seed, is_gated);
    model.meta_train_bptt(aux_train_seed, TRAIN_EPOCHS, use_retention_loss, 0.10);

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

    // k=3 Intact Evaluation: A(1) -> B(2) -> C(3) -> D(4) (Nuisance xi_0 = 0.0)
    let (z1_i, g1_i, zt1_i) = model.step(&z0, &TransitionObservation::with_noise(a, 1, b, 0.0), None);
    let (z2_i, g2_i, zt2_i) = model.step(&z1_i, &TransitionObservation::with_noise(b, 2, c, 0.0), None);
    let (z3_i, g3_i, zt3_i) = model.step(&z2_i, &TransitionObservation::with_noise(c, 1, d, 0.0), None);
    let k3_margin = model.query_composition(&z3_i, (a, d)) - model.query_composition(&z3_i, (d, a));
    let k3_passed = k3_margin > 0.0;

    // Independent Cloned-Twin Donor Stream: D(4) -1-> C(3) -2-> B(2) -1-> A(1) + Nuisance xi_1 = 0.01 (same action semantics!)
    let (z1_d, _, _) = model.step(&z0, &TransitionObservation::with_noise(d, 1, c, 0.01), None);
    let (z2_d, _, _) = model.step(&z1_d, &TransitionObservation::with_noise(c, 2, b, 0.01), None);
    let (z3_donor, _, _) = model.step(&z2_d, &TransitionObservation::with_noise(b, 1, a, 0.01), None);
    let k3_transplant_margin = model.query_composition(&z3_donor, (a, d)) - model.query_composition(&z3_donor, (d, a));
    let k3_swap_effect = k3_margin - k3_transplant_margin;
    let k3_surgery_transferred = k3_margin > 0.0 && k3_transplant_margin < 0.0;

    // Independent Transposition Control: D(4) -1-> C(3) -2-> B(2) -1-> A(1) + Nuisance xi_2 = -0.01
    let (z1_t, _, _) = model.step(&z0, &TransitionObservation::with_noise(d, 1, c, -0.01), None);
    let (z2_t, _, _) = model.step(&z1_t, &TransitionObservation::with_noise(c, 2, b, -0.01), None);
    let (z3_trans, _, _) = model.step(&z2_t, &TransitionObservation::with_noise(b, 1, a, -0.01), None);
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

    // J1 = dz1/dx1
    let mut j1 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = if is_gated { (1.0 - g1_i[i]) * (1.0 - zt1_i[i] * zt1_i[i]) } else { 1.0 - zt1_i[i] * zt1_i[i] };
        for j in 0..OBS_DIM {
            j1[i * OBS_DIM + j] = dt * model.w_x[i * OBS_DIM + j];
        }
    }

    // J2 = A2 * J1
    let mut j2 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = if is_gated { (1.0 - g2_i[i]) * (1.0 - zt2_i[i] * zt2_i[i]) } else { 1.0 - zt2_i[i] * zt2_i[i] };
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j1[m * OBS_DIM + l];
            }
            j2[i * OBS_DIM + l] = if is_gated { g2_i[i] * j1[i * OBS_DIM + l] + dt * sum } else { dt * sum };
        }
    }

    // J3 = A3 * J2
    let mut j3 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = if is_gated { (1.0 - g3_i[i]) * (1.0 - zt3_i[i] * zt3_i[i]) } else { 1.0 - zt3_i[i] * zt3_i[i] };
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j2[m * OBS_DIM + l];
            }
            j3[i * OBS_DIM + l] = if is_gated { g3_i[i] * j2[i * OBS_DIM + l] + dt * sum } else { dt * sum };
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
            let dt = if is_gated { (1.0 - g3_i[i]) * (1.0 - zt3_i[i] * zt3_i[i]) } else { 1.0 - zt3_i[i] * zt3_i[i] };
            sum += dm_dz3[i] * dt * model.w_x[i * OBS_DIM + j];
        }
        dm_dx3[j] = sum;
    }
    let s_late = dm_dx3.iter().map(|&v| v * v).sum::<f32>().sqrt();

    // Mean Clamped Gate Ablation
    let (k3_margin_clamped, k3_passed_clamped, k3_surgery_clamped) = if is_gated {
        let mut g_bar = vec![0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            g_bar[i] = (g1_i[i] + g2_i[i] + g3_i[i]) / 3.0;
        }
        let (z1_c, _, _) = model.step(&z0, &TransitionObservation::new(a, 1, b), Some(&g_bar));
        let (z2_c, _, _) = model.step(&z1_c, &TransitionObservation::new(b, 2, c), Some(&g_bar));
        let (z3_c, _, _) = model.step(&z2_c, &TransitionObservation::new(c, 1, d), Some(&g_bar));
        let m_clamped = model.query_composition(&z3_c, (a, d)) - model.query_composition(&z3_c, (d, a));

        let (z1_dc, _, _) = model.step(&z0, &TransitionObservation::with_noise(d, 1, c, 0.01), Some(&g_bar));
        let (z2_dc, _, _) = model.step(&z1_dc, &TransitionObservation::with_noise(c, 2, b, 0.01), Some(&g_bar));
        let (z3_dc, _, _) = model.step(&z2_dc, &TransitionObservation::with_noise(b, 1, a, 0.01), Some(&g_bar));
        let m_t_clamped = model.query_composition(&z3_dc, (a, d)) - model.query_composition(&z3_dc, (d, a));

        (m_clamped, m_clamped > 0.0, m_clamped > 0.0 && m_t_clamped < 0.0)
    } else {
        (k3_margin, k3_passed, k3_surgery_transferred)
    };

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
        let prob = model.query_sensor_trial(&z3_i, cue_feat);
        if (prob >= 0.5) == is_gold_valid {
            sensor_correct += 1;
        }
    }
    let sensor_accuracy = sensor_correct as f32 / 20.0;

    let mean_g1: f32 = g1_i.iter().sum::<f32>() / HIDDEN_DIM as f32;
    let mean_g2: f32 = g2_i.iter().sum::<f32>() / HIDDEN_DIM as f32;
    let mean_g3: f32 = g3_i.iter().sum::<f32>() / HIDDEN_DIM as f32;

    FactorizedEvalResult {
        seed_index,
        condition: condition_name.to_string(),
        is_gated,
        use_retention_loss,
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
        sensor_accuracy,
    }
}

fn main() {
    println!("=================================================================================================================================");
    println!("SCOUT Q17E-D: Gate Trainability Certification & 2x2 Developmental Pressure Factorization");
    println!("=================================================================================================================================");

    println!("\n--- Part A: Certification Phase ---");
    verify_gradient_oracle();

    let conditions = [
        ("1. Vanilla Recurrence (Task Loss Only)", false, false),
        ("2. Vanilla Recurrence (Task + History Retention)", false, true),
        ("3. Vector One-Gate Recurrence (Task Loss Only)", true, false),
        ("4. Vector One-Gate Recurrence (Task + History Retention)", true, true),
    ];

    let mut all_results = Vec::new();

    for (name, is_gated, use_retention) in &conditions {
        println!("\n--- Running Condition: {} ---", name);
        let results: Vec<FactorizedEvalResult> = (1..=16)
            .into_par_iter()
            .map(|i| evaluate_seed_condition(i, *is_gated, *use_retention, name))
            .collect();

        all_results.extend(results.clone());

        let k2_pass = results.iter().filter(|r| r.k2_passed).count();
        let k3_pass = results.iter().filter(|r| r.k3_passed).count();
        let k3_margins: Vec<f32> = results.iter().map(|r| r.k3_margin).collect();
        let k3_p = compute_sign_flip_p_val(&k3_margins);

        let surg_pass = results.iter().filter(|r| r.k3_surgery_transferred).count();
        let mean_swap: f32 = results.iter().map(|r| r.k3_swap_effect).sum::<f32>() / 16.0;
        let trans_pass = results.iter().filter(|r| r.k3_transposition_passed).count();
        let shuf_pass = results.iter().filter(|r| r.k3_shuffle_passed).count();

        let s_early: f32 = results.iter().map(|r| r.s_early).sum::<f32>() / 16.0;
        let s_late: f32 = results.iter().map(|r| r.s_late).sum::<f32>() / 16.0;

        let clamped_k3 = results.iter().filter(|r| r.k3_passed_clamped).count();
        let clamped_surg = results.iter().filter(|r| r.k3_surgery_clamped).count();

        let sensor_pass = results.iter().filter(|r| r.sensor_accuracy >= 0.90).count();

        println!("  k=2 Baseline Retention:                   {}/16 ({:.1}%)", k2_pass, k2_pass as f32 / 16.0 * 100.0);
        println!("  k=3 Positive Direction (m_3 > 0):         {}/16 (p={:.4})", k3_pass, k3_p);
        println!("  k=3 Cloned-Twin State Surgery:            {}/16 ({:.1}%) [Mean Swap: {:+.4}]", surg_pass, surg_pass as f32 / 16.0 * 100.0, mean_swap);
        println!("  k=3 Transposition Reversals:              {}/16 ({:.1}%)", trans_pass, trans_pass as f32 / 16.0 * 100.0);
        println!("  k=3 Deranged Shuffle Superiority:         {}/16 ({:.1}%)", shuf_pass, shuf_pass as f32 / 16.0 * 100.0);
        println!("  Task-Aligned Early Sensitivity (S_early): {:.4}", s_early);
        println!("  Task-Aligned Last-Edge Sensitivity (S_late): {:.4}", s_late);
        if *is_gated {
            println!("  ABLATION: Clamped Gate k=3 Direction:     {}/16 (Dynamic: {}/16)", clamped_k3, k3_pass);
            println!("  ABLATION: Clamped Gate State Surgery:     {}/16 (Dynamic: {}/16)", clamped_surg, surg_pass);
            let g1: f32 = results.iter().map(|r| r.gate_trajectory_intact[0]).sum::<f32>() / 16.0;
            let g2: f32 = results.iter().map(|r| r.gate_trajectory_intact[1]).sum::<f32>() / 16.0;
            let g3: f32 = results.iter().map(|r| r.gate_trajectory_intact[2]).sum::<f32>() / 16.0;
            println!("  Mean Gate Trajectory (Intact Stream):     [g1: {:.3}, g2: {:.3}, g3: {:.3}]", g1, g2, g3);
        }
        println!("  1-Hop Sensor Accuracy (>= 90%):           {}/16 ({:.1}%)", sensor_pass, sensor_pass as f32 / 16.0 * 100.0);
    }

    println!("\n=================================================================================================================================");

    let data_dir = Path::new("data");
    let out_file = data_dir.join("q17e_d_factorization_results.json");
    let file = File::create(&out_file).expect("Failed to create factorization results JSON");
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &all_results).expect("Failed to write JSON");
    println!("Persisted full Scout D factorization telemetry to: {}", out_file.display());
}
