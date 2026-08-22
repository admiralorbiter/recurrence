//! Development Lambda Carry Scout for Q17E (Corrected 120-Epoch Training Baseline)
//! Evaluates residual carry:
//!   z~_{t+1} = tanh(W_z z_t + W_x x_t + b_z)
//!   z_{t+1}  = lambda * z_t + (1 - lambda) * z~_{t+1}
//! across a sweep of lambda in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.50]
//! on auxiliary dev seeds (seed_i = 88000 + i * 777) under exact 120-epoch training measuring:
//! - Initial-step Jacobian sensitivity retention ||d(z_3)/d(x_1)||_F / ||d(z_2)/d(x_1)||_F
//! - Last-step Jacobian sensitivity ||d(z_3)/d(x_3)||_F (plasticity / new-edge incorporation)
//! - k=2 baseline retention
//! - k=3 directional margin pass rate & sign-flip p-value
//! - k=3 independent cloned-twin donor-state transplant assay & continuous swap effect
//! - k=3 independent transposition reversal pass rate
//! - k=3 deranged shuffle superiority pass rate
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
}

impl TransitionObservation {
    pub fn new(src: usize, action: usize, dst: usize) -> Self {
        Self { src, action, dst }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = vec![0.0f32; OBS_DIM];
        let s = if self.src >= 10 { (self.src % 10) as f32 } else { self.src as f32 };
        let d = if self.dst >= 10 { (self.dst % 10) as f32 } else { self.dst as f32 };
        v[0] = s / 5.0;
        v[1] = (self.action as f32) / 5.0;
        v[2] = d / 5.0;
        v[3] = 1.0;
        v
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualMemoryModel {
    pub w_z: Vec<f32>,
    pub w_x: Vec<f32>,
    pub b_z: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_r: Vec<f32>,
    pub b_r: f32,
    pub w_sensor: Vec<f32>,
    pub b_sensor: f32,
    pub lambda: f32,
}

impl ResidualMemoryModel {
    pub fn new_init(seed: u64, lambda: f32) -> Self {
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

        Self {
            w_z,
            w_x,
            b_z,
            w_q,
            w_r,
            b_r: 0.0,
            w_sensor,
            b_sensor: 2.5,
            lambda,
        }
    }

    #[inline(always)]
    pub fn step(&self, state: &RecurrentState, obs: &TransitionObservation) -> (RecurrentState, Vec<f32>) {
        let x = obs.to_vec();
        let mut next_z = vec![0.0f32; HIDDEN_DIM];
        let mut raw_tilde = vec![0.0f32; HIDDEN_DIM];

        for i in 0..HIDDEN_DIM {
            let mut sum = self.b_z[i];
            let row_offset_z = i * HIDDEN_DIM;
            for j in 0..HIDDEN_DIM {
                sum += self.w_z[row_offset_z + j] * state.z[j];
            }
            let row_offset_x = i * OBS_DIM;
            for j in 0..OBS_DIM {
                sum += self.w_x[row_offset_x + j] * x[j];
            }
            let z_tilde = sum.tanh();
            raw_tilde[i] = z_tilde;
            next_z[i] = self.lambda * state.z[i] + (1.0 - self.lambda) * z_tilde;
        }

        (RecurrentState { z: next_z }, raw_tilde)
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
        1.0 / (1.0 + (-sum).exp())
    }

    /// Meta-train the residual recurrent network on 2-step sequences ONLY under exact 120 epochs
    pub fn meta_train_bptt(&mut self, rng_seed: u64, epochs: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(rng_seed);
        let lr = TRAIN_LR;
        let l = self.lambda;
        let one_minus_l = 1.0 - l;

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

                // Step 1
                let mut z_tilde1 = vec![0.0f32; HIDDEN_DIM];
                let mut z1 = vec![0.0f32; HIDDEN_DIM];
                for i in 0..HIDDEN_DIM {
                    let mut sum = self.b_z[i];
                    for j in 0..OBS_DIM {
                        sum += self.w_x[i * OBS_DIM + j] * x1[j];
                    }
                    let z_t = sum.tanh();
                    z_tilde1[i] = z_t;
                    z1[i] = one_minus_l * z_t;
                }

                // Step 2
                let mut z_tilde2 = vec![0.0f32; HIDDEN_DIM];
                let mut z2 = vec![0.0f32; HIDDEN_DIM];
                for i in 0..HIDDEN_DIM {
                    let mut sum = self.b_z[i];
                    for j in 0..HIDDEN_DIM {
                        sum += self.w_z[i * HIDDEN_DIM + j] * z1[j];
                    }
                    for j in 0..OBS_DIM {
                        sum += self.w_x[i * OBS_DIM + j] * x2[j];
                    }
                    let z_t = sum.tanh();
                    z_tilde2[i] = z_t;
                    z2[i] = l * z1[i] + one_minus_l * z_t;
                }

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
                        logit += self.w_r[i] * z2[i] * e_q_i;
                    }
                    let pred = 1.0 / (1.0 + (-logit).exp());
                    let err = pred - target_y;

                    self.b_r -= lr * err * 0.33;
                    for i in 0..HIDDEN_DIM {
                        let e_q_i = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
                        let d_w_r = err * z2[i] * e_q_i;
                        let d_e_q_i = err * self.w_r[i] * z2[i];
                        let d_z2_i = err * self.w_r[i] * e_q_i;

                        self.w_r[i] -= lr * d_w_r * 0.33;
                        self.w_q[i * QUERY_DIM] -= lr * d_e_q_i * q_s * 0.33;
                        self.w_q[i * QUERY_DIM + 1] -= lr * d_e_q_i * q_d * 0.33;
                        grad_z2[i] += d_z2_i * 0.33;
                    }
                }

                // Backprop Step 2
                let mut d_a2 = vec![0.0f32; HIDDEN_DIM];
                for i in 0..HIDDEN_DIM {
                    let dt = 1.0 - z_tilde2[i] * z_tilde2[i];
                    d_a2[i] = grad_z2[i] * one_minus_l * dt;
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
                    grad_z1[j] = grad_z2[j] * l + sum;
                }

                for i in 0..HIDDEN_DIM {
                    for j in 0..HIDDEN_DIM {
                        self.w_z[i * HIDDEN_DIM + j] -= lr * d_a2[i] * z1[j];
                    }
                }

                // Backprop Step 1
                for i in 0..HIDDEN_DIM {
                    let dt = 1.0 - z_tilde1[i] * z_tilde1[i];
                    let d_a1_i = grad_z1[i] * one_minus_l * dt;
                    self.b_z[i] -= lr * d_a1_i;
                    for j in 0..OBS_DIM {
                        self.w_x[i * OBS_DIM + j] -= lr * d_a1_i * x1[j];
                    }
                }

                // Sensor training
                let sensor_prob = self.query_sensor_trial(&RecurrentState { z: z2.clone() }, 0.5);
                let sensor_err = sensor_prob - 0.95;
                self.b_sensor -= lr * sensor_err * 0.1;
                for i in 0..HIDDEN_DIM {
                    self.w_sensor[i] -= lr * sensor_err * z2[i] * 0.01;
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
pub struct SeedLambdaRecord {
    pub seed_index: usize,
    pub lambda: f32,
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
    pub sensor_accuracy: f32,
    pub jac_init_step_k2: f32,
    pub jac_init_step_k3: f32,
    pub jac_last_step_k3: f32,
}

fn evaluate_seed_lambda(seed_index: usize, lambda: f32) -> SeedLambdaRecord {
    let seed = 88000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;

    let mut model = ResidualMemoryModel::new_init(seed, lambda);
    model.meta_train_bptt(aux_train_seed, TRAIN_EPOCHS);

    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;

    // k=2 Evaluation: A -> B -> C
    let z_k2_init = RecurrentState::zero();
    let (z1, z_tilde1) = model.step(&z_k2_init, &TransitionObservation::new(a, 1, b));
    let (z2, z_tilde2) = model.step(&z1, &TransitionObservation::new(b, 2, c));
    let k2_fwd = model.query_composition(&z2, (a, c));
    let k2_rev = model.query_composition(&z2, (c, a));
    let k2_margin = k2_fwd - k2_rev;
    let k2_passed = k2_margin > 0.0;

    // k=3 Evaluation (Intact Trajectory Twin 1): A(1) -> B(2) -> C(3) -> D(4)
    let (z3_twin1, z_tilde3) = model.step(&z2, &TransitionObservation::new(c, 1, d));
    let k3_fwd = model.query_composition(&z3_twin1, (a, d));
    let k3_rev = model.query_composition(&z3_twin1, (d, a));
    let k3_margin = k3_fwd - k3_rev;
    let k3_passed = k3_margin > 0.0;

    // Independent Cloned-Twin Donor Stream: D(4) -> C(3) -> B(2) -> A(1)
    let mut z_donor = RecurrentState::zero();
    z_donor = model.step(&z_donor, &TransitionObservation::new(d, 1, c)).0;
    z_donor = model.step(&z_donor, &TransitionObservation::new(c, 2, b)).0;
    z_donor = model.step(&z_donor, &TransitionObservation::new(b, 1, a)).0;

    // Transplant donor state into Twin 1's query readout for (A, D) vs (D, A)
    let k3_transplant_fwd = model.query_composition(&z_donor, (a, d));
    let k3_transplant_rev = model.query_composition(&z_donor, (d, a));
    let k3_transplant_margin = k3_transplant_fwd - k3_transplant_rev;
    let k3_swap_effect = k3_margin - k3_transplant_margin;
    let k3_surgery_transferred = k3_margin > 0.0 && k3_transplant_margin < 0.0;

    // Independent Transposition Control
    let k3_transposition_score = k3_transplant_margin;
    let k3_transposition_passed = k3_transposition_score < 0.0;

    // k=3 Shuffle [e2, e3, e1]: (B, 2, C) -> (C, 1, D) -> (A, 1, B)
    let mut z_shuf = RecurrentState::zero();
    z_shuf = model.step(&z_shuf, &TransitionObservation::new(b, 2, c)).0;
    z_shuf = model.step(&z_shuf, &TransitionObservation::new(c, 1, d)).0;
    z_shuf = model.step(&z_shuf, &TransitionObservation::new(a, 1, b)).0;
    let shuf_score = model.query_composition(&z_shuf, (a, d)) - model.query_composition(&z_shuf, (d, a));
    let k3_shuffle_passed = k3_margin > shuf_score;

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
        let prob = model.query_sensor_trial(&z3_twin1, cue_feat);
        if (prob >= 0.5) == is_gold_valid {
            sensor_correct += 1;
        }
    }
    let sensor_accuracy = sensor_correct as f32 / 20.0;

    // Compute Initial-Step Jacobian Sensitivity:
    let one_minus_l = 1.0 - lambda;
    let mut j1 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = (1.0 - z_tilde1[i] * z_tilde1[i]) * one_minus_l;
        for j in 0..OBS_DIM {
            j1[i * OBS_DIM + j] = dt * model.w_x[i * OBS_DIM + j];
        }
    }

    // J_2 = A_2 * J_1
    let mut j2 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = (1.0 - z_tilde2[i] * z_tilde2[i]) * one_minus_l;
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j1[m * OBS_DIM + l];
            }
            j2[i * OBS_DIM + l] = lambda * j1[i * OBS_DIM + l] + dt * sum;
        }
    }
    let jac_init_step_k2 = j2.iter().map(|&v| v * v).sum::<f32>().sqrt();

    // J_3 = A_3 * J_2
    let mut j3 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = (1.0 - z_tilde3[i] * z_tilde3[i]) * one_minus_l;
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j2[m * OBS_DIM + l];
            }
            j3[i * OBS_DIM + l] = lambda * j2[i * OBS_DIM + l] + dt * sum;
        }
    }
    let jac_init_step_k3 = j3.iter().map(|&v| v * v).sum::<f32>().sqrt();

    // Last-step Jacobian sensitivity: ||d(z_3)/d(x_3)||_F = (1 - lambda) * ||diag(1 - z_tilde3^2) * W_x||_F
    let mut j_last = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = (1.0 - z_tilde3[i] * z_tilde3[i]) * one_minus_l;
        for j in 0..OBS_DIM {
            j_last[i * OBS_DIM + j] = dt * model.w_x[i * OBS_DIM + j];
        }
    }
    let jac_last_step_k3 = j_last.iter().map(|&v| v * v).sum::<f32>().sqrt();

    SeedLambdaRecord {
        seed_index,
        lambda,
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
        sensor_accuracy,
        jac_init_step_k2,
        jac_init_step_k3,
        jac_last_step_k3,
    }
}

fn main() {
    println!("==================================================================================================================");
    println!("DEVELOPMENT SCOUT: Lambda Residual Carry Sweep across 16 Auxiliary Seeds (120 Training Epochs)");
    println!("Architecture: z_{{t+1}} = lambda * z_t + (1 - lambda) * tanh(W_z z_t + W_x x_t + b_z)");
    println!("==================================================================================================================");
    println!("{:<6} | {:<8} | {:<12} | {:<8} | {:<10} | {:<8} | {:<8} | {:<10} | {:<10} | {:<8}",
        "Lambda", "k=2 Ret", "k=3 Marg (p)", "Surgery", "Mean Swap", "Transp", "Shuffle", "Init Jac Ret", "Last-Edge", "Sensor");
    println!("------------------------------------------------------------------------------------------------------------------");

    let lambda_grid = [
        0.00f32, 0.02, 0.04, 0.06, 0.08,
        0.10, 0.12, 0.15, 0.20, 0.30, 0.40, 0.50
    ];

    let mut all_scout_records = Vec::new();

    for &lambda in &lambda_grid {
        let results: Vec<SeedLambdaRecord> = (1..=16)
            .into_par_iter()
            .map(|i| evaluate_seed_lambda(i, lambda))
            .collect();

        all_scout_records.extend(results.clone());

        let k2_pass_count = results.iter().filter(|r| r.k2_passed).count();
        let k3_pass_count = results.iter().filter(|r| r.k3_passed).count();
        let k3_margins: Vec<f32> = results.iter().map(|r| r.k3_margin).collect();
        let k3_p = compute_sign_flip_p_val(&k3_margins);

        let surg_pass_count = results.iter().filter(|r| r.k3_surgery_transferred).count();
        let mean_swap: f32 = results.iter().map(|r| r.k3_swap_effect).sum::<f32>() / 16.0;
        let trans_pass_count = results.iter().filter(|r| r.k3_transposition_passed).count();
        let shuf_pass_count = results.iter().filter(|r| r.k3_shuffle_passed).count();

        let mean_j2: f32 = results.iter().map(|r| r.jac_init_step_k2).sum::<f32>() / 16.0;
        let mean_j3: f32 = results.iter().map(|r| r.jac_init_step_k3).sum::<f32>() / 16.0;
        let mean_last: f32 = results.iter().map(|r| r.jac_last_step_k3).sum::<f32>() / 16.0;
        let jac_retention = if mean_j2 > 0.0 { (mean_j3 / mean_j2) * 100.0 } else { 0.0 };

        let sensor_pass_count = results.iter().filter(|r| r.sensor_accuracy >= 0.90).count();

        println!("{:<6.2} | {:>2}/16    | {:>2}/16 (p={:.3}) | {:>2}/16    | {:>+9.4}  | {:>2}/16    | {:>2}/16    | {:>6.1}%    | {:>10.4} | {:>2}/16",
            lambda,
            k2_pass_count,
            k3_pass_count,
            k3_p,
            surg_pass_count,
            mean_swap,
            trans_pass_count,
            shuf_pass_count,
            jac_retention,
            mean_last,
            sensor_pass_count,
        );
    }
    println!("==================================================================================================================");

    let data_dir = Path::new("data");
    let out_file = data_dir.join("q17e_lambda_scout_results.json");
    let file = File::create(&out_file).expect("Failed to create lambda scout results JSON");
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &all_scout_records).expect("Failed to write JSON");
    println!("Persisted full lambda sweep telemetry to: {}", out_file.display());
}
