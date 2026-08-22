//! Diagnostic Scout Q17D-B: Decomposition of Multi-Hop Extrapolation Dissociation
//! Investigates:
//! 1. Zero-History / Static Query Bias: r_theta(0, (u, v)) vs r_theta(0, (v, u))
//! 2. Unrelated-History Control: Feeding random unrelated transitions before querying (A, D)
//! 3. Initial-Step Jacobian Sensitivity: || d(z_k) / d(x_1) ||_F across k in {2, 3, 4, 5}
//! 4. Readout Cosine Alignment: cos(z_t, W_r (x) e_q) across depths

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const HIDDEN_DIM: usize = 128;
pub const OBS_DIM: usize = 4;
pub const QUERY_DIM: usize = 2;

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
pub struct RecurrentMemoryModel {
    pub w_z: Vec<f32>,
    pub w_x: Vec<f32>,
    pub b_z: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_r: Vec<f32>,
    pub b_r: f32,
}

impl RecurrentMemoryModel {
    pub fn new_init(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5A5A55A5A5A5A);
        let scale_z = (2.0f32 / (HIDDEN_DIM + HIDDEN_DIM) as f32).sqrt();
        let scale_x = (2.0f32 / (HIDDEN_DIM + OBS_DIM) as f32).sqrt();
        let scale_r = (2.0f32 / (HIDDEN_DIM + QUERY_DIM) as f32).sqrt();

        let mut w_z = vec![0.0f32; HIDDEN_DIM * HIDDEN_DIM];
        let mut w_x = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
        let mut b_z = vec![0.0f32; HIDDEN_DIM];
        let mut w_q = vec![0.0f32; HIDDEN_DIM * QUERY_DIM];
        let mut w_r = vec![0.0f32; HIDDEN_DIM];

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
        }

        Self {
            w_z,
            w_x,
            b_z,
            w_q,
            w_r,
            b_r: 0.0,
        }
    }

    #[inline(always)]
    pub fn step(&self, state: &RecurrentState, obs: &TransitionObservation) -> (RecurrentState, Vec<f32>) {
        let x = obs.to_vec();
        let mut next_z = vec![0.0f32; HIDDEN_DIM];
        let mut pre_act = vec![0.0f32; HIDDEN_DIM];

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
            pre_act[i] = sum;
            next_z[i] = sum.tanh();
        }

        (RecurrentState { z: next_z }, pre_act)
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

    pub fn meta_train_bptt(&mut self, rng_seed: u64, epochs: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(rng_seed);
        let lr = 0.030f32;

        for _ in 0..epochs {
            for _ in 0..64 {
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

                let mut a1 = vec![0.0f32; HIDDEN_DIM];
                let mut z1 = vec![0.0f32; HIDDEN_DIM];
                for i in 0..HIDDEN_DIM {
                    let mut sum = self.b_z[i];
                    for j in 0..OBS_DIM {
                        sum += self.w_x[i * OBS_DIM + j] * x1[j];
                    }
                    a1[i] = sum;
                    z1[i] = sum.tanh();
                }

                let mut a2 = vec![0.0f32; HIDDEN_DIM];
                let mut z2 = vec![0.0f32; HIDDEN_DIM];
                for i in 0..HIDDEN_DIM {
                    let mut sum = self.b_z[i];
                    for j in 0..HIDDEN_DIM {
                        sum += self.w_z[i * HIDDEN_DIM + j] * z1[j];
                    }
                    for j in 0..OBS_DIM {
                        sum += self.w_x[i * OBS_DIM + j] * x2[j];
                    }
                    a2[i] = sum;
                    z2[i] = sum.tanh();
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

                let mut d_a2 = vec![0.0f32; HIDDEN_DIM];
                for i in 0..HIDDEN_DIM {
                    d_a2[i] = grad_z2[i] * (1.0 - z2[i] * z2[i]);
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
                    grad_z1[j] = sum;
                }
                for i in 0..HIDDEN_DIM {
                    for j in 0..HIDDEN_DIM {
                        self.w_z[i * HIDDEN_DIM + j] -= lr * d_a2[i] * z1[j];
                    }
                }

                for i in 0..HIDDEN_DIM {
                    let d_a1_i = grad_z1[i] * (1.0 - z1[i] * z1[i]);
                    self.b_z[i] -= lr * d_a1_i;
                    for j in 0..OBS_DIM {
                        self.w_x[i * OBS_DIM + j] -= lr * d_a1_i * x1[j];
                    }
                }
            }
        }
    }
}

pub struct DiagnosticResults {
    pub seed_index: usize,
    // Probe 1: Zero-History static bias
    pub zero_hist_m_ad: f32,
    pub zero_hist_m_ae: f32,
    // Probe 2: Unrelated-History specificity
    pub intact_m_ad: f32,
    pub unrelated_m_ad: f32,
    pub specificity_delta: f32,
    // Probe 3: Jacobian Frobenius norms
    pub jac_norm_k2: f32,
    pub jac_norm_k3: f32,
    pub jac_norm_k4: f32,
    pub jac_norm_k5: f32,
}

fn compute_jacobian_norm(model: &RecurrentMemoryModel, obs_seq: &[TransitionObservation]) -> f32 {
    let k = obs_seq.len();
    if k == 0 {
        return 0.0;
    }

    // Forward pass storing states and pre-activations
    let mut states = Vec::with_capacity(k + 1);
    let mut z = RecurrentState::zero();
    states.push(z.clone());

    let mut dtanh_list = Vec::with_capacity(k);
    for obs in obs_seq {
        let (next_z, _) = model.step(&z, obs);
        let dtanh: Vec<f32> = next_z.z.iter().map(|&v| 1.0 - v * v).collect();
        dtanh_list.push(dtanh);
        z = next_z;
        states.push(z.clone());
    }

    // Compute J_1 = diag(1 - z_1^2) * W_x  (HIDDEN_DIM x OBS_DIM)
    let mut j_curr = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = dtanh_list[0][i];
        for j in 0..OBS_DIM {
            j_curr[i * OBS_DIM + j] = dt * model.w_x[i * OBS_DIM + j];
        }
    }

    // Step through t=2..k: J_t = diag(1 - z_t^2) * W_z * J_{t-1}
    for t in 1..k {
        let dt = &dtanh_list[t];
        let mut j_next = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
        for i in 0..HIDDEN_DIM {
            let dt_i = dt[i];
            for l in 0..OBS_DIM {
                let mut sum = 0.0f32;
                for m in 0..HIDDEN_DIM {
                    sum += model.w_z[i * HIDDEN_DIM + m] * j_curr[m * OBS_DIM + l];
                }
                j_next[i * OBS_DIM + l] = dt_i * sum;
            }
        }
        j_curr = j_next;
    }

    let frob_sq: f32 = j_curr.iter().map(|&v| v * v).sum();
    frob_sq.sqrt()
}

fn evaluate_seed_diagnostics(seed_index: usize) -> DiagnosticResults {
    let seed = 17000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;

    let mut model = RecurrentMemoryModel::new_init(seed);
    model.meta_train_bptt(aux_train_seed, 250);

    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;
    let e = 5;
    let f = 6;

    // Probe 1: Zero-History static bias
    let z_zero = RecurrentState::zero();
    let zero_hist_m_ad = model.query_composition(&z_zero, (a, d)) - model.query_composition(&z_zero, (d, a));
    let zero_hist_m_ae = model.query_composition(&z_zero, (a, e)) - model.query_composition(&z_zero, (e, a));

    // Probe 2: Intact vs Unrelated History
    let mut z_k3 = RecurrentState::zero();
    z_k3 = model.step(&z_k3, &TransitionObservation::new(a, 1, b)).0;
    z_k3 = model.step(&z_k3, &TransitionObservation::new(b, 2, c)).0;
    z_k3 = model.step(&z_k3, &TransitionObservation::new(c, 1, d)).0;
    let intact_m_ad = model.query_composition(&z_k3, (a, d)) - model.query_composition(&z_k3, (d, a));

    // Unrelated history: X(8) -> Y(9) -> Z(10) -> W(11)
    let mut z_unrelated = RecurrentState::zero();
    z_unrelated = model.step(&z_unrelated, &TransitionObservation::new(8, 1, 9)).0;
    z_unrelated = model.step(&z_unrelated, &TransitionObservation::new(9, 2, 10)).0;
    z_unrelated = model.step(&z_unrelated, &TransitionObservation::new(10, 1, 11)).0;
    let unrelated_m_ad = model.query_composition(&z_unrelated, (a, d)) - model.query_composition(&z_unrelated, (d, a));
    let specificity_delta = intact_m_ad - unrelated_m_ad;

    // Probe 3: Jacobian Sensitivity Norms
    let seq_k2 = vec![
        TransitionObservation::new(a, 1, b),
        TransitionObservation::new(b, 2, c),
    ];
    let seq_k3 = vec![
        TransitionObservation::new(a, 1, b),
        TransitionObservation::new(b, 2, c),
        TransitionObservation::new(c, 1, d),
    ];
    let seq_k4 = vec![
        TransitionObservation::new(a, 1, b),
        TransitionObservation::new(b, 2, c),
        TransitionObservation::new(c, 1, d),
        TransitionObservation::new(d, 2, e),
    ];
    let seq_k5 = vec![
        TransitionObservation::new(a, 1, b),
        TransitionObservation::new(b, 2, c),
        TransitionObservation::new(c, 1, d),
        TransitionObservation::new(d, 2, e),
        TransitionObservation::new(e, 1, f),
    ];

    let jac_norm_k2 = compute_jacobian_norm(&model, &seq_k2);
    let jac_norm_k3 = compute_jacobian_norm(&model, &seq_k3);
    let jac_norm_k4 = compute_jacobian_norm(&model, &seq_k4);
    let jac_norm_k5 = compute_jacobian_norm(&model, &seq_k5);

    DiagnosticResults {
        seed_index,
        zero_hist_m_ad,
        zero_hist_m_ae,
        intact_m_ad,
        unrelated_m_ad,
        specificity_delta,
        jac_norm_k2,
        jac_norm_k3,
        jac_norm_k4,
        jac_norm_k5,
    }
}

fn main() {
    println!("================================================================================");
    println!("RUNNING DIAGNOSTIC SCOUT Q17D-B: Multi-Hop Causal Dissociation Diagnostics");
    println!("================================================================================");

    let results: Vec<DiagnosticResults> = (1..=16)
        .into_par_iter()
        .map(|i| evaluate_seed_diagnostics(i))
        .collect();

    println!("\n--- PROBE 1: Zero-History Static Query Margin Bias ---");
    let mean_zero_ad: f32 = results.iter().map(|r| r.zero_hist_m_ad).sum::<f32>() / 16.0;
    let mean_zero_ae: f32 = results.iter().map(|r| r.zero_hist_m_ae).sum::<f32>() / 16.0;
    println!("Mean Static Margin for (A, D) with ZERO history: {:.4}", mean_zero_ad);
    println!("Mean Static Margin for (A, E) with ZERO history: {:.4}", mean_zero_ae);

    println!("\n--- PROBE 2: History Specificity vs Unrelated Transitions ---");
    let mean_intact_ad: f32 = results.iter().map(|r| r.intact_m_ad).sum::<f32>() / 16.0;
    let mean_unrelated_ad: f32 = results.iter().map(|r| r.unrelated_m_ad).sum::<f32>() / 16.0;
    let mean_delta: f32 = results.iter().map(|r| r.specificity_delta).sum::<f32>() / 16.0;
    println!("Mean Margin under Intact History (A->B->C->D):   {:.4}", mean_intact_ad);
    println!("Mean Margin under Unrelated History (X->Y->Z->W): {:.4}", mean_unrelated_ad);
    println!("History-Specific Gain Delta:                     {:.4}", mean_delta);

    println!("\n--- PROBE 3: Initial-Step Jacobian Sensitivity Attenuation || d(z_k) / d(x_1) ||_F ---");
    let mean_j2: f32 = results.iter().map(|r| r.jac_norm_k2).sum::<f32>() / 16.0;
    let mean_j3: f32 = results.iter().map(|r| r.jac_norm_k3).sum::<f32>() / 16.0;
    let mean_j4: f32 = results.iter().map(|r| r.jac_norm_k4).sum::<f32>() / 16.0;
    let mean_j5: f32 = results.iter().map(|r| r.jac_norm_k5).sum::<f32>() / 16.0;
    println!("k=2 Frobenius Norm: {:.6} (100.0%)", mean_j2);
    println!("k=3 Frobenius Norm: {:.6} ({:.1}% of k=2)", mean_j3, (mean_j3 / mean_j2) * 100.0);
    println!("k=4 Frobenius Norm: {:.6} ({:.1}% of k=2)", mean_j4, (mean_j4 / mean_j2) * 100.0);
    println!("k=5 Frobenius Norm: {:.6} ({:.1}% of k=2)", mean_j5, (mean_j5 / mean_j2) * 100.0);
    println!("================================================================================");
}
