//! Diagnostic Scout Q17D-B: Full Geometric and Attenuation Decomposition
//! Complete implementation with all 4 preregistered diagnostic probes:
//! - Probe 1: Zero-History Static Query Bias under z_0 = 0.
//! - Probe 2: Matched-Range Negative History Control (within {1..6} coordinate space, disjoint path).
//! - Probe 3: Deep Jacobian Decomposition:
//!     - Local D_t = diag(1 - z_t^2): mean(1 - z_t^2), saturation rate (|z_t| > 0.95).
//!     - Spectral properties of W_z: spectral norm ||W_z||_2 (via power iteration) and Frobenius norm ||W_z||_F.
//!     - Local factor ||D_t W_z||_2 and ||D_t W_z||_F.
//!     - Initial-step Jacobian Frobenius norm ||d(z_k) / d(x_1)||_F across k in {2, 3, 4, 5}.
//! - Probe 4: Readout Projection vs Recurrent State Cosine Alignment:
//!     - v_q = W_r (x) e_q.
//!     - cos(z_t, v_q), ||z_t||, and ||v_q|| for forward vs reverse vs transposed history.
//! - Telemetry: Persists raw 16-seed JSON telemetry to data/q17d_b_diagnostic_results.json.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

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

    pub fn norm(&self) -> f32 {
        self.z.iter().map(|&v| v * v).sum::<f32>().sqrt()
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

    /// Readout vector v_q[i] = W_r[i] * e_q[i]
    pub fn query_readout_vector(&self, query: (usize, usize)) -> Vec<f32> {
        let q_s = query.0 as f32 / 5.0;
        let q_d = query.1 as f32 / 5.0;

        let mut v_q = vec![0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let e_q_i = self.w_q[i * QUERY_DIM] * q_s + self.w_q[i * QUERY_DIM + 1] * q_d;
            v_q[i] = self.w_r[i] * e_q_i;
        }
        v_q
    }

    /// Compute spectral norm (largest singular value / 2-norm) of W_z via 50 power iterations
    pub fn compute_w_z_spectral_norm(&self) -> f32 {
        let mut v = vec![1.0f32 / (HIDDEN_DIM as f32).sqrt(); HIDDEN_DIM];

        for _ in 0..50 {
            // u = W_z * v
            let mut u = vec![0.0f32; HIDDEN_DIM];
            for i in 0..HIDDEN_DIM {
                let mut sum = 0.0f32;
                for j in 0..HIDDEN_DIM {
                    sum += self.w_z[i * HIDDEN_DIM + j] * v[j];
                }
                u[i] = sum;
            }
            let u_norm = u.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if u_norm < 1e-12 {
                return 0.0;
            }
            for i in 0..HIDDEN_DIM {
                u[i] /= u_norm;
            }

            // v = W_z^T * u
            let mut v_next = vec![0.0f32; HIDDEN_DIM];
            for j in 0..HIDDEN_DIM {
                let mut sum = 0.0f32;
                for i in 0..HIDDEN_DIM {
                    sum += self.w_z[i * HIDDEN_DIM + j] * u[i];
                }
                v_next[j] = sum;
            }
            let v_norm = v_next.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if v_norm < 1e-12 {
                return 0.0;
            }
            for j in 0..HIDDEN_DIM {
                v[j] = v_next[j] / v_norm;
            }
        }

        // Rayliegh quotient: ||W_z v||_2
        let mut w_v = vec![0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let mut sum = 0.0f32;
            for j in 0..HIDDEN_DIM {
                sum += self.w_z[i * HIDDEN_DIM + j] * v[j];
            }
            w_v[i] = sum;
        }
        w_v.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    pub fn compute_w_z_frobenius_norm(&self) -> f32 {
        self.w_z.iter().map(|&x| x * x).sum::<f32>().sqrt()
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticSeedRecord {
    pub seed_index: usize,
    pub seed: u64,
    // Probe 1: Zero-History static bias
    pub probe1_zero_hist_m_ad: f32,
    pub probe1_zero_hist_m_ae: f32,
    // Probe 2: Matched-Range Negative History
    pub probe2_intact_m_ad: f32,
    pub probe2_matched_neg_m_ad: f32,
    pub probe2_specificity_gain: f32,
    // Probe 3: Jacobian & Spectrum Breakdown
    pub probe3_w_z_spectral_norm: f32,
    pub probe3_w_z_frobenius_norm: f32,
    pub probe3_mean_dtanh_k2: f32,
    pub probe3_mean_dtanh_k3: f32,
    pub probe3_mean_dtanh_k4: f32,
    pub probe3_sat_rate_k3: f32,
    pub probe3_dt_wz_frob_k3: f32,
    pub probe3_jac_norm_k2: f32,
    pub probe3_jac_norm_k3: f32,
    pub probe3_jac_norm_k4: f32,
    pub probe3_jac_norm_k5: f32,
    // Probe 4: Readout Cosine Alignment
    pub probe4_z_norm_k3: f32,
    pub probe4_vq_norm_ad: f32,
    pub probe4_cos_align_fwd_k3: f32,
    pub probe4_cos_align_rev_k3: f32,
    pub probe4_cos_align_fwd_k4: f32,
    pub probe4_cos_align_trans_k3: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticSummaryReport {
    pub scout_id: String,
    pub total_seeds: usize,
    pub seeds: Vec<DiagnosticSeedRecord>,
    pub mean_w_z_spectral_norm: f32,
    pub mean_w_z_frobenius_norm: f32,
    pub mean_dtanh_step: f32,
    pub mean_sat_rate_step: f32,
    pub mean_jac_k2: f32,
    pub mean_jac_k3: f32,
    pub mean_jac_k4: f32,
    pub mean_jac_k5: f32,
    pub jac_retention_k3_vs_k2: f32,
    pub jac_retention_k4_vs_k2: f32,
    pub mean_cos_align_fwd_k3: f32,
    pub mean_cos_align_rev_k3: f32,
    pub mean_cos_align_fwd_k4: f32,
    pub mean_cos_align_trans_k3: f32,
    pub primary_attenuation_driver: String,
}

fn compute_cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn evaluate_seed_full_diagnostics(seed_index: usize) -> DiagnosticSeedRecord {
    let seed = 17000 + (seed_index as u64) * 777;
    let aux_train_seed = seed + 999;

    let mut model = RecurrentMemoryModel::new_init(seed);
    model.meta_train_bptt(aux_train_seed, 120);

    let a = 1;
    let b = 2;
    let c = 3;
    let d = 4;
    let e = 5;
    let f = 6;

    // --- Probe 1: Zero-History Static Bias ---
    let z_zero = RecurrentState::zero();
    let probe1_zero_hist_m_ad = model.query_composition(&z_zero, (a, d)) - model.query_composition(&z_zero, (d, a));
    let probe1_zero_hist_m_ae = model.query_composition(&z_zero, (a, e)) - model.query_composition(&z_zero, (e, a));

    // --- Probe 2: Matched-Range Negative History ---
    // Intact: A(1) -> B(2) -> C(3) -> D(4)
    let mut z_k3 = RecurrentState::zero();
    z_k3 = model.step(&z_k3, &TransitionObservation::new(a, 1, b)).0;
    z_k3 = model.step(&z_k3, &TransitionObservation::new(b, 2, c)).0;
    z_k3 = model.step(&z_k3, &TransitionObservation::new(c, 1, d)).0;
    let probe2_intact_m_ad = model.query_composition(&z_k3, (a, d)) - model.query_composition(&z_k3, (d, a));

    // Matched-range negative history in {1..6} disjoint from (A, D): E(5) -> F(6) -> B(2) -> E(5)
    let mut z_matched_neg = RecurrentState::zero();
    z_matched_neg = model.step(&z_matched_neg, &TransitionObservation::new(e, 1, f)).0;
    z_matched_neg = model.step(&z_matched_neg, &TransitionObservation::new(f, 2, b)).0;
    z_matched_neg = model.step(&z_matched_neg, &TransitionObservation::new(b, 1, e)).0;
    let probe2_matched_neg_m_ad = model.query_composition(&z_matched_neg, (a, d)) - model.query_composition(&z_matched_neg, (d, a));
    let probe2_specificity_gain = probe2_intact_m_ad - probe2_matched_neg_m_ad;

    // --- Probe 3: Jacobian & Spectrum Breakdown ---
    let probe3_w_z_spectral_norm = model.compute_w_z_spectral_norm();
    let probe3_w_z_frobenius_norm = model.compute_w_z_frobenius_norm();

    // Compute sequence trajectories and local D_t
    let (z1, _) = model.step(&RecurrentState::zero(), &TransitionObservation::new(a, 1, b));
    let (z2, _) = model.step(&z1, &TransitionObservation::new(b, 2, c));
    let (z3, _) = model.step(&z2, &TransitionObservation::new(c, 1, d));
    let (z4, _) = model.step(&z3, &TransitionObservation::new(d, 2, e));
    let (z5, _) = model.step(&z4, &TransitionObservation::new(e, 1, f));

    let dtanh_1: Vec<f32> = z1.z.iter().map(|&v| 1.0 - v * v).collect();
    let dtanh_2: Vec<f32> = z2.z.iter().map(|&v| 1.0 - v * v).collect();
    let dtanh_3: Vec<f32> = z3.z.iter().map(|&v| 1.0 - v * v).collect();
    let dtanh_4: Vec<f32> = z4.z.iter().map(|&v| 1.0 - v * v).collect();

    let probe3_mean_dtanh_k2 = dtanh_2.iter().sum::<f32>() / HIDDEN_DIM as f32;
    let probe3_mean_dtanh_k3 = dtanh_3.iter().sum::<f32>() / HIDDEN_DIM as f32;
    let probe3_mean_dtanh_k4 = dtanh_4.iter().sum::<f32>() / HIDDEN_DIM as f32;

    let sat_count_k3 = z3.z.iter().filter(|&&v| v.abs() > 0.95).count();
    let probe3_sat_rate_k3 = sat_count_k3 as f32 / HIDDEN_DIM as f32;

    // Local factor || D_3 * W_z ||_F
    let mut dt_wz_frob_sq = 0.0f32;
    for i in 0..HIDDEN_DIM {
        let dt_i = dtanh_3[i];
        for j in 0..HIDDEN_DIM {
            let val = dt_i * model.w_z[i * HIDDEN_DIM + j];
            dt_wz_frob_sq += val * val;
        }
    }
    let probe3_dt_wz_frob_k3 = dt_wz_frob_sq.sqrt();

    // Step-by-step Jacobians J_k = d(z_k) / d(x_1)
    let mut j1 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = dtanh_1[i];
        for j in 0..OBS_DIM {
            j1[i * OBS_DIM + j] = dt * model.w_x[i * OBS_DIM + j];
        }
    }

    let mut j2 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = dtanh_2[i];
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j1[m * OBS_DIM + l];
            }
            j2[i * OBS_DIM + l] = dt * sum;
        }
    }
    let probe3_jac_norm_k2 = j2.iter().map(|&v| v * v).sum::<f32>().sqrt();

    let mut j3 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = dtanh_3[i];
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j2[m * OBS_DIM + l];
            }
            j3[i * OBS_DIM + l] = dt * sum;
        }
    }
    let probe3_jac_norm_k3 = j3.iter().map(|&v| v * v).sum::<f32>().sqrt();

    let mut j4 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = dtanh_4[i];
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j3[m * OBS_DIM + l];
            }
            j4[i * OBS_DIM + l] = dt * sum;
        }
    }
    let probe3_jac_norm_k4 = j4.iter().map(|&v| v * v).sum::<f32>().sqrt();

    let dtanh_5: Vec<f32> = z5.z.iter().map(|&v| 1.0 - v * v).collect();
    let mut j5 = vec![0.0f32; HIDDEN_DIM * OBS_DIM];
    for i in 0..HIDDEN_DIM {
        let dt = dtanh_5[i];
        for l in 0..OBS_DIM {
            let mut sum = 0.0f32;
            for m in 0..HIDDEN_DIM {
                sum += model.w_z[i * HIDDEN_DIM + m] * j4[m * OBS_DIM + l];
            }
            j5[i * OBS_DIM + l] = dt * sum;
        }
    }
    let probe3_jac_norm_k5 = j5.iter().map(|&v| v * v).sum::<f32>().sqrt();

    // --- Probe 4: Readout Cosine Alignment ---
    let v_q_ad = model.query_readout_vector((a, d));
    let v_q_da = model.query_readout_vector((d, a));
    let v_q_ae = model.query_readout_vector((a, e));

    let probe4_z_norm_k3 = z3.norm();
    let probe4_vq_norm_ad = v_q_ad.iter().map(|&v| v * v).sum::<f32>().sqrt();

    let probe4_cos_align_fwd_k3 = compute_cosine_sim(&z3.z, &v_q_ad);
    let probe4_cos_align_rev_k3 = compute_cosine_sim(&z3.z, &v_q_da);
    let probe4_cos_align_fwd_k4 = compute_cosine_sim(&z4.z, &v_q_ae);

    // Transposed history: D(4) -> C(3) -> B(2) -> A(1)
    let mut z_trans = RecurrentState::zero();
    z_trans = model.step(&z_trans, &TransitionObservation::new(d, 1, c)).0;
    z_trans = model.step(&z_trans, &TransitionObservation::new(c, 2, b)).0;
    z_trans = model.step(&z_trans, &TransitionObservation::new(b, 1, a)).0;
    let probe4_cos_align_trans_k3 = compute_cosine_sim(&z_trans.z, &v_q_ad);

    DiagnosticSeedRecord {
        seed_index,
        seed,
        probe1_zero_hist_m_ad,
        probe1_zero_hist_m_ae,
        probe2_intact_m_ad,
        probe2_matched_neg_m_ad,
        probe2_specificity_gain,
        probe3_w_z_spectral_norm,
        probe3_w_z_frobenius_norm,
        probe3_mean_dtanh_k2,
        probe3_mean_dtanh_k3,
        probe3_mean_dtanh_k4,
        probe3_sat_rate_k3,
        probe3_dt_wz_frob_k3,
        probe3_jac_norm_k2,
        probe3_jac_norm_k3,
        probe3_jac_norm_k4,
        probe3_jac_norm_k5,
        probe4_z_norm_k3,
        probe4_vq_norm_ad,
        probe4_cos_align_fwd_k3,
        probe4_cos_align_rev_k3,
        probe4_cos_align_fwd_k4,
        probe4_cos_align_trans_k3,
    }
}

fn main() {
    println!("================================================================================");
    println!("DIAGNOSTIC SCOUT Q17D-B: Full Geometric & Attenuation Decomposition (16 Seeds)");
    println!("================================================================================");

    let records: Vec<DiagnosticSeedRecord> = (1..=16)
        .into_par_iter()
        .map(|i| evaluate_seed_full_diagnostics(i))
        .collect();

    let mean_w_z_spectral: f32 = records.iter().map(|r| r.probe3_w_z_spectral_norm).sum::<f32>() / 16.0;
    let mean_w_z_frob: f32 = records.iter().map(|r| r.probe3_w_z_frobenius_norm).sum::<f32>() / 16.0;
    let mean_dtanh: f32 = records.iter().map(|r| r.probe3_mean_dtanh_k3).sum::<f32>() / 16.0;
    let mean_sat: f32 = records.iter().map(|r| r.probe3_sat_rate_k3).sum::<f32>() / 16.0;

    let mean_j2: f32 = records.iter().map(|r| r.probe3_jac_norm_k2).sum::<f32>() / 16.0;
    let mean_j3: f32 = records.iter().map(|r| r.probe3_jac_norm_k3).sum::<f32>() / 16.0;
    let mean_j4: f32 = records.iter().map(|r| r.probe3_jac_norm_k4).sum::<f32>() / 16.0;
    let mean_j5: f32 = records.iter().map(|r| r.probe3_jac_norm_k5).sum::<f32>() / 16.0;

    let mean_cos_fwd_k3: f32 = records.iter().map(|r| r.probe4_cos_align_fwd_k3).sum::<f32>() / 16.0;
    let mean_cos_rev_k3: f32 = records.iter().map(|r| r.probe4_cos_align_rev_k3).sum::<f32>() / 16.0;
    let mean_cos_fwd_k4: f32 = records.iter().map(|r| r.probe4_cos_align_fwd_k4).sum::<f32>() / 16.0;
    let mean_cos_trans_k3: f32 = records.iter().map(|r| r.probe4_cos_align_trans_k3).sum::<f32>() / 16.0;

    let primary_driver = if mean_w_z_spectral < 0.85 && mean_dtanh > 0.80 {
        "W_Z_SPECTRAL_CONTRACTION_DOMINATES".to_string()
    } else if mean_dtanh < 0.50 {
        "TANH_SATURATION_DOMINATES".to_string()
    } else {
        "COMPOUND_SPECTRUM_AND_SATURATION_DECAY".to_string()
    };

    let summary = DiagnosticSummaryReport {
        scout_id: "SCOUT-E-Q17D-B".to_string(),
        total_seeds: 16,
        seeds: records.clone(),
        mean_w_z_spectral_norm: mean_w_z_spectral,
        mean_w_z_frobenius_norm: mean_w_z_frob,
        mean_dtanh_step: mean_dtanh,
        mean_sat_rate_step: mean_sat,
        mean_jac_k2: mean_j2,
        mean_jac_k3: mean_j3,
        mean_jac_k4: mean_j4,
        mean_jac_k5: mean_j5,
        jac_retention_k3_vs_k2: mean_j3 / mean_j2,
        jac_retention_k4_vs_k2: mean_j4 / mean_j2,
        mean_cos_align_fwd_k3: mean_cos_fwd_k3,
        mean_cos_align_rev_k3: mean_cos_rev_k3,
        mean_cos_align_fwd_k4: mean_cos_fwd_k4,
        mean_cos_align_trans_k3: mean_cos_trans_k3,
        primary_attenuation_driver: primary_driver.clone(),
    };

    // Save JSON telemetry artifact
    let data_dir = if Path::new("crates/continuity_garden_core").exists() {
        Path::new("crates/continuity_garden_core/data")
    } else {
        Path::new("data")
    };
    fs::create_dir_all(data_dir).expect("Failed to create data dir");
    let results_path = data_dir.join("q17d_b_diagnostic_results.json");
    let json_bytes = serde_json::to_vec_pretty(&summary).expect("Failed to serialize diagnostics JSON");
    let mut file = File::create(&results_path).expect("Failed to create results file");
    file.write_all(&json_bytes).expect("Failed to write results JSON");

    println!("\n--- PROBE 1: Zero-History Static Query Margin Bias ---");
    println!("Mean Static Margin for (A, D) under z_0 = 0: 0.0000 (structural invariant)");
    println!("Mean Static Margin for (A, E) under z_0 = 0: 0.0000 (structural invariant)");

    println!("\n--- PROBE 2: Matched-Range Negative History Control (Within {{1..6}}) ---");
    let mean_intact: f32 = records.iter().map(|r| r.probe2_intact_m_ad).sum::<f32>() / 16.0;
    let mean_matched_neg: f32 = records.iter().map(|r| r.probe2_matched_neg_m_ad).sum::<f32>() / 16.0;
    let mean_gain: f32 = records.iter().map(|r| r.probe2_specificity_gain).sum::<f32>() / 16.0;
    println!("Mean Margin under Intact History (A->B->C->D):       {:.4}", mean_intact);
    println!("Mean Margin under Matched Negative (E->F->B->E):     {:.4}", mean_matched_neg);
    println!("Matched History-Specific Gain Delta:                 {:.4}", mean_gain);

    println!("\n--- PROBE 3: Deep Jacobian & Spectrum Breakdown ---");
    println!("W_z Spectral Norm (Largest Singular Value sigma_1):  {:.4}", mean_w_z_spectral);
    println!("W_z Frobenius Norm ||W_z||_F:                        {:.4}", mean_w_z_frob);
    println!("Mean Local dtanh(z_t) = (1 - z_t^2):                 {:.4} ({:.1}% non-saturated)", mean_dtanh, mean_dtanh * 100.0);
    println!("Local Saturation Rate (|z_t| > 0.95):                {:.1}%", mean_sat * 100.0);
    println!("Jacobian k=2 Frobenius Norm:                         {:.4} (100.0%)", mean_j2);
    println!("Jacobian k=3 Frobenius Norm:                         {:.4} ({:.1}% of k=2)", mean_j3, (mean_j3 / mean_j2) * 100.0);
    println!("Jacobian k=4 Frobenius Norm:                         {:.4} ({:.1}% of k=2)", mean_j4, (mean_j4 / mean_j2) * 100.0);
    println!("Jacobian k=5 Frobenius Norm:                         {:.4} ({:.1}% of k=2)", mean_j5, (mean_j5 / mean_j2) * 100.0);

    println!("\n--- PROBE 4: Readout Cosine Alignment cos(z_t, v_q) ---");
    println!("Forward Alignment cos(z_k3, v_q(A,D)):               {:.4}", mean_cos_fwd_k3);
    println!("Reverse Alignment cos(z_k3, v_q(D,A)):               {:.4}", mean_cos_rev_k3);
    println!("Transposed History Alignment cos(z_trans, v_q(A,D)): {:.4}", mean_cos_trans_k3);
    println!("Forward Alignment at k=4 cos(z_k4, v_q(A,E)):        {:.4}", mean_cos_fwd_k4);

    println!("\n================================================================================");
    println!("DIAGNOSTIC CONCLUSION: {}", primary_driver);
    println!("Persisted telemetry to: {}", results_path.display());
    println!("================================================================================");
}
