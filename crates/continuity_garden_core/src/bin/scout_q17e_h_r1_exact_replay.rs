//! SCOUT-E-Q17E-H-R1: Exact Replay of Scout-H Probes on Exact Scout-G Winning Organisms
//!
//! Interrogates the exact 16 Scout-G Condition 5 organisms (meta_train_bptt with lr=0.03, 64 batches/epoch,
//! 120 epochs, seed schedule 88000 + 777*i, u=1,v=2,w=3) to verify whether the unbound-source failure
//! (X -> D scoring >= intact C -> D) is present in the exact Scout-G models.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub const OBS_DIM: usize = 4;
pub const EDGE_DIM: usize = 32;
pub const REL_DIM: usize = 128;
pub const QUERY_DIM: usize = 2;
pub const N_SEEDS: usize = 16;
pub const AUX_SEEDS: [u64; 16] = [
    42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021, 2223, 2425, 2627, 2829, 3031, 3233,
];

pub const TRAIN_EPOCHS: usize = 120;
pub const TRAIN_BATCHES_PER_EPOCH: usize = 64;
pub const TRAIN_LR: f32 = 0.03;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionObservation {
    pub src: usize,
    pub action: usize,
    pub dst: usize,
    pub src_jitter: f32,
    pub dst_jitter: f32,
}

impl TransitionObservation {
    pub fn new(src: usize, action: usize, dst: usize) -> Self {
        Self {
            src,
            action,
            dst,
            src_jitter: 0.0,
            dst_jitter: 0.0,
        }
    }

    pub fn new_with_jitter(src: usize, action: usize, dst: usize, rng: &mut ChaCha8Rng) -> Self {
        Self {
            src,
            action,
            dst,
            src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
            dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
        }
    }

    #[inline(always)]
    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(OBS_DIM);
        v.push(self.src as f32 / 5.0 + self.src_jitter);
        v.push(self.action as f32 / 5.0);
        v.push(self.dst as f32 / 5.0 + self.dst_jitter);
        v.push(1.0);
        v
    }
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoutGWinningModel {
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
}

impl ScoutGWinningModel {
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
        let dt_e = vec![1.0f32; EDGE_DIM]; // Linear edge
        for i in 0..EDGE_DIM {
            let mut sum = self.b_e[i];
            for j in 0..OBS_DIM {
                sum += self.w_e[i * OBS_DIM + j] * x[j];
            }
            e[i] = sum;
        }
        (e, dt_e)
    }

    #[inline(always)]
    pub fn compose_relation(&self, m_prev: &[f32], e: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut m_next = vec![0.0f32; REL_DIM];
        let mut m_tilde = vec![0.0f32; REL_DIM];
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
            m_tilde[i] = val;
            dt_m[i] = 1.0 - val * val;
            m_next[i] = m_prev[i] + 1.0 * val; // Additive residual accumulator
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

                // Relational loss 2-step
                let s_pos2 = self.query_composition(&m2, q_pos2);
                let s_neg2 = self.query_composition(&m2, q_neg2);
                let margin2 = s_pos2 - s_neg2;
                let target2 = if is_causal { 1.0 } else { -1.0 };
                let err2 = sigmoid(margin2) - if target2 > 0.0 { 1.0 } else { 0.0 };

                // Prefix Relational loss 1-step
                let s_pos1 = self.query_composition(&m1, q_pos1);
                let s_neg1 = self.query_composition(&m1, q_neg1);
                let margin1 = s_pos1 - s_neg1;
                let target1 = if is_causal { 1.0 } else { -1.0 };
                let err1 = sigmoid(margin1) - if target1 > 0.0 { 1.0 } else { 0.0 };

                // Sensor query loss on m1
                let sensor_pred = sigmoid(self.b_sensor + obs1.src as f32 / 5.0);
                let sensor_err = sensor_pred - (obs1.src as f32 / 5.0);

                // Backward pass gradients
                let mut grad_m2 = vec![0.0f32; REL_DIM];
                let mut grad_w_r = vec![0.0f32; REL_DIM];
                let mut grad_w_q = vec![0.0f32; REL_DIM * QUERY_DIM];

                let q_pos2_vec = [q_pos2.0 as f32 / 5.0, q_pos2.1 as f32 / 5.0];
                let q_neg2_vec = [q_neg2.0 as f32 / 5.0, q_neg2.1 as f32 / 5.0];

                for i in 0..REL_DIM {
                    let eq_pos = self.w_q[i * QUERY_DIM] * q_pos2_vec[0] + self.w_q[i * QUERY_DIM + 1] * q_pos2_vec[1];
                    let eq_neg = self.w_q[i * QUERY_DIM] * q_neg2_vec[0] + self.w_q[i * QUERY_DIM + 1] * q_neg2_vec[1];
                    let diff_q = eq_pos - eq_neg;

                    grad_m2[i] += err2 * self.w_r[i] * diff_q;
                    grad_w_r[i] += err2 * m2[i] * diff_q;

                    grad_w_q[i * QUERY_DIM] += err2 * self.w_r[i] * m2[i] * (q_pos2_vec[0] - q_neg2_vec[0]);
                    grad_w_q[i * QUERY_DIM + 1] += err2 * self.w_r[i] * m2[i] * (q_pos2_vec[1] - q_neg2_vec[1]);
                }

                // Step 2 compose backward
                let mut grad_m1 = vec![0.0f32; REL_DIM];
                let mut grad_e2 = vec![0.0f32; EDGE_DIM];
                let mut grad_w_m = vec![0.0f32; REL_DIM * REL_DIM];
                let mut grad_w_c = vec![0.0f32; REL_DIM * EDGE_DIM];
                let mut grad_b_m = vec![0.0f32; REL_DIM];

                for i in 0..REL_DIM {
                    let d_tilde = grad_m2[i] * 1.0 * dt_m2[i]; // eta = 1.0
                    grad_m1[i] += grad_m2[i]; // Identity skip path
                    grad_b_m[i] += d_tilde;

                    for j in 0..REL_DIM {
                        grad_w_m[i * REL_DIM + j] += d_tilde * m1[j];
                        grad_m1[j] += d_tilde * self.w_m[i * REL_DIM + j];
                    }
                    for j in 0..EDGE_DIM {
                        grad_w_c[i * EDGE_DIM + j] += d_tilde * e2[j];
                        grad_e2[j] += d_tilde * self.w_c[i * EDGE_DIM + j];
                    }
                }

                // Prefix m1 loss contribution
                let q_pos1_vec = [q_pos1.0 as f32 / 5.0, q_pos1.1 as f32 / 5.0];
                let q_neg1_vec = [q_neg1.0 as f32 / 5.0, q_neg1.1 as f32 / 5.0];

                for i in 0..REL_DIM {
                    let eq_pos = self.w_q[i * QUERY_DIM] * q_pos1_vec[0] + self.w_q[i * QUERY_DIM + 1] * q_pos1_vec[1];
                    let eq_neg = self.w_q[i * QUERY_DIM] * q_neg1_vec[0] + self.w_q[i * QUERY_DIM + 1] * q_neg1_vec[1];
                    let diff_q = eq_pos - eq_neg;

                    grad_m1[i] += err1 * self.w_r[i] * diff_q;
                    grad_w_r[i] += err1 * m1[i] * diff_q;

                    grad_w_q[i * QUERY_DIM] += err1 * self.w_r[i] * m1[i] * (q_pos1_vec[0] - q_neg1_vec[0]);
                    grad_w_q[i * QUERY_DIM + 1] += err1 * self.w_r[i] * m1[i] * (q_pos1_vec[1] - q_neg1_vec[1]);
                }

                // Step 1 compose backward
                let mut grad_e1 = vec![0.0f32; EDGE_DIM];
                for i in 0..REL_DIM {
                    let d_tilde = grad_m1[i] * 1.0 * dt_m1[i];
                    grad_b_m[i] += d_tilde;

                    for j in 0..EDGE_DIM {
                        grad_w_c[i * EDGE_DIM + j] += d_tilde * e1[j];
                        grad_e1[j] += d_tilde * self.w_c[i * EDGE_DIM + j];
                    }
                }

                // Linear edge encode backward
                let mut grad_w_e = vec![0.0f32; EDGE_DIM * OBS_DIM];
                let mut grad_b_e = vec![0.0f32; EDGE_DIM];
                let x1 = obs1.to_vec();
                let x2 = obs2.to_vec();

                for i in 0..EDGE_DIM {
                    let de1 = grad_e1[i] * dt_e1[i];
                    let de2 = grad_e2[i] * dt_e2[i];
                    grad_b_e[i] += de1 + de2;

                    for j in 0..OBS_DIM {
                        grad_w_e[i * OBS_DIM + j] += de1 * x1[j] + de2 * x2[j];
                    }
                }

                // Apply SGD update
                for i in 0..self.w_e.len() {
                    self.w_e[i] -= lr * grad_w_e[i];
                }
                for i in 0..self.b_e.len() {
                    self.b_e[i] -= lr * grad_b_e[i];
                }
                for i in 0..self.w_m.len() {
                    self.w_m[i] -= lr * grad_w_m[i];
                }
                for i in 0..self.w_c.len() {
                    self.w_c[i] -= lr * grad_w_c[i];
                }
                for i in 0..self.b_m.len() {
                    self.b_m[i] -= lr * grad_b_m[i];
                }
                for i in 0..self.w_q.len() {
                    self.w_q[i] -= lr * grad_w_q[i];
                }
                for i in 0..self.w_r.len() {
                    self.w_r[i] -= lr * grad_w_r[i];
                }
                self.b_r -= lr * (err2 + err1);
                self.b_sensor -= lr * sensor_err;
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactReplaySeedAssay {
    pub seed_index: usize,
    pub seed: u64,
    pub train_seed: u64,
    pub margin_pre_edge: f32,       // Q(1, 4) on m2 (before step 3)
    pub margin_intact: f32,         // C(m2, e_34)
    pub margin_zero_edge: f32,      // C(m2, 0)
    pub margin_wrong_dst: f32,      // C(m2, e_35)
    pub margin_wrong_src: f32,      // C(m2, e_54)
    pub margin_donor_hist: f32,     // C(m2_donor, e_34)
    pub zero_edge_drop: f32,        // intact - zero
    pub wrong_dst_drop: f32,        // intact - wrong_dst
    pub wrong_src_drop: f32,        // intact - wrong_src
    pub donor_hist_drop: f32,       // intact - donor_hist
    pub unbound_source_anomaly: bool, // wrong_src_drop <= 0 (i.e. X->D does not drop)
}

fn main() {
    println!("================================================================================");
    println!("SCOUT-E-Q17E-H-R1: Exact Replay of Scout-H Probes on Exact Scout-G Winning Organisms");
    println!("Evaluating the 16 exact Scout-G trained organisms (lr=0.03, 64 b/e, 88000 + 777i)");
    println!("================================================================================");

    let mut assays = Vec::new();

    for (i, &seed) in AUX_SEEDS.iter().enumerate() {
        let train_seed = 88000 + 777 * (i as u64);
        let mut model = ScoutGWinningModel::new_init(seed);
        model.meta_train_bptt(train_seed, TRAIN_EPOCHS);

        // Sequence: 1 -> 2 -> 3 -> 4
        // Step 1: 1 -> 2
        let obs1 = TransitionObservation::new(1, 1, 2);
        let (e1, _) = model.encode_edge(&obs1);
        let (m1, _, _) = model.compose_relation(&vec![0.0; REL_DIM], &e1);

        // Step 2: 2 -> 3
        let obs2 = TransitionObservation::new(2, 2, 3);
        let (e2, _) = model.encode_edge(&obs2);
        let (m2, _, _) = model.compose_relation(&m1, &e2);

        // Probe 1: Pre-edge probe on m2 for (1, 4)
        let s_pre_pos = model.query_composition(&m2, (1, 4));
        let s_pre_neg = model.query_composition(&m2, (4, 1));
        let margin_pre = s_pre_pos - s_pre_neg;

        // Step 3 Intact: 3 -> 4
        let obs3_intact = TransitionObservation::new(3, 3, 4);
        let (e3_intact, _) = model.encode_edge(&obs3_intact);
        let (m3_intact, _, _) = model.compose_relation(&m2, &e3_intact);
        let margin_intact = model.query_composition(&m3_intact, (1, 4)) - model.query_composition(&m3_intact, (4, 1));

        // Probe 3: Zero final edge
        let (m3_zero, _, _) = model.compose_relation(&m2, &vec![0.0; EDGE_DIM]);
        let margin_zero = model.query_composition(&m3_zero, (1, 4)) - model.query_composition(&m3_zero, (4, 1));

        // Probe 4: Wrong destination 3 -> 5
        let obs3_wrong_dst = TransitionObservation::new(3, 3, 5);
        let (e3_wrong_dst, _) = model.encode_edge(&obs3_wrong_dst);
        let (m3_wrong_dst, _, _) = model.compose_relation(&m2, &e3_wrong_dst);
        let margin_wrong_dst = model.query_composition(&m3_wrong_dst, (1, 4)) - model.query_composition(&m3_wrong_dst, (4, 1));

        // Probe 5: Wrong source 5 -> 4 (Source 5 != 3)
        let obs3_wrong_src = TransitionObservation::new(5, 3, 4);
        let (e3_wrong_src, _) = model.encode_edge(&obs3_wrong_src);
        let (m3_wrong_src, _, _) = model.compose_relation(&m2, &e3_wrong_src);
        let margin_wrong_src = model.query_composition(&m3_wrong_src, (1, 4)) - model.query_composition(&m3_wrong_src, (4, 1));

        // Probe 6: Donor history transplant (Donor: 4 -> 3 -> 2 with intact 3 -> 4)
        let obs_d1 = TransitionObservation::new(4, 2, 3);
        let (e_d1, _) = model.encode_edge(&obs_d1);
        let (m_d1, _, _) = model.compose_relation(&vec![0.0; REL_DIM], &e_d1);
        let obs_d2 = TransitionObservation::new(3, 1, 2);
        let (e_d2, _) = model.encode_edge(&obs_d2);
        let (m2_donor, _, _) = model.compose_relation(&m_d1, &e_d2);

        let (m3_donor_hist, _, _) = model.compose_relation(&m2_donor, &e3_intact);
        let margin_donor_hist = model.query_composition(&m3_donor_hist, (1, 4)) - model.query_composition(&m3_donor_hist, (4, 1));

        let zero_drop = margin_intact - margin_zero;
        let wrong_dst_drop = margin_intact - margin_wrong_dst;
        let wrong_src_drop = margin_intact - margin_wrong_src;
        let donor_hist_drop = margin_intact - margin_donor_hist;
        let unbound_source = wrong_src_drop <= 0.0;

        println!(
            "Seed [{:>2}] {:>4} | Pre: {:>+6.2} | Intact: {:>+6.2} | Zero: {:>+6.2} | W_Dst: {:>+6.2} | W_Src(5->4): {:>+6.2} | Hist: {:>+6.2} | Unbound: {}",
            i, seed, margin_pre, margin_intact, margin_zero, margin_wrong_dst, margin_wrong_src, margin_donor_hist,
            if unbound_source { "YES (Anomaly)" } else { "NO (Bound)" }
        );

        assays.push(ExactReplaySeedAssay {
            seed_index: i,
            seed,
            train_seed,
            margin_pre_edge: margin_pre,
            margin_intact,
            margin_zero_edge: margin_zero,
            margin_wrong_dst: margin_wrong_dst,
            margin_wrong_src: margin_wrong_src,
            margin_donor_hist: margin_donor_hist,
            zero_edge_drop: zero_drop,
            wrong_dst_drop,
            wrong_src_drop,
            donor_hist_drop,
            unbound_source_anomaly: unbound_source,
        });
    }

    let n = assays.len();
    let avg_pre = assays.iter().map(|a| a.margin_pre_edge).sum::<f32>() / n as f32;
    let avg_intact = assays.iter().map(|a| a.margin_intact).sum::<f32>() / n as f32;
    let avg_zero = assays.iter().map(|a| a.margin_zero_edge).sum::<f32>() / n as f32;
    let avg_wrong_dst = assays.iter().map(|a| a.margin_wrong_dst).sum::<f32>() / n as f32;
    let avg_wrong_src = assays.iter().map(|a| a.margin_wrong_src).sum::<f32>() / n as f32;
    let avg_donor_hist = assays.iter().map(|a| a.margin_donor_hist).sum::<f32>() / n as f32;
    let unbound_count = assays.iter().filter(|a| a.unbound_source_anomaly).count();

    println!("--------------------------------------------------------------------------------");
    println!("EXACT REPLAY SUMMARY ACROSS 16 WINNING SCOUT-G ORGANISMS:");
    println!("  Pre-Edge Probe on m2:           {:>+6.2}", avg_pre);
    println!("  Intact 3-Step Sequence:         {:>+6.2}", avg_intact);
    println!("  Zero Final Edge Ablation:       {:>+6.2} (Drop: {:>+6.2})", avg_zero, avg_intact - avg_zero);
    println!("  Wrong Destination (3 -> 5):     {:>+6.2} (Drop: {:>+6.2})", avg_wrong_dst, avg_intact - avg_wrong_dst);
    println!("  Wrong Source (5 -> 4):          {:>+6.2} (Drop: {:>+6.2})", avg_wrong_src, avg_intact - avg_wrong_src);
    println!("  Donor History Transplant:       {:>+6.2} (Drop: {:>+6.2})", avg_donor_hist, avg_intact - avg_donor_hist);
    println!("  Unbound Source Anomaly Rate:    {}/{} ({:.1}%)", unbound_count, n, unbound_count as f32 / n as f32 * 100.0);
    println!("================================================================================");

    let out_dir = Path::new("crates/continuity_garden_core/data");
    fs::create_dir_all(out_dir).expect("Failed to create data directory");
    let out_path = out_dir.join("q17e_h_r1_exact_replay_results.json");
    let json_bytes = serde_json::to_string_pretty(&assays).expect("Failed to serialize results");
    let mut file = File::create(&out_path).expect("Failed to create results file");
    file.write_all(json_bytes.as_bytes()).expect("Failed to write results file");
    println!("Exact Replay telemetry saved to: {}", out_path.display());
}
