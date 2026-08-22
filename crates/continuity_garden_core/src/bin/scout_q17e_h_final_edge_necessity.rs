//! SCOUT-E-Q17E-H: Final-Edge Necessity & Compositional Binding Assay
//!
//! Interrogates the winning Scout G organism (Linear Edge Encoder + Additive Residual Accumulator eta=1.00)
//! to establish whether the incoming compatible final edge (C -> D) is causally necessary alongside
//! the accumulated historical state (m2 for A -> B -> C) to construct the (A -> D) 3-hop relation.

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
pub const SAMPLES_PER_SEED: usize = 100;
pub const N_SEEDS: usize = 16;
pub const AUX_SEEDS: [u64; 16] = [
    42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021, 2223, 2425, 2627, 2829, 3031, 3233,
];

pub const TRAIN_EPOCHS: usize = 120;
pub const TRAIN_BATCHES_PER_EPOCH: usize = 20;
pub const TRAIN_BATCH_SIZE: usize = 16;
pub const TRAIN_LR: f32 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Left,
    Right,
    Stay,
    Jump,
    Up,
    Down,
}

impl Action {
    pub const ALL: [Action; 6] = [
        Action::Left,
        Action::Right,
        Action::Stay,
        Action::Jump,
        Action::Up,
        Action::Down,
    ];

    #[inline(always)]
    pub fn to_f32(&self) -> f32 {
        match self {
            Action::Left => 0.0,
            Action::Right => 1.0,
            Action::Stay => 2.0,
            Action::Jump => 3.0,
            Action::Up => 4.0,
            Action::Down => 5.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionObservation {
    pub src: usize,
    pub action: Action,
    pub dst: usize,
    pub src_jitter: f32,
    pub dst_jitter: f32,
}

impl TransitionObservation {
    #[inline(always)]
    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(OBS_DIM);
        v.push(self.src as f32 / 5.0 + self.src_jitter);
        v.push(self.action.to_f32() / 5.0);
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
pub struct TypedResidualModel {
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

impl TypedResidualModel {
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
        let mut lin = self.b_e.clone();
        for i in 0..EDGE_DIM {
            let row = i * OBS_DIM;
            let mut sum = lin[i];
            for j in 0..OBS_DIM {
                sum += self.w_e[row + j] * x[j];
            }
            lin[i] = sum;
        }
        // Linear edge encoder
        (lin.clone(), lin)
    }

    #[inline(always)]
    pub fn initial_relational_state(&self, e0: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut pre = self.b_m.clone();
        for i in 0..REL_DIM {
            let mut sum = pre[i];
            let row_c = i * EDGE_DIM;
            for j in 0..EDGE_DIM {
                sum += self.w_c[row_c + j] * e0[j];
            }
            pre[i] = sum;
        }
        let post = pre.iter().map(|&x| x.tanh()).collect();
        (pre, post)
    }

    #[inline(always)]
    pub fn step_relational_state(&self, m_prev: &[f32], e_curr: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut pre = self.b_m.clone();
        for i in 0..REL_DIM {
            let mut sum = pre[i];
            let row_m = i * REL_DIM;
            for j in 0..REL_DIM {
                sum += self.w_m[row_m + j] * m_prev[j];
            }
            let row_c = i * EDGE_DIM;
            for j in 0..EDGE_DIM {
                sum += self.w_c[row_c + j] * e_curr[j];
            }
            pre[i] = sum;
        }
        let delta: Vec<f32> = pre.iter().map(|&x| x.tanh()).collect();
        let mut post = vec![0.0f32; REL_DIM];
        for i in 0..REL_DIM {
            post[i] = m_prev[i] + 1.0 * delta[i]; // Additive residual accumulator
        }
        (pre, post)
    }

    #[inline(always)]
    pub fn query_relational(&self, m: &[f32], src: usize, dst: usize) -> f32 {
        let q = [src as f32 / 5.0, dst as f32 / 5.0];
        let mut score = self.b_r;
        for i in 0..REL_DIM {
            let row_q = i * QUERY_DIM;
            let q_proj = self.w_q[row_q] * q[0] + self.w_q[row_q + 1] * q[1];
            score += self.w_r[i] * m[i] * q_proj;
        }
        score
    }

    #[inline(always)]
    pub fn compute_relational_margin(&self, m: &[f32], src: usize, dst: usize) -> f32 {
        let fwd = self.query_relational(m, src, dst);
        let rev = self.query_relational(m, dst, src);
        fwd - rev
    }

    pub fn train_2step(&mut self, train_seed: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(train_seed);

        for _epoch in 0..TRAIN_EPOCHS {
            for _b in 0..TRAIN_BATCHES_PER_EPOCH {
                let mut gw_e = vec![0.0f32; self.w_e.len()];
                let mut gb_e = vec![0.0f32; self.b_e.len()];
                let mut gw_m = vec![0.0f32; self.w_m.len()];
                let mut gw_c = vec![0.0f32; self.w_c.len()];
                let mut gb_m = vec![0.0f32; self.b_m.len()];
                let mut gw_q = vec![0.0f32; self.w_q.len()];
                let mut gw_r = vec![0.0f32; self.w_r.len()];
                let mut gb_r = 0.0f32;

                for _s in 0..TRAIN_BATCHSIZE {
                    let u = rng.gen_range(0..4);
                    let v1 = (u + 1) % 5;
                    let w = (u + 2) % 5;

                    let obs1 = TransitionObservation {
                        src: u,
                        action: Action::Right,
                        dst: v1,
                        src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                        dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                    };
                    let obs2 = TransitionObservation {
                        src: v1,
                        action: Action::Right,
                        dst: w,
                        src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                        dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                    };

                    let (_e1_pre, e1) = self.encode_edge(&obs1);
                    let (_m1_pre, m1) = self.initial_relational_state(&e1);

                    let (_e2_pre, e2) = self.encode_edge(&obs2);
                    let (m2_pre, m2) = self.step_relational_state(&m1, &e2);

                    let m2_margin = self.compute_relational_margin(&m2, u, w);
                    let m1_margin = self.compute_relational_margin(&m1, u, v1);

                    let p2 = sigmoid(m2_margin);
                    let grad_loss2 = p2 - 1.0;

                    let p1 = sigmoid(m1_margin);
                    let grad_loss1 = p1 - 1.0;

                    // Backprop on m2 query head
                    let q_fwd2 = [u as f32 / 5.0, w as f32 / 5.0];
                    let q_rev2 = [w as f32 / 5.0, u as f32 / 5.0];
                    let mut gm2 = vec![0.0f32; REL_DIM];

                    for i in 0..REL_DIM {
                        let row_q = i * QUERY_DIM;
                        let qf = self.w_q[row_q] * q_fwd2[0] + self.w_q[row_q + 1] * q_fwd2[1];
                        let qr = self.w_q[row_q] * q_rev2[0] + self.w_q[row_q + 1] * q_rev2[1];
                        let diff = qf - qr;

                        let term = grad_loss2 * self.w_r[i] * diff;
                        gm2[i] += term;
                        gw_r[i] += grad_loss2 * m2[i] * diff;

                        let q_diff_0 = q_fwd2[0] - q_rev2[0];
                        let q_diff_1 = q_fwd2[1] - q_rev2[1];
                        gw_q[row_q] += grad_loss2 * self.w_r[i] * m2[i] * q_diff_0;
                        gw_q[row_q + 1] += grad_loss2 * self.w_r[i] * m2[i] * q_diff_1;
                    }

                    // Backprop on m1 query head (shared prefix supervision)
                    let q_fwd1 = [u as f32 / 5.0, v1 as f32 / 5.0];
                    let q_rev1 = [v1 as f32 / 5.0, u as f32 / 5.0];
                    let mut gm1 = vec![0.0f32; REL_DIM];

                    for i in 0..REL_DIM {
                        let row_q = i * QUERY_DIM;
                        let qf = self.w_q[row_q] * q_fwd1[0] + self.w_q[row_q + 1] * q_fwd1[1];
                        let qr = self.w_q[row_q] * q_rev1[0] + self.w_q[row_q + 1] * q_rev1[1];
                        let diff = qf - qr;

                        let term = grad_loss1 * self.w_r[i] * diff;
                        gm1[i] += term;
                        gw_r[i] += grad_loss1 * m1[i] * diff;

                        let q_diff_0 = q_fwd1[0] - q_rev1[0];
                        let q_diff_1 = q_fwd1[1] - q_rev1[1];
                        gw_q[row_q] += grad_loss1 * self.w_r[i] * m1[i] * q_diff_0;
                        gw_q[row_q + 1] += grad_loss1 * self.w_r[i] * m1[i] * q_diff_1;
                    }

                    // Step 2 Accumulator Backprop: m2 = m1 + tanh(m2_pre)
                    let mut g_m2_pre = vec![0.0f32; REL_DIM];
                    for i in 0..REL_DIM {
                        let dt = 1.0 - m2_pre[i].tanh().powi(2);
                        g_m2_pre[i] = gm2[i] * dt;
                        gb_m[i] += g_m2_pre[i];
                        gm1[i] += gm2[i]; // Identity residual backprop
                    }

                    let mut ge2 = vec![0.0f32; EDGE_DIM];
                    for i in 0..REL_DIM {
                        let row_m = i * REL_DIM;
                        for j in 0..REL_DIM {
                            gw_m[row_m + j] += g_m2_pre[i] * m1[j];
                            gm1[j] += g_m2_pre[i] * self.w_m[row_m + j];
                        }
                        let row_c = i * EDGE_DIM;
                        for j in 0..EDGE_DIM {
                            gw_c[row_c + j] += g_m2_pre[i] * e2[j];
                            ge2[j] += g_m2_pre[i] * self.w_c[row_c + j];
                        }
                    }

                    // Step 2 Edge Backprop: linear encoder
                    let x2 = obs2.to_vec();
                    for i in 0..EDGE_DIM {
                        gb_e[i] += ge2[i];
                        let row = i * OBS_DIM;
                        for j in 0..OBS_DIM {
                            gw_e[row + j] += ge2[i] * x2[j];
                        }
                    }

                    // Step 1 Initial Accumulator Backprop: m1 = tanh(m1_pre)
                    let (_m1_pre_raw, _) = self.initial_relational_state(&e1);
                    let mut g_m1_pre = vec![0.0f32; REL_DIM];
                    for i in 0..REL_DIM {
                        let dt = 1.0 - _m1_pre_raw[i].tanh().powi(2);
                        g_m1_pre[i] = gm1[i] * dt;
                        gb_m[i] += g_m1_pre[i];
                    }

                    let mut ge1 = vec![0.0f32; EDGE_DIM];
                    for i in 0..REL_DIM {
                        let row_c = i * EDGE_DIM;
                        for j in 0..EDGE_DIM {
                            gw_c[row_c + j] += g_m1_pre[i] * e1[j];
                            ge1[j] += g_m1_pre[i] * self.w_c[row_c + j];
                        }
                    }

                    // Step 1 Edge Backprop: linear encoder
                    let x1 = obs1.to_vec();
                    for i in 0..EDGE_DIM {
                        gb_e[i] += ge1[i];
                        let row = i * OBS_DIM;
                        for j in 0..OBS_DIM {
                            gw_e[row + j] += ge1[i] * x1[j];
                        }
                    }
                }

                let norm = TRAIN_BATCH_SIZE as f32;
                for i in 0..self.w_e.len() {
                    self.w_e[i] -= TRAIN_LR * (gw_e[i] / norm);
                }
                for i in 0..self.b_e.len() {
                    self.b_e[i] -= TRAIN_LR * (gb_e[i] / norm);
                }
                for i in 0..self.w_m.len() {
                    self.w_m[i] -= TRAIN_LR * (gw_m[i] / norm);
                }
                for i in 0..self.w_c.len() {
                    self.w_c[i] -= TRAIN_LR * (gw_c[i] / norm);
                }
                for i in 0..self.b_m.len() {
                    self.b_m[i] -= TRAIN_LR * (gb_m[i] / norm);
                }
                for i in 0..self.w_q.len() {
                    self.w_q[i] -= TRAIN_LR * (gw_q[i] / norm);
                }
                for i in 0..self.w_r.len() {
                    self.w_r[i] -= TRAIN_LR * (gw_r[i] / norm);
                }
                self.b_r -= TRAIN_LR * (gb_r / norm);
            }
        }
    }
}

pub const TRAIN_BATCHSIZE: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedFinalEdgeAssay {
    pub seed: u64,
    pub k2_valid: bool,
    pub k2_margin: f32,
    pub mean_margin_pre_edge: f32,       // Q(A, D) directly on m2
    pub mean_margin_intact: f32,         // C(m2, e_CD)
    pub mean_margin_zero_edge: f32,      // C(m2, 0)
    pub mean_margin_wrong_dst: f32,      // C(m2, e_CE)
    pub mean_margin_wrong_src: f32,      // C(m2, e_XD)
    pub mean_margin_donor_history: f32,  // C(m2_donor, e_CD)
    pub edge_gain: f32,                  // intact - pre_edge
    pub zero_edge_drop: f32,             // intact - zero
    pub wrong_dst_drop: f32,             // intact - wrong_dst
    pub wrong_src_drop: f32,             // intact - wrong_src
    pub donor_history_drop: f32,         // intact - donor_history
    pub final_edge_necessary: bool,      // zero_edge_drop > 0 && wrong_dst_drop > 0 && wrong_src_drop > 0
    pub history_necessary: bool,         // donor_history_drop > 0
    pub dual_compositional_binding: bool,// final_edge_necessary && history_necessary && intact > 0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoutHSummary {
    pub n_seeds: usize,
    pub k2_retention_pass: usize,
    pub intact_k3_positive_count: usize,
    pub final_edge_necessity_count: usize,
    pub wrong_dst_sensitivity_count: usize,
    pub wrong_src_sensitivity_count: usize,
    pub history_necessity_count: usize,
    pub dual_compositional_binding_count: usize,
    pub overall_mean_margin_pre_edge: f32,
    pub overall_mean_margin_intact: f32,
    pub overall_mean_margin_zero_edge: f32,
    pub overall_mean_margin_wrong_dst: f32,
    pub overall_mean_margin_wrong_src: f32,
    pub overall_mean_margin_donor_history: f32,
    pub seeds: Vec<SeedFinalEdgeAssay>,
}

fn main() {
    println!("================================================================================");
    println!("SCOUT-E-Q17E-H: Final-Edge Necessity & Compositional Binding Assay");
    println!("Testing: Is incoming edge (C->D) causally necessary alongside history (m2) for (A->D)?");
    println!("================================================================================");

    let mut seed_assays = Vec::new();

    for &seed in &AUX_SEEDS {
        let mut model = TypedResidualModel::new_init(seed);
        model.train_2step(seed + 999);

        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xBEEFBEEF);

        // 1. Verify k=2 base validity
        let mut k2_margins = Vec::new();
        for _ in 0..SAMPLES_PER_SEED {
            let u = rng.gen_range(0..4);
            let v1 = (u + 1) % 5;
            let w = (u + 2) % 5;

            let obs1 = TransitionObservation {
                src: u,
                action: Action::Right,
                dst: v1,
                src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
            };
            let obs2 = TransitionObservation {
                src: v1,
                action: Action::Right,
                dst: w,
                src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
            };
            let (_, e1) = model.encode_edge(&obs1);
            let (_, m1) = model.initial_relational_state(&e1);
            let (_, e2) = model.encode_edge(&obs2);
            let (_, m2) = model.step_relational_state(&m1, &e2);

            let m = model.compute_relational_margin(&m2, u, w);
            k2_margins.push(m);
        }
        let mean_k2 = k2_margins.iter().sum::<f32>() / (SAMPLES_PER_SEED as f32);
        let k2_valid = mean_k2 > 0.0;

        // 2. Interrogate 3-step sequence A -> B -> C -> D under 5 conditions
        let mut pre_edge_margins = Vec::new();
        let mut intact_margins = Vec::new();
        let mut zero_margins = Vec::new();
        let mut wrong_dst_margins = Vec::new();
        let mut wrong_src_margins = Vec::new();
        let mut donor_hist_margins = Vec::new();

        for _ in 0..SAMPLES_PER_SEED {
            let a = rng.gen_range(0..2); // 0 or 1 so A, B, C, D are valid nodes
            let b = (a + 1) % 5;
            let c = (a + 2) % 5;
            let d = (a + 3) % 5;
            let e = (a + 4) % 5; // wrong destination (distinct from D)
            let x = (a + 4) % 5; // wrong source (distinct from C)

            // Step 1: A -> B
            let obs1 = TransitionObservation {
                src: a,
                action: Action::Right,
                dst: b,
                src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
            };
            let (_, e1) = model.encode_edge(&obs1);
            let (_, m1) = model.initial_relational_state(&e1);

            // Step 2: B -> C
            let obs2 = TransitionObservation {
                src: b,
                action: Action::Right,
                dst: c,
                src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
            };
            let (_, e2) = model.encode_edge(&obs2);
            let (_, m2) = model.step_relational_state(&m1, &e2);

            // Condition 1: Query (A, D) directly on m2 (Pre-edge probe)
            let m_pre = model.compute_relational_margin(&m2, a, d);
            pre_edge_margins.push(m_pre);

            // Step 3 Compatible Edge: C -> D
            let obs3_intact = TransitionObservation {
                src: c,
                action: Action::Right,
                dst: d,
                src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
            };
            let (_, e3_cd) = model.encode_edge(&obs3_intact);

            // Condition 2: Intact C(m2, e_CD)
            let (_, m3_intact) = model.step_relational_state(&m2, &e3_cd);
            let m_intact = model.compute_relational_margin(&m3_intact, a, d);
            intact_margins.push(m_intact);

            // Condition 3: Zero final edge C(m2, 0)
            let e_zero = vec![0.0f32; EDGE_DIM];
            let (_, m3_zero) = model.step_relational_state(&m2, &e_zero);
            let m_zero = model.compute_relational_margin(&m3_zero, a, d);
            zero_margins.push(m_zero);

            // Condition 4: Wrong destination edge C(m2, e_CE)
            let obs3_wrong_dst = TransitionObservation {
                src: c,
                action: Action::Right,
                dst: e,
                src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
            };
            let (_, e3_ce) = model.encode_edge(&obs3_wrong_dst);
            let (_, m3_wrong_dst) = model.step_relational_state(&m2, &e3_ce);
            let m_wrong_dst = model.compute_relational_margin(&m3_wrong_dst, a, d);
            wrong_dst_margins.push(m_wrong_dst);

            // Condition 5: Wrong source edge C(m2, e_XD)
            let obs3_wrong_src = TransitionObservation {
                src: x,
                action: Action::Right,
                dst: d,
                src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
            };
            let (_, e3_xd) = model.encode_edge(&obs3_wrong_src);
            let (_, m3_wrong_src) = model.step_relational_state(&m2, &e3_xd);
            let m_wrong_src = model.compute_relational_margin(&m3_wrong_src, a, d);
            wrong_src_margins.push(m_wrong_src);

            // Condition 6: Donor history transplant C(m2_donor, e_CD)
            // Donor path: Reversed sequence D -> C -> B with separate jitter
            let obs_donor1 = TransitionObservation {
                src: d,
                action: Action::Left,
                dst: c,
                src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
            };
            let (_, e_d1) = model.encode_edge(&obs_donor1);
            let (_, m_d1) = model.initial_relational_state(&e_d1);
            let obs_donor2 = TransitionObservation {
                src: c,
                action: Action::Left,
                dst: b,
                src_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
                dst_jitter: (rng.gen::<f32>() - 0.5) * 0.02,
            };
            let (_, e_d2) = model.encode_edge(&obs_donor2);
            let (_, m2_donor) = model.step_relational_state(&m_d1, &e_d2);

            let (_, m3_donor_hist) = model.step_relational_state(&m2_donor, &e3_cd);
            let m_donor_hist = model.compute_relational_margin(&m3_donor_hist, a, d);
            donor_hist_margins.push(m_donor_hist);
        }

        let mean_pre = pre_edge_margins.iter().sum::<f32>() / (SAMPLES_PER_SEED as f32);
        let mean_intact = intact_margins.iter().sum::<f32>() / (SAMPLES_PER_SEED as f32);
        let mean_zero = zero_margins.iter().sum::<f32>() / (SAMPLES_PER_SEED as f32);
        let mean_wrong_dst = wrong_dst_margins.iter().sum::<f32>() / (SAMPLES_PER_SEED as f32);
        let mean_wrong_src = wrong_src_margins.iter().sum::<f32>() / (SAMPLES_PER_SEED as f32);
        let mean_donor_hist = donor_hist_margins.iter().sum::<f32>() / (SAMPLES_PER_SEED as f32);

        let edge_gain = mean_intact - mean_pre;
        let zero_drop = mean_intact - mean_zero;
        let wrong_dst_drop = mean_intact - mean_wrong_dst;
        let wrong_src_drop = mean_intact - mean_wrong_src;
        let donor_hist_drop = mean_intact - mean_donor_hist;

        let final_edge_necessary = zero_drop > 0.0 && wrong_dst_drop > 0.0 && wrong_src_drop > 0.0;
        let history_necessary = donor_hist_drop > 0.0;
        let dual_binding = final_edge_necessary && history_necessary && mean_intact > 0.0;

        println!(
            "Seed {:>4} | k2: {:>+6.2} | Pre: {:>+6.2} | Intact: {:>+6.2} | Zero: {:>+6.2} | W_Dst: {:>+6.2} | W_Src: {:>+6.2} | Hist: {:>+6.2} | Dual: {}",
            seed, mean_k2, mean_pre, mean_intact, mean_zero, mean_wrong_dst, mean_wrong_src, mean_donor_hist,
            if dual_binding { "PASS" } else { "FAIL" }
        );

        seed_assays.push(SeedFinalEdgeAssay {
            seed,
            k2_valid,
            k2_margin: mean_k2,
            mean_margin_pre_edge: mean_pre,
            mean_margin_intact: mean_intact,
            mean_margin_zero_edge: mean_zero,
            mean_margin_wrong_dst: mean_wrong_dst,
            mean_margin_wrong_src: mean_wrong_src,
            mean_margin_donor_history: mean_donor_hist,
            edge_gain,
            zero_edge_drop: zero_drop,
            wrong_dst_drop,
            wrong_src_drop,
            donor_history_drop: donor_hist_drop,
            final_edge_necessary,
            history_necessary,
            dual_compositional_binding: dual_binding,
        });
    }

    let n = seed_assays.len();
    let k2_pass = seed_assays.iter().filter(|s| s.k2_valid).count();
    let intact_pass = seed_assays.iter().filter(|s| s.mean_margin_intact > 0.0).count();
    let final_edge_pass = seed_assays.iter().filter(|s| s.final_edge_necessary).count();
    let wrong_dst_pass = seed_assays.iter().filter(|s| s.wrong_dst_drop > 0.0).count();
    let wrong_src_pass = seed_assays.iter().filter(|s| s.wrong_src_drop > 0.0).count();
    let history_pass = seed_assays.iter().filter(|s| s.history_necessary).count();
    let dual_pass = seed_assays.iter().filter(|s| s.dual_compositional_binding).count();

    let avg_pre = seed_assays.iter().map(|s| s.mean_margin_pre_edge).sum::<f32>() / n as f32;
    let avg_intact = seed_assays.iter().map(|s| s.mean_margin_intact).sum::<f32>() / n as f32;
    let avg_zero = seed_assays.iter().map(|s| s.mean_margin_zero_edge).sum::<f32>() / n as f32;
    let avg_wrong_dst = seed_assays.iter().map(|s| s.mean_margin_wrong_dst).sum::<f32>() / n as f32;
    let avg_wrong_src = seed_assays.iter().map(|s| s.mean_margin_wrong_src).sum::<f32>() / n as f32;
    let avg_donor_hist = seed_assays.iter().map(|s| s.mean_margin_donor_history).sum::<f32>() / n as f32;

    println!("--------------------------------------------------------------------------------");
    println!("SCOUT H SYNTHESIS ACROSS {} SEEDS:", n);
    println!("  k=2 Validity Floor:             {}/{} ({:.1}%)", k2_pass, n, k2_pass as f32 / n as f32 * 100.0);
    println!("  Intact k=3 Direction (A->D):    {}/{} ({:.1}%) [Overall Mean: {:>+6.2}]", intact_pass, n, intact_pass as f32 / n as f32 * 100.0, avg_intact);
    println!("  Pre-Edge Probe on m2:           [Overall Mean: {:>+6.2}] (Edge Gain: {:>+6.2})", avg_pre, avg_intact - avg_pre);
    println!("  Zero Final Edge Ablation Drop:  [Overall Mean: {:>+6.2}] (Ablation Drop: {:>+6.2})", avg_zero, avg_intact - avg_zero);
    println!("  Wrong Destination Drop (C->E):  {}/{} pass [Overall Mean: {:>+6.2}] (Drop: {:>+6.2})", wrong_dst_pass, n, avg_wrong_dst, avg_intact - avg_wrong_dst);
    println!("  Wrong Source Drop (X->D):       {}/{} pass [Overall Mean: {:>+6.2}] (Drop: {:>+6.2})", wrong_src_pass, n, avg_wrong_src, avg_intact - avg_wrong_src);
    println!("  Donor History Swap Drop:        {}/{} pass [Overall Mean: {:>+6.2}] (Drop: {:>+6.2})", history_pass, n, avg_donor_hist, avg_intact - avg_donor_hist);
    println!("  DUAL COMPOSITIONAL BINDING:     {}/{} ({:.1}%)", dual_pass, n, dual_pass as f32 / n as f32 * 100.0);
    println!("================================================================================");

    let summary = ScoutHSummary {
        n_seeds: n,
        k2_retention_pass: k2_pass,
        intact_k3_positive_count: intact_pass,
        final_edge_necessity_count: final_edge_pass,
        wrong_dst_sensitivity_count: wrong_dst_pass,
        wrong_src_sensitivity_count: wrong_src_pass,
        history_necessity_count: history_pass,
        dual_compositional_binding_count: dual_pass,
        overall_mean_margin_pre_edge: avg_pre,
        overall_mean_margin_intact: avg_intact,
        overall_mean_margin_zero_edge: avg_zero,
        overall_mean_margin_wrong_dst: avg_wrong_dst,
        overall_mean_margin_wrong_src: avg_wrong_src,
        overall_mean_margin_donor_history: avg_donor_hist,
        seeds: seed_assays,
    };

    let out_dir = Path::new("crates/continuity_garden_core/data");
    fs::create_dir_all(out_dir).expect("Failed to create data directory");
    let out_path = out_dir.join("q17e_h_final_edge_results.json");
    let json_bytes = serde_json::to_string_pretty(&summary).expect("Failed to serialize Scout H results");
    let mut file = File::create(&out_path).expect("Failed to create results file");
    file.write_all(json_bytes.as_bytes()).expect("Failed to write results file");
    println!("Scout H telemetry saved to: {}", out_path.display());
}
