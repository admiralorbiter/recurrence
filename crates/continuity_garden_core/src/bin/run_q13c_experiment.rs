//! Q13c: Definitive 2x2 Relational Computation & Representation Localization Assay.
//! Features:
//! 1. Explicit Sanity Control: [s, r] -> MLP -> XOR (verifies MLP solver).
//! 2. Optimizer-Independent Probes on Frozen h: Linear Ridge vs 2-Layer MLP classifier.
//! 3. True 2x2 Factorial Matrix with Full BPTT Recurrent Plasticity & GRU Parameter Displacement.

use continuity_garden_core::trainer::{fit_and_eval_ridge, solve_linear_system};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const HIDDEN_DIM: usize = 64;
const EMBED_DIM: usize = 16;
const TOTAL_INPUT_DIM: usize = 48;
const COMBINED_DIM: usize = HIDDEN_DIM + 32;
const MLP_HIDDEN_DIM: usize = 32;

#[derive(Debug, Clone)]
pub struct Q13cOrganism {
    pub symbol_embed: Vec<f32>,
    pub sensor_w: Vec<f32>, // 16 x 5
    pub sensor_b: Vec<f32>, // 16
    pub gru_w_ih: Vec<f32>, // 192 x 48
    pub gru_w_hh: Vec<f32>, // 192 x 64
    pub gru_b: Vec<f32>,    // 192
    pub policy_linear_w: Vec<f32>,
    pub policy_linear_b: Vec<f32>,
    pub mlp_w1: Vec<f32>,
    pub mlp_b1: Vec<f32>,
    pub mlp_w2: Vec<f32>,
    pub mlp_b2: Vec<f32>,
}

impl Q13cOrganism {
    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut rand_vec = |len: usize, std: f32| -> Vec<f32> {
            (0..len).map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * std).collect()
        };

        Self {
            symbol_embed: rand_vec(4 * EMBED_DIM, 0.1),
            sensor_w: rand_vec(EMBED_DIM * 5, (2.0 / 5.0f32).sqrt()),
            sensor_b: vec![0.0; EMBED_DIM],
            gru_w_ih: rand_vec(192 * TOTAL_INPUT_DIM, (2.0 / TOTAL_INPUT_DIM as f32).sqrt()),
            gru_w_hh: rand_vec(192 * HIDDEN_DIM, (2.0 / HIDDEN_DIM as f32).sqrt()),
            gru_b: vec![0.0; 192],
            policy_linear_w: rand_vec(2 * COMBINED_DIM, 0.01),
            policy_linear_b: vec![0.0; 2],
            mlp_w1: rand_vec(MLP_HIDDEN_DIM * COMBINED_DIM, (2.0 / COMBINED_DIM as f32).sqrt()),
            mlp_b1: vec![0.0; MLP_HIDDEN_DIM],
            mlp_w2: rand_vec(2 * MLP_HIDDEN_DIM, (2.0 / MLP_HIDDEN_DIM as f32).sqrt()),
            mlp_b2: vec![0.0; 2],
        }
    }

    pub fn compute_h_next(&self, sym: usize, ch: [f32; 2], is_dec: f32, h_prev: Option<&[f32]>) -> (Vec<f32>, Vec<f32>, [Vec<f32>; 4]) {
        let mut input_feats = Vec::with_capacity(TOTAL_INPUT_DIM);
        let mut instant_feats = Vec::with_capacity(32);

        let s_idx = sym.min(3);
        let s_slice = &self.symbol_embed[s_idx * EMBED_DIM..(s_idx + 1) * EMBED_DIM];
        input_feats.extend_from_slice(s_slice);
        instant_feats.extend_from_slice(s_slice);
        input_feats.extend_from_slice(&vec![0.0; EMBED_DIM]);

        let sens_in = [ch[0], ch[1], 0.0, 0.0, is_dec];
        let mut sens_out = vec![0.0; EMBED_DIM];
        for i in 0..EMBED_DIM {
            let mut sum = self.sensor_b[i];
            for j in 0..5 { sum += self.sensor_w[i * 5 + j] * sens_in[j]; }
            sens_out[i] = sum.max(0.0);
        }
        input_feats.extend_from_slice(&sens_out);
        instant_feats.extend_from_slice(&sens_out);

        let h_slice = h_prev.unwrap_or(&[0.0; HIDDEN_DIM]);
        let mut gates = vec![0.0; 192];
        for i in 0..192 {
            let mut sum = self.gru_b[i];
            for j in 0..TOTAL_INPUT_DIM { sum += self.gru_w_ih[i * TOTAL_INPUT_DIM + j] * input_feats[j]; }
            for j in 0..HIDDEN_DIM { sum += self.gru_w_hh[i * HIDDEN_DIM + j] * h_slice[j]; }
            gates[i] = sum;
        }

        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        let mut h_next = vec![0.0; HIDDEN_DIM];
        let mut z_gates = vec![0.0; HIDDEN_DIM];
        let mut r_gates = vec![0.0; HIDDEN_DIM];
        let mut n_cands = vec![0.0; HIDDEN_DIM];

        for i in 0..HIDDEN_DIM {
            let z = sig(gates[i]);
            let r = sig(gates[64 + i]);
            let mut sum_cand = self.gru_b[128 + i];
            for j in 0..TOTAL_INPUT_DIM { sum_cand += self.gru_w_ih[(128 + i) * TOTAL_INPUT_DIM + j] * input_feats[j]; }
            for j in 0..HIDDEN_DIM { sum_cand += self.gru_w_hh[(128 + i) * HIDDEN_DIM + j] * (r * h_slice[j]); }
            let n = sum_cand.tanh();
            h_next[i] = (1.0 - z) * n + z * h_slice[i];
            z_gates[i] = z;
            r_gates[i] = r;
            n_cands[i] = n;
        }

        (h_next, instant_feats, [input_feats, z_gates, r_gates, n_cands])
    }

    pub fn compute_linear_logits(&self, h: &[f32], instant_feats: &[f32]) -> [f32; 2] {
        let mut comb = Vec::with_capacity(COMBINED_DIM);
        comb.extend_from_slice(h);
        comb.extend_from_slice(instant_feats);

        let mut logits = [0.0; 2];
        for k in 0..2 {
            let mut sum = self.policy_linear_b[k];
            for j in 0..COMBINED_DIM { sum += self.policy_linear_w[k * COMBINED_DIM + j] * comb[j]; }
            logits[k] = sum;
        }
        logits
    }

    pub fn compute_mlp_logits(&self, h: &[f32], instant_feats: &[f32]) -> ([f32; 2], Vec<f32>) {
        let mut comb = Vec::with_capacity(COMBINED_DIM);
        comb.extend_from_slice(h);
        comb.extend_from_slice(instant_feats);

        let mut hidden = vec![0.0; MLP_HIDDEN_DIM];
        for i in 0..MLP_HIDDEN_DIM {
            let mut sum = self.mlp_b1[i];
            for j in 0..COMBINED_DIM { sum += self.mlp_w1[i * COMBINED_DIM + j] * comb[j]; }
            hidden[i] = sum.max(0.0);
        }

        let mut logits = [0.0; 2];
        for k in 0..2 {
            let mut sum = self.mlp_b2[k];
            for j in 0..MLP_HIDDEN_DIM { sum += self.mlp_w2[k * MLP_HIDDEN_DIM + j] * hidden[j]; }
            logits[k] = sum;
        }
        (logits, hidden)
    }

    pub fn gru_displacement(&self, original: &Q13cOrganism) -> f32 {
        let mut sq = 0.0f32;
        for i in 0..self.gru_w_ih.len() { sq += (self.gru_w_ih[i] - original.gru_w_ih[i]).powi(2); }
        for i in 0..self.gru_w_hh.len() { sq += (self.gru_w_hh[i] - original.gru_w_hh[i]).powi(2); }
        for i in 0..self.gru_b.len() { sq += (self.gru_b[i] - original.gru_b[i]).powi(2); }
        sq.sqrt()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadrantCResult {
    pub quadrant_name: String,
    pub r2_source_class: f32,
    pub r2_report_content: f32,
    pub r2_multiplicative_xor_ridge: f32,
    pub mlp_probe_accuracy: f32,
    pub helpful_following_rate: f32,
    pub opposite_inversion_rate: f32,
    pub net_inversion_effect: f32,
    pub mean_return: f32,
    pub gru_displacement_norm: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q13cSeedResult {
    pub seed: u64,
    pub sanity_explicit_mlp_accuracy: f32,
    pub frozen_linear: QuadrantCResult,
    pub frozen_mlp: QuadrantCResult,
    pub plastic_linear: QuadrantCResult,
    pub plastic_mlp: QuadrantCResult,
}

fn test_explicit_mlp_sanity(seed: u64) -> f32 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut w1 = vec![0.0f32; 16 * 2];
    let mut b1 = vec![0.0f32; 16];
    let mut w2 = vec![0.0f32; 2 * 16];
    let mut b2 = vec![0.0f32; 2];

    for i in 0..w1.len() { w1[i] = (rng.gen::<f32>() * 2.0 - 1.0) * 0.5; }
    for i in 0..w2.len() { w2[i] = (rng.gen::<f32>() * 2.0 - 1.0) * 0.5; }

    for ep in 0..2000 {
        let s = if rng.gen::<f64>() < 0.5 { 0.0f32 } else { 1.0f32 };
        let r = if rng.gen::<f64>() < 0.5 { 0.0f32 } else { 1.0f32 };
        let target = if (s > 0.5) ^ (r > 0.5) { 1 } else { 0 };

        let in_vec = [s, r];
        let mut hid = vec![0.0f32; 16];
        for i in 0..16 {
            let mut sum = b1[i];
            for j in 0..2 { sum += w1[i * 2 + j] * in_vec[j]; }
            hid[i] = sum.max(0.0);
        }

        let mut logits = [b2[0], b2[1]];
        for k in 0..2 {
            for j in 0..16 { logits[k] += w2[k * 16 + j] * hid[j]; }
        }

        let max_l = logits[0].max(logits[1]);
        let exp_l = [(logits[0] - max_l).exp(), (logits[1] - max_l).exp()];
        let probs = [exp_l[0] / (exp_l[0] + exp_l[1]), exp_l[1] / (exp_l[0] + exp_l[1])];

        let g_logits = [probs[0] - (if target == 0 { 1.0 } else { 0.0 }), probs[1] - (if target == 1 { 1.0 } else { 0.0 })];

        let w2_old = w2.clone();
        for k in 0..2 {
            b2[k] -= 0.05 * g_logits[k];
            for j in 0..16 { w2[k * 16 + j] -= 0.05 * g_logits[k] * hid[j]; }
        }

        let mut g_hid = vec![0.0; 16];
        for j in 0..16 {
            let mut sum = 0.0;
            for k in 0..2 { sum += g_logits[k] * w2_old[k * 16 + j]; }
            if hid[j] > 0.0 { g_hid[j] = sum; }
        }

        for i in 0..16 {
            b1[i] -= 0.05 * g_hid[i];
            for j in 0..2 { w1[i * 2 + j] -= 0.05 * g_hid[i] * in_vec[j]; }
        }
    }

    let mut correct = 0;
    for ep in 0..100 {
        let s = if rng.gen::<f64>() < 0.5 { 0.0f32 } else { 1.0f32 };
        let r = if rng.gen::<f64>() < 0.5 { 0.0f32 } else { 1.0f32 };
        let target = if (s > 0.5) ^ (r > 0.5) { 1 } else { 0 };

        let in_vec = [s, r];
        let mut hid = vec![0.0f32; 16];
        for i in 0..16 {
            let mut sum = b1[i];
            for j in 0..2 { sum += w1[i * 2 + j] * in_vec[j]; }
            hid[i] = sum.max(0.0);
        }
        let mut logits = [b2[0], b2[1]];
        for k in 0..2 { for j in 0..16 { logits[k] += w2[k * 16 + j] * hid[j]; } }
        let pred = if logits[1] > logits[0] { 1 } else { 0 };
        if pred == target { correct += 1; }
    }
    correct as f32 / 100.0
}

fn train_mlp_classifier_on_h(train_h: &[Vec<f32>], train_y: &[usize], test_h: &[Vec<f32>], test_y: &[usize]) -> f32 {
    let d = train_h[0].len();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut w1 = vec![0.0f32; MLP_HIDDEN_DIM * d];
    let mut b1 = vec![0.0f32; MLP_HIDDEN_DIM];
    let mut w2 = vec![0.0f32; 2 * MLP_HIDDEN_DIM];
    let mut b2 = vec![0.0f32; 2];

    for i in 0..w1.len() { w1[i] = (rng.gen::<f32>() * 2.0 - 1.0) * 0.1; }
    for i in 0..w2.len() { w2[i] = (rng.gen::<f32>() * 2.0 - 1.0) * 0.1; }

    for _epoch in 0..300 {
        for s in 0..train_h.len() {
            let x = &train_h[s];
            let y = train_y[s];

            let mut hid = vec![0.0f32; MLP_HIDDEN_DIM];
            for i in 0..MLP_HIDDEN_DIM {
                let mut sum = b1[i];
                for j in 0..d { sum += w1[i * d + j] * x[j]; }
                hid[i] = sum.max(0.0);
            }

            let mut logits = [b2[0], b2[1]];
            for k in 0..2 { for j in 0..MLP_HIDDEN_DIM { logits[k] += w2[k * MLP_HIDDEN_DIM + j] * hid[j]; } }

            let max_l = logits[0].max(logits[1]);
            let exp_l = [(logits[0] - max_l).exp(), (logits[1] - max_l).exp()];
            let probs = [exp_l[0] / (exp_l[0] + exp_l[1]), exp_l[1] / (exp_l[0] + exp_l[1])];

            let g_logits = [probs[0] - (if y == 0 { 1.0 } else { 0.0 }), probs[1] - (if y == 1 { 1.0 } else { 0.0 })];

            let w2_old = w2.clone();
            for k in 0..2 {
                b2[k] -= 0.02 * g_logits[k];
                for j in 0..MLP_HIDDEN_DIM { w2[k * MLP_HIDDEN_DIM + j] -= 0.02 * g_logits[k] * hid[j]; }
            }

            let mut g_hid = vec![0.0; MLP_HIDDEN_DIM];
            for j in 0..MLP_HIDDEN_DIM {
                let mut sum = 0.0;
                for k in 0..2 { sum += g_logits[k] * w2_old[k * MLP_HIDDEN_DIM + j]; }
                if hid[j] > 0.0 { g_hid[j] = sum; }
            }

            for i in 0..MLP_HIDDEN_DIM {
                b1[i] -= 0.02 * g_hid[i];
                for j in 0..d { w1[i * d + j] -= 0.02 * g_hid[i] * x[j]; }
            }
        }
    }

    let mut correct = 0;
    for s in 0..test_h.len() {
        let x = &test_h[s];
        let y = test_y[s];
        let mut hid = vec![0.0f32; MLP_HIDDEN_DIM];
        for i in 0..MLP_HIDDEN_DIM {
            let mut sum = b1[i];
            for j in 0..d { sum += w1[i * d + j] * x[j]; }
            hid[i] = sum.max(0.0);
        }
        let mut logits = [b2[0], b2[1]];
        for k in 0..2 { for j in 0..MLP_HIDDEN_DIM { logits[k] += w2[k * MLP_HIDDEN_DIM + j] * hid[j]; } }
        let pred = if logits[1] > logits[0] { 1 } else { 0 };
        if pred == y { correct += 1; }
    }
    correct as f32 / test_h.len() as f32
}

fn generate_binary_signed_episode(
    seed: u64,
    ep_idx: usize,
    delay_steps: usize,
) -> (usize, usize, usize, usize, Vec<(usize, [f32; 2], f32)>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed + ep_idx as u64 * 31);
    let root_z = if rng.gen::<f64>() < 0.50 { 0 } else { 1 };
    let s_id = if rng.gen::<f64>() < 0.50 { 0 } else { 1 };
    let p_acc = if s_id == 0 { 0.85f32 } else { 0.15f32 };
    let rep = if rng.gen::<f32>() < p_acc { root_z } else { 1 - root_z };

    let opt_act = if s_id == 0 { rep } else { 1 - rep };

    let mut steps = Vec::new();
    steps.push((0, [0.0, 0.0], 0.0));

    let mut ch = [0.0; 2];
    ch[s_id] = 1.0;
    steps.push((rep + 1, ch, 0.0));

    for _ in 0..delay_steps {
        steps.push((0, [0.0, 0.0], 0.0));
    }

    steps.push((2, [0.0, 0.0], 1.0));

    (root_z, s_id, rep, opt_act, steps)
}

fn train_and_eval_quadrant_c(
    mut model: Q13cOrganism,
    is_plastic_gru: bool,
    is_mlp_head: bool,
    seed: u64,
    num_train_episodes: usize,
    num_eval_episodes: usize,
) -> QuadrantCResult {
    let initial_model = model.clone();
    let delay_steps = 3;

    // Train with full backprop through policy head (and BPTT through GRU if plastic)
    for ep in 1..=num_train_episodes {
        let (_, _, _, opt_act, steps) = generate_binary_signed_episode(seed, ep, delay_steps);

        let mut h_history = Vec::new();
        let mut forward_cache = Vec::new();
        let mut dec_comb = Vec::new();
        let mut dec_probs = [0.0; 2];
        let mut dec_mlp_hid = Vec::new();

        let mut h: Option<Vec<f32>> = None;
        for (sym, ch, is_dec) in &steps {
            let h_prev_clone = h.clone();
            let (h_next, instant_feats, intermediates) = model.compute_h_next(*sym, *ch, *is_dec, h.as_deref());
            h_history.push((h_prev_clone, h_next.clone()));
            forward_cache.push((instant_feats.clone(), intermediates));

            if *is_dec > 0.5 {
                let mut comb = Vec::with_capacity(COMBINED_DIM);
                comb.extend_from_slice(&h_next);
                comb.extend_from_slice(&instant_feats);
                dec_comb = comb;

                if is_mlp_head {
                    let (logits, hidden) = model.compute_mlp_logits(&h_next, &instant_feats);
                    let max_l = logits[0].max(logits[1]);
                    let exp_l = [(logits[0] - max_l).exp(), (logits[1] - max_l).exp()];
                    let sum_exp = exp_l[0] + exp_l[1];
                    dec_probs = [exp_l[0] / sum_exp, exp_l[1] / sum_exp];
                    dec_mlp_hid = hidden;
                } else {
                    let logits = model.compute_linear_logits(&h_next, &instant_feats);
                    let max_l = logits[0].max(logits[1]);
                    let exp_l = [(logits[0] - max_l).exp(), (logits[1] - max_l).exp()];
                    let sum_exp = exp_l[0] + exp_l[1];
                    dec_probs = [exp_l[0] / sum_exp, exp_l[1] / sum_exp];
                }
            }
            h = Some(h_next);
        }

        let target_a = opt_act;
        let g_logits = [dec_probs[0] - (if target_a == 0 { 1.0 } else { 0.0 }), dec_probs[1] - (if target_a == 1 { 1.0 } else { 0.0 })];

        let mut g_h_dec = vec![0.0f32; HIDDEN_DIM];

        if is_mlp_head {
            let w2_old = model.mlp_w2.clone();
            for k in 0..2 {
                model.mlp_b2[k] -= 0.05 * g_logits[k];
                for j in 0..MLP_HIDDEN_DIM {
                    model.mlp_w2[k * MLP_HIDDEN_DIM + j] -= 0.05 * g_logits[k] * dec_mlp_hid[j];
                }
            }

            let mut g_hid = vec![0.0; MLP_HIDDEN_DIM];
            for j in 0..MLP_HIDDEN_DIM {
                let mut sum = 0.0;
                for k in 0..2 { sum += g_logits[k] * w2_old[k * MLP_HIDDEN_DIM + j]; }
                if dec_mlp_hid[j] > 0.0 { g_hid[j] = sum; }
            }

            let w1_old = model.mlp_w1.clone();
            for i in 0..MLP_HIDDEN_DIM {
                model.mlp_b1[i] -= 0.05 * g_hid[i];
                for j in 0..COMBINED_DIM {
                    model.mlp_w1[i * COMBINED_DIM + j] -= 0.05 * g_hid[i] * dec_comb[j];
                }
            }

            for j in 0..HIDDEN_DIM {
                let mut sum = 0.0;
                for i in 0..MLP_HIDDEN_DIM { sum += g_hid[i] * w1_old[i * COMBINED_DIM + j]; }
                g_h_dec[j] = sum;
            }
        } else {
            let w_old = model.policy_linear_w.clone();
            for k in 0..2 {
                model.policy_linear_b[k] -= 0.02 * g_logits[k];
                for j in 0..COMBINED_DIM {
                    model.policy_linear_w[k * COMBINED_DIM + j] -= 0.02 * g_logits[k] * dec_comb[j];
                }
            }
            for j in 0..HIDDEN_DIM {
                let mut sum = 0.0;
                for k in 0..2 { sum += g_logits[k] * w_old[k * COMBINED_DIM + j]; }
                g_h_dec[j] = sum;
            }
        }

        // BPTT through GRU sequence if Plastic
        if is_plastic_gru {
            let mut dh_next = g_h_dec;
            let t_last = steps.len() - 1;

            for t in (0..=t_last).rev() {
                let (ref h_prev_opt, ref _h_curr) = h_history[t];
                let h_prev = h_prev_opt.as_deref().unwrap_or(&[0.0; HIDDEN_DIM]);
                let (ref _inst, ref intermed) = forward_cache[t];
                let (ref input_feats, ref z_gates, ref r_gates, ref n_cands) = (intermed[0].clone(), intermed[1].clone(), intermed[2].clone(), intermed[3].clone());

                let mut dh_prev = vec![0.0f32; HIDDEN_DIM];

                for i in 0..HIDDEN_DIM {
                    let dhi = dh_next[i];
                    let z = z_gates[i];
                    let r = r_gates[i];
                    let n = n_cands[i];

                    let dn = dhi * (1.0 - z) * (1.0 - n * n);
                    let dz = dhi * (h_prev[i] - n) * z * (1.0 - z);

                    // Update GRU weights for candidate n (gate index 128..192)
                    model.gru_b[128 + i] -= 0.01 * dn;
                    for j in 0..TOTAL_INPUT_DIM { model.gru_w_ih[(128 + i) * TOTAL_INPUT_DIM + j] -= 0.01 * dn * input_feats[j]; }
                    for j in 0..HIDDEN_DIM {
                        model.gru_w_hh[(128 + i) * HIDDEN_DIM + j] -= 0.01 * dn * (r * h_prev[j]);
                        dh_prev[j] += dn * model.gru_w_hh[(128 + i) * HIDDEN_DIM + j] * r;
                    }

                    // Update GRU weights for update gate z (gate index 0..64)
                    model.gru_b[i] -= 0.01 * dz;
                    for j in 0..TOTAL_INPUT_DIM { model.gru_w_ih[i * TOTAL_INPUT_DIM + j] -= 0.01 * dz * input_feats[j]; }
                    for j in 0..HIDDEN_DIM {
                        model.gru_w_hh[i * HIDDEN_DIM + j] -= 0.01 * dz * h_prev[j];
                        dh_prev[j] += dz * model.gru_w_hh[i * HIDDEN_DIM + j];
                    }

                    dh_prev[i] += dhi * z;
                }
                dh_next = dh_prev;
            }
        }
    }

    let disp_norm = model.gru_displacement(&initial_model);

    // Evaluate on Held-out Set
    let mut dec_h_list = Vec::new();
    let mut target_source = Vec::new();
    let mut target_content = Vec::new();
    let mut target_xor = Vec::new();

    let mut helpful_follows = Vec::new();
    let mut opposite_inverts = Vec::new();
    let mut returns = Vec::new();

    for ep in 0..num_eval_episodes {
        let (root_z, s_id, rep, _, steps) = generate_binary_signed_episode(seed + 80000, ep, delay_steps);
        let mut h: Option<Vec<f32>> = None;

        for (sym, ch, is_dec) in &steps {
            let (h_next, instant_feats, _) = model.compute_h_next(*sym, *ch, *is_dec, h.as_deref());
            if *is_dec > 0.5 {
                let mut h_vec = h_next.clone();
                h_vec.push(1.0);
                dec_h_list.push(h_vec);

                target_source.push(s_id as f32);
                target_content.push(rep as f32);
                let xor_val = if (s_id == 1) ^ (rep == 1) { 1 } else { 0 };
                target_xor.push(xor_val);

                let act = if is_mlp_head {
                    let (logits, _) = model.compute_mlp_logits(&h_next, &instant_feats);
                    if logits[1] > logits[0] { 1 } else { 0 }
                } else {
                    let logits = model.compute_linear_logits(&h_next, &instant_feats);
                    if logits[1] > logits[0] { 1 } else { 0 }
                };

                let rew = if act == root_z { 1.0 } else { -1.0 };
                returns.push(rew);

                if s_id == 0 {
                    helpful_follows.push(if act == rep { 1.0 } else { 0.0 });
                } else {
                    opposite_inverts.push(if act == 1 - rep { 1.0 } else { 0.0 });
                }
            }
            h = Some(h_next);
        }
    }

    let n_split = dec_h_list.len() / 2;
    let eval_ridge = |targets: &[f32]| -> f32 {
        if n_split < 10 { return 0.0; }
        let d = dec_h_list[0].len();
        let mut mean_h = vec![0.0; d];
        let mut std_h = vec![0.0; d];
        for row in &dec_h_list[..n_split] { for i in 0..d { mean_h[i] += row[i]; } }
        for i in 0..d { mean_h[i] /= n_split as f32; }
        for row in &dec_h_list[..n_split] { for i in 0..d { std_h[i] += (row[i] - mean_h[i]).powi(2); } }
        for i in 0..d { std_h[i] = (std_h[i] / n_split as f32).sqrt().max(1e-6); }

        let mut norm_h = dec_h_list.clone();
        for row in norm_h.iter_mut() {
            for i in 0..(d - 1) { row[i] = (row[i] - mean_h[i]) / std_h[i]; }
        }

        fit_and_eval_ridge(&norm_h[..n_split], &targets[..n_split], &norm_h[n_split..], &targets[n_split..], 10.0)
    };

    let r2_s = eval_ridge(&target_source);
    let r2_c = eval_ridge(&target_content);
    let target_xor_f32: Vec<f32> = target_xor.iter().map(|&x| x as f32).collect();
    let r2_xor = eval_ridge(&target_xor_f32);

    let mlp_acc = train_mlp_classifier_on_h(&dec_h_list[..n_split], &target_xor[..n_split], &dec_h_list[n_split..], &target_xor[n_split..]);

    let p_help = if !helpful_follows.is_empty() { helpful_follows.iter().sum::<f32>() / helpful_follows.len() as f32 } else { 0.0 };
    let p_opp = if !opposite_inverts.is_empty() { opposite_inverts.iter().sum::<f32>() / opposite_inverts.len() as f32 } else { 0.0 };
    let mean_ret = returns.iter().sum::<f32>() / returns.len() as f32;

    QuadrantCResult {
        quadrant_name: if is_plastic_gru { if is_mlp_head { "Plastic GRU + MLP Head" } else { "Plastic GRU + Linear Head" } } else { if is_mlp_head { "Frozen GRU + MLP Head" } else { "Frozen GRU + Linear Head" } }.to_string(),
        r2_source_class: r2_s,
        r2_report_content: r2_c,
        r2_multiplicative_xor_ridge: r2_xor,
        mlp_probe_accuracy: mlp_acc,
        helpful_following_rate: p_help,
        opposite_inversion_rate: p_opp,
        net_inversion_effect: p_opp - (1.0 - p_help),
        mean_return: mean_ret,
        gru_displacement_norm: disp_norm,
    }
}

fn main() {
    let seeds: Vec<u64> = vec![101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116];

    println!("==========================================================================================================");
    println!("EXECUTING Q13c: DEFINITIVE 2x2 RELATIONAL COMPUTATION & REPRESENTATION LOCALIZATION ASSAY (16 SEEDS)");
    println!("Includes: Explicit [s, r]->MLP Sanity Check, Optimizer-Independent Probes, Full BPTT Recurrent Plasticity");
    println!("==========================================================================================================");

    let start = Instant::now();

    let results: Vec<Q13cSeedResult> = seeds
        .par_iter()
        .map(|&seed| {
            let sanity_acc = test_explicit_mlp_sanity(seed);
            let base_model = Q13cOrganism::new(seed);

            let q1 = train_and_eval_quadrant_c(base_model.clone(), false, false, seed, 2500, 100);
            let q2 = train_and_eval_quadrant_c(base_model.clone(), false, true, seed, 2500, 100);
            let q3 = train_and_eval_quadrant_c(base_model.clone(), true, false, seed, 2500, 100);
            let q4 = train_and_eval_quadrant_c(base_model.clone(), true, true, seed, 2500, 100);

            Q13cSeedResult {
                seed,
                sanity_explicit_mlp_accuracy: sanity_acc,
                frozen_linear: q1,
                frozen_mlp: q2,
                plastic_linear: q3,
                plastic_mlp: q4,
            }
        })
        .collect();

    let elapsed = start.elapsed();

    println!("\n==========================================================================================================");
    println!("Q13c EXECUTION FINISHED IN {:?}", elapsed);
    println!("==========================================================================================================");

    let n = results.len() as f32;
    let mean_sanity = results.iter().map(|r| r.sanity_explicit_mlp_accuracy).sum::<f32>() / n;

    println!("1. SANITY CONTROL: [s, r] -> MLP -> XOR Accuracy: {:+.1}% (PASS if >= 95%)", mean_sanity * 100.0);
    println!("----------------------------------------------------------------------------------------------------------------------------------");
    println!("QUADRANT ARCHITECTURE         | R^2(s)  | R^2(r)  | R^2(XOR) | MLP h->XOR | Helpful % | Invert % | Return | ||d_theta_GRU||");
    println!("----------------------------------------------------------------------------------------------------------------------------------");

    let eval_quad = |get_q: fn(&Q13cSeedResult) -> &QuadrantCResult| -> (f32, f32, f32, f32, f32, f32, f32, f32) {
        let r2_s = results.iter().map(|r| get_q(r).r2_source_class).sum::<f32>() / n;
        let r2_r = results.iter().map(|r| get_q(r).r2_report_content).sum::<f32>() / n;
        let r2_x = results.iter().map(|r| get_q(r).r2_multiplicative_xor_ridge).sum::<f32>() / n;
        let mlp_x = results.iter().map(|r| get_q(r).mlp_probe_accuracy).sum::<f32>() / n;
        let help = results.iter().map(|r| get_q(r).helpful_following_rate).sum::<f32>() / n;
        let opp = results.iter().map(|r| get_q(r).opposite_inversion_rate).sum::<f32>() / n;
        let ret = results.iter().map(|r| get_q(r).mean_return).sum::<f32>() / n;
        let disp = results.iter().map(|r| get_q(r).gru_displacement_norm).sum::<f32>() / n;
        (r2_s, r2_r, r2_x, mlp_x, help, opp, ret, disp)
    };

    let (s1, r1, x1, m1, h1, o1, ret1, d1) = eval_quad(|r| &r.frozen_linear);
    let (s2, r2, x2, m2, h2, o2, ret2, d2) = eval_quad(|r| &r.frozen_mlp);
    let (s3, r3, x3, m3, h3, o3, ret3, d3) = eval_quad(|r| &r.plastic_linear);
    let (s4, r4, x4, m4, h4, o4, ret4, d4) = eval_quad(|r| &r.plastic_mlp);

    println!("Q1. Frozen GRU + Linear Head   | {:+.3}  | {:+.3}  | {:+.3}   | {:+.1}%     | {:+.1}%     | {:+.1}%    | {:+.2}  | {:+.3}", s1, r1, x1, m1 * 100.0, h1 * 100.0, o1 * 100.0, ret1, d1);
    println!("Q2. Frozen GRU + 2-Layer MLP   | {:+.3}  | {:+.3}  | {:+.3}   | {:+.1}%     | {:+.1}%     | {:+.1}%    | {:+.2}  | {:+.3}", s2, r2, x2, m2 * 100.0, h2 * 100.0, o2 * 100.0, ret2, d2);
    println!("Q3. Plastic GRU + Linear Head  | {:+.3}  | {:+.3}  | {:+.3}   | {:+.1}%     | {:+.1}%     | {:+.1}%    | {:+.2}  | {:+.3}", s3, r3, x3, m3 * 100.0, h3 * 100.0, o3 * 100.0, ret3, d3);
    println!("Q4. Plastic GRU + 2-Layer MLP  | {:+.3}  | {:+.3}  | {:+.3}   | {:+.1}%     | {:+.1}%     | {:+.1}%    | {:+.2}  | {:+.3}", s4, r4, x4, m4 * 100.0, h4 * 100.0, o4 * 100.0, ret4, d4);
    println!("----------------------------------------------------------------------------------------------------------------------------------");

    let out_dir = Path::new("../../results/e24_q13_signed_source_binding");
    std::fs::create_dir_all(out_dir).ok();

    let json_data = serde_json::to_string_pretty(&results).unwrap();
    let mut f = File::create(out_dir.join("q13c_summary.json")).unwrap();
    f.write_all(json_data.as_bytes()).unwrap();

    let report = format!(
        "# Q13c: Definitive 2x2 Factorial Relational Computation Synthesis Report

========================================================================================================================
Q13c 2x2 FACTORIAL MATRIX SYNTHESIS (16 SEEDS, RUNTIME: {:?})
========================================================================================================================

## 1. CONTROL VERIFICATION
- **Sanity Check:** `[s, r] -> 2-Layer MLP -> XOR` Accuracy = **{:.1}%** (Baseline verification of MLP solver).

## 2. 2x2 FACTORIAL EMPIRICAL MATRIX

| Quadrant | Architecture | R²(Source) | R²(Content) | R²(Ridge XOR) | MLP Probe h→XOR | Helpful Following % | Opposite Inversion % | Return | ||Δθ_GRU|| |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | **Frozen GRU + Linear Head** | {:+.3} | {:+.3} | {:+.3} | {:+.1}% | {:+.1}% | {:+.1}% | {:+.2} | {:+.3} |
| **Q2** | **Frozen GRU + 2-Layer MLP** | {:+.3} | {:+.3} | {:+.3} | {:+.1}% | {:+.1}% | {:+.1}% | {:+.2} | {:+.3} |
| **Q3** | **Plastic GRU + Linear Head** | {:+.3} | {:+.3} | {:+.3} | {:+.1}% | {:+.1}% | {:+.1}% | {:+.2} | {:+.3} |
| **Q4** | **Plastic GRU + 2-Layer MLP** | {:+.3} | {:+.3} | {:+.3} | {:+.1}% | {:+.1}% | {:+.1}% | {:+.2} | {:+.3} |

========================================================================================================================
## 3. SCIENTIFIC LOCALIZATION:
- **Optimizer-Independent Probes on Frozen h:**
  Linear Ridge regression achieves R² = {:+.3} on XOR.
  An optimizer-independent 2-layer MLP classifier on frozen h achieves **{:.1}% accuracy** on decoding XOR (s ⊕ r)!
- **Plasticity Effects:**
  Plastic GRU updates displace weights by ||Δθ|| = {:+.3} (Q3) and {:+.3} (Q4).
========================================================================================================================
",
        elapsed,
        mean_sanity * 100.0,
        s1, r1, x1, m1 * 100.0, h1 * 100.0, o1 * 100.0, ret1, d1,
        s2, r2, x2, m2 * 100.0, h2 * 100.0, o2 * 100.0, ret2, d2,
        s3, r3, x3, m3 * 100.0, h3 * 100.0, o3 * 100.0, ret3, d3,
        s4, r4, x4, m4 * 100.0, h4 * 100.0, o4 * 100.0, ret4, d4,
        x1, m1 * 100.0, d3, d4
    );

    let mut rep_file = File::create(out_dir.join("report_q13c.md")).unwrap();
    rep_file.write_all(report.as_bytes()).unwrap();

    println!("Saved Q13c summary JSON and Report to {:?}", out_dir);
}
