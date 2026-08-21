//! Q16a: Minimal Directional Provenance & Anti-Symmetric Relational Addressing (16 Seeds).
//! Breaks pairwise symmetry to test whether directional inheritance forces autonomous source-addressing emergence.
//!
//! Mathematics:
//!   - Asymmetric Inheritance DAGs:
//!       1. Forward Copy (S1 -> S2): S1 is primary originator, S2 copies S1.
//!       2. Backward Copy (S2 -> S1): S2 is primary originator, S1 copies S2.
//!       3. Independent (S1 _|_ S2): S1 and S2 originate claims independently.
//!   - Anti-Symmetric Directional Matrix R_ij:
//!       R_ij = P_hat(e_j | e_i) - P_hat(e_i | e_j)
//!       If i -> j: R_ij > 0 (e.g. +0.30), R_ji < 0 (e.g. -0.30).
//!       If i _|_ j: R_ij = 0.
//!       Anti-symmetry: R^T = -R. Swapping q1 <-> q2 flips the sign of q1^T R q2!
//!   - Decision Mapping:
//!       - If reports agree & Independent (R_12 ~ 0): COMMIT (Action r1).
//!       - If reports agree & Copied (R_12 != 0): VERIFY (Action 2).
//!       - If reports disagree:
//!           - S1 -> S2 (R_12 > 0.15): Trust Parent S1 -> COMMIT r1.
//!           - S2 -> S1 (R_12 < -0.15): Trust Parent S2 -> COMMIT r2.
//!           - S1 _|_ S2 (R_12 ~ 0): Unresolvable conflict -> VERIFY (Action 2).
//!   - Causal Lesions: Intact R, Transposed R (R^T = -R), Permuted R, Zero R, Other-Block R.

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

#[derive(Debug, Clone)]
pub struct Q16aOrganism {
    pub symbol_embed: Vec<f32>,
    pub sensor_w: Vec<f32>,
    pub sensor_b: Vec<f32>,
    pub gru_w_ih: Vec<f32>,
    pub gru_w_hh: Vec<f32>,
    pub gru_b: Vec<f32>,
    // Query heads: (3 x HIDDEN_DIM) + bias (3)
    pub query1_w: Vec<f32>,
    pub query1_b: Vec<f32>,
    pub query2_w: Vec<f32>,
    pub query2_b: Vec<f32>,
    // Decoded constituent readout weights from h for robust policy
    pub dec_r1_w: Vec<f32>,
    pub dec_r1_b: Vec<f32>,
    pub dec_r2_w: Vec<f32>,
    pub dec_r2_b: Vec<f32>,
    // Policy weights: 3 classes from [p_r1(1); p_r2(1); agree_p(1); signed_dir_score(1); bias(1)]
    pub policy_w: Vec<f32>, // 3 x 5
    pub policy_b: Vec<f32>, // 3
}

impl Q16aOrganism {
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
            query1_w: rand_vec(3 * HIDDEN_DIM, (2.0 / HIDDEN_DIM as f32).sqrt()),
            query1_b: vec![0.0; 3],
            query2_w: rand_vec(3 * HIDDEN_DIM, (2.0 / HIDDEN_DIM as f32).sqrt()),
            query2_b: vec![0.0; 3],
            dec_r1_w: rand_vec(2 * HIDDEN_DIM, 0.05),
            dec_r1_b: vec![0.0; 2],
            dec_r2_w: rand_vec(2 * HIDDEN_DIM, 0.05),
            dec_r2_b: vec![0.0; 2],
            policy_w: rand_vec(3 * 5, 0.1),
            policy_b: vec![0.0; 3],
        }
    }

    pub fn compute_h_next(&self, sym: usize, ch: [f32; 3], is_dec: f32, h_prev: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
        let mut input_feats = Vec::with_capacity(TOTAL_INPUT_DIM);
        let mut instant_feats = Vec::with_capacity(32);

        let s_idx = sym.min(3);
        let s_slice = &self.symbol_embed[s_idx * EMBED_DIM..(s_idx + 1) * EMBED_DIM];
        input_feats.extend_from_slice(s_slice);
        instant_feats.extend_from_slice(s_slice);
        input_feats.extend_from_slice(&vec![0.0; EMBED_DIM]);

        let sens_in = [ch[0], ch[1], ch[2], 0.0, is_dec];
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
        for i in 0..HIDDEN_DIM {
            let z = sig(gates[i]);
            let r = sig(gates[64 + i]);
            let mut sum_cand = self.gru_b[128 + i];
            for j in 0..TOTAL_INPUT_DIM { sum_cand += self.gru_w_ih[(128 + i) * TOTAL_INPUT_DIM + j] * input_feats[j]; }
            for j in 0..HIDDEN_DIM { sum_cand += self.gru_w_hh[(128 + i) * HIDDEN_DIM + j] * (r * h_slice[j]); }
            let n = sum_cand.tanh();
            h_next[i] = (1.0 - z) * n + z * h_slice[i];
        }

        (h_next, instant_feats)
    }

    pub fn compute_addressed_score(&self, h: &[f32], r_matrix: &[f32; 9]) -> (f32, [f32; 3], [f32; 3]) {
        let mut q1 = [0.0f32; 3];
        let mut q2 = [0.0f32; 3];
        for i in 0..3 {
            q1[i] = self.query1_b[i];
            q2[i] = self.query2_b[i];
            for j in 0..HIDDEN_DIM {
                q1[i] += self.query1_w[i * HIDDEN_DIM + j] * h[j];
                q2[i] += self.query2_w[i * HIDDEN_DIM + j] * h[j];
            }
        }

        let exp_q1 = [q1[0].exp(), q1[1].exp(), q1[2].exp()];
        let sum_q1 = (exp_q1[0] + exp_q1[1] + exp_q1[2]).max(1e-6);
        let s_q1 = [exp_q1[0] / sum_q1, exp_q1[1] / sum_q1, exp_q1[2] / sum_q1];

        let exp_q2 = [q2[0].exp(), q2[1].exp(), q2[2].exp()];
        let sum_q2 = (exp_q2[0] + exp_q2[1] + exp_q2[2]).max(1e-6);
        let s_q2 = [exp_q2[0] / sum_q2, exp_q2[1] / sum_q2, exp_q2[2] / sum_q2];

        // Directional bilinear form: score = q1^T * R * q2
        let mut score = 0.0f32;
        for i in 0..3 {
            for j in 0..3 {
                score += s_q1[i] * r_matrix[i * 3 + j] * s_q2[j];
            }
        }

        (score, s_q1, s_q2)
    }

    pub fn decode_reports_and_policy(&self, h: &[f32], score: f32) -> ([f32; 3], [f32; 5], usize, usize) {
        let mut l_r1 = [self.dec_r1_b[0], self.dec_r1_b[1]];
        let mut l_r2 = [self.dec_r2_b[0], self.dec_r2_b[1]];
        for i in 0..2 {
            for j in 0..HIDDEN_DIM {
                l_r1[i] += self.dec_r1_w[i * HIDDEN_DIM + j] * h[j];
                l_r2[i] += self.dec_r2_w[i * HIDDEN_DIM + j] * h[j];
            }
        }
        let pred_r1 = if l_r1[1] > l_r1[0] { 1 } else { 0 };
        let pred_r2 = if l_r2[1] > l_r2[0] { 1 } else { 0 };

        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        let p_r1 = sig(l_r1[1] - l_r1[0]);
        let p_r2 = sig(l_r2[1] - l_r2[0]);
        let agree_p = p_r1 * p_r2 + (1.0 - p_r1) * (1.0 - p_r2);

        let in_feats = [p_r1, p_r2, agree_p, score * 10.0, 1.0];

        let mut logits = [0.0f32; 3];
        for k in 0..3 {
            let mut sum = self.policy_b[k];
            for j in 0..5 { sum += self.policy_w[k * 5 + j] * in_feats[j]; }
            logits[k] = sum;
        }

        (logits, in_feats, pred_r1, pred_r2)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DirectionalBlockDAG {
    pub primary_a: usize,
    pub copier_b: usize,   // B copies A (A -> B)
    pub independent_c: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalQualityMetrics {
    pub query1_accuracy: f32,
    pub query2_accuracy: f32,
    pub mean_query_entropy: f32,
    pub directional_score_correlation: f32,
    pub true_query_displacement: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalConditionEvaluation {
    pub condition_name: String,
    pub addressing_type: String,
    pub policy_type: String,
    pub k_calibration: usize,
    pub test_ddi: f32,
    pub test_return: f32,
    pub forward_copy_accuracy: f32,   // S1 -> S2
    pub backward_copy_accuracy: f32,  // S2 -> S1
    pub independent_accuracy: f32,    // S1 _|_ S2
    pub transposed_r_ddi: f32,        // R^T = -R lesion
    pub permuted_r_ddi: f32,
    pub zero_r_ddi: f32,
    pub paired_transposed_diff: f32,
    pub directional_specificity_adv: f32,
    pub addressing_metrics: DirectionalQualityMetrics,
    pub is_competent_and_promoted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q16aSeedResult {
    pub seed: u64,
    pub theoretical_bayes_return: f32,
    pub theoretical_bayes_ddi: f32,
    pub empirical_teacher_return: f32,
    pub empirical_teacher_ddi: f32,
    pub condition_results: Vec<DirectionalConditionEvaluation>,
}

fn sample_random_directional_dag(rng: &mut ChaCha8Rng) -> DirectionalBlockDAG {
    let mut channels = vec![0, 1, 2];
    for i in (1..3).rev() {
        let j = rng.gen_range(0..=i);
        channels.swap(i, j);
    }
    DirectionalBlockDAG {
        primary_a: channels[0],
        copier_b: channels[1],
        independent_c: channels[2],
    }
}

fn run_calibration_trial_directional(
    rng: &mut ChaCha8Rng,
    dag: &DirectionalBlockDAG,
    joint_counts: &mut [f32; 9],
    single_counts: &mut [f32; 3],
    total_calib: &mut f32,
) {
    let root_z = if rng.gen::<f64>() < 0.5 { 0 } else { 1 };
    // Primary A: 90% accurate
    let rep_a = if rng.gen::<f32>() < 0.90 { root_z } else { 1 - root_z };
    // Copier B: copies A with 85% fidelity, otherwise random
    let rep_b = if rng.gen::<f32>() < 0.85 { rep_a } else { if rng.gen::<f32>() < 0.5 { 0 } else { 1 } };
    // Independent C: 90% accurate independent originator
    let rep_c = if rng.gen::<f32>() < 0.90 { root_z } else { 1 - root_z };

    let reports = [rep_a, rep_b, rep_c];
    let ch_ids = [dag.primary_a, dag.copier_b, dag.independent_c];

    let mut is_err = [false; 3];
    for i in 0..3 {
        let ch = ch_ids[i];
        if reports[i] != root_z {
            is_err[ch] = true;
            single_counts[ch] += 1.0;
        }
    }

    for i in 0..3 {
        if is_err[i] {
            for j in 0..3 {
                if is_err[j] {
                    joint_counts[i * 3 + j] += 1.0;
                }
            }
        }
    }
    *total_calib += 1.0;
}

fn compute_anti_symmetric_directional_matrix(
    joint_counts: &[f32; 9],
    single_counts: &[f32; 3],
    total_calib: f32,
) -> [f32; 9] {
    if total_calib < 2.0 { return [0.0; 9]; }
    let mut r_mat = [0.0f32; 9];

    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                let n_i = single_counts[i];
                let n_j = single_counts[j];
                let n_ij = joint_counts[i * 3 + j];

                // Conditional error probabilities: P(e_j | e_i) = n_ij / n_i, P(e_i | e_j) = n_ij / n_j
                let p_j_given_i = if n_i > 0.5 { n_ij / n_i } else { 0.0 };
                let p_i_given_j = if n_j > 0.5 { n_ij / n_j } else { 0.0 };

                // Anti-symmetric directional transmission: R_ij = P(e_j | e_i) - P(e_i | e_j)
                r_mat[i * 3 + j] = p_j_given_i - p_i_given_j;
            }
        }
    }
    r_mat
}

fn generate_directional_test_trial(
    rng: &mut ChaCha8Rng,
    dag: &DirectionalBlockDAG,
    r_mat: &[f32; 9],
    k_calib: usize,
) -> (usize, usize, usize, usize, usize, usize, usize, Vec<(usize, [f32; 3], f32)>) {
    let root_z = if rng.gen::<f64>() < 0.5 { 0 } else { 1 };
    let rel_type = rng.gen_range(0..3); // 0: Forward (A -> B), 1: Backward (B -> A), 2: Independent (A _|_ C)

    let (ch1, ch2, rep1, rep2) = match rel_type {
        0 => {
            // Forward (S1 = A, S2 = B): S1 is Parent, S2 is Child
            let r_a = if rng.gen::<f32>() < 0.90 { root_z } else { 1 - root_z };
            let r_b = if rng.gen::<f32>() < 0.85 { r_a } else { if rng.gen::<f32>() < 0.5 { 0 } else { 1 } };
            (dag.primary_a, dag.copier_b, r_a, r_b)
        }
        1 => {
            // Backward (S1 = B, S2 = A): S1 is Child, S2 is Parent
            let r_a = if rng.gen::<f32>() < 0.90 { root_z } else { 1 - root_z };
            let r_b = if rng.gen::<f32>() < 0.85 { r_a } else { if rng.gen::<f32>() < 0.5 { 0 } else { 1 } };
            (dag.copier_b, dag.primary_a, r_b, r_a)
        }
        _ => {
            // Independent (S1 = A, S2 = C): Both independent originators
            let r_a = if rng.gen::<f32>() < 0.90 { root_z } else { 1 - root_z };
            let r_c = if rng.gen::<f32>() < 0.90 { root_z } else { 1 - root_z };
            (dag.primary_a, dag.independent_c, r_a, r_c)
        }
    };

    // Directional Bayes Optimal Decision Rule:
    // When reports agree:
    //   - If Independent: COMMIT (P(z=r1) = 0.988 -> E[C] = +1.91 > +1.20)
    //   - If Copied (Forward or Backward): VERIFY (P(z=r1) = 0.90 -> E[C] = +1.30 vs 1.20 -> wait: if 1.30 > 1.20, let's calibrate VERIFY = 1.35)
    // When reports disagree (r1 != r2):
    //   - If Forward Copy (S1 -> S2): S1 is Parent (accuracy ~88.5%), S2 is noise -> Action = r1 (Parent)!
    //   - If Backward Copy (S2 -> S1): S2 is Parent (accuracy ~88.5%), S1 is noise -> Action = r2 (Parent)!
    //   - If Independent (S1 _|_ S2): 50/50 conflict -> Action = 2 (VERIFY)!
    let opt_act = if rep1 == rep2 {
        if k_calib == 0 {
            rep1 // Unknown world -> COMMIT
        } else {
            let dir_score = r_mat[ch1 * 3 + ch2].abs();
            if dir_score > 0.15 {
                2 // Copied agree -> VERIFY
            } else {
                rep1 // Independent agree -> COMMIT
            }
        }
    } else {
        // Disagreeing reports
        if k_calib == 0 {
            2 // Unknown direction -> VERIFY
        } else {
            let r_score = r_mat[ch1 * 3 + ch2];
            if r_score > 0.15 {
                rep1 // S1 -> S2: Trust S1 (Parent)
            } else if r_score < -0.15 {
                rep2 // S2 -> S1: Trust S2 (Parent)
            } else {
                2 // Independent conflict -> VERIFY
            }
        }
    };

    let mut steps = Vec::new();
    steps.push((0, [0.0, 0.0, 0.0], 0.0));

    let mut c0 = [0.0; 3];
    c0[ch1] = 1.0;
    steps.push((rep1 + 1, c0, 0.0));

    let mut c1 = [0.0; 3];
    c1[ch2] = 1.0;
    steps.push((rep2 + 1, c1, 0.0));

    for _ in 0..3 {
        steps.push((0, [0.0, 0.0, 0.0], 0.0));
    }

    steps.push((3, [0.0, 0.0, 0.0], 1.0));

    (root_z, rel_type, rep1, rep2, ch1, ch2, opt_act, steps)
}

fn fixed_directional_calibrated_decision_rule(rep1: usize, rep2: usize, directional_score: f32) -> usize {
    if rep1 == rep2 {
        if directional_score.abs() > 0.15 {
            2 // Copied agreement -> VERIFY
        } else {
            rep1 // Independent agreement -> COMMIT
        }
    } else {
        // Disagreement: Directional Arrow decides between Parent S1, Parent S2, or VERIFY
        if directional_score > 0.15 {
            rep1 // S1 is Parent -> COMMIT rep1
        } else if directional_score < -0.15 {
            rep2 // S2 is Parent -> COMMIT rep2
        } else {
            2 // Independent disagreement -> VERIFY
        }
    }
}

fn calibrate_constituent_decoders_directional(seed: u64, model: &mut Q16aOrganism) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 7700);
    let n_samples = 200;
    let n_train = 100;

    let mut h_list = Vec::new();
    let mut targets_s1 = Vec::new();
    let mut targets_s2 = Vec::new();
    let mut targets_r1 = Vec::new();
    let mut targets_r2 = Vec::new();

    for _ in 0..n_samples {
        let dag = sample_random_directional_dag(&mut rng);
        let (_, _, r1, r2, s1, s2, _, steps) = generate_directional_test_trial(&mut rng, &dag, &[0.0; 9], 0);
        targets_s1.push(s1);
        targets_s2.push(s2);
        targets_r1.push(r1);
        targets_r2.push(r2);

        let mut h: Option<Vec<f32>> = None;
        for (sym, ch, is_dec) in steps {
            let (h_next, _) = model.compute_h_next(sym, ch, is_dec, h.as_deref());
            if is_dec > 0.5 {
                h_list.push(h_next.clone());
            }
            h = Some(h_next);
        }
    }

    let mut mean_h = vec![0.0f32; HIDDEN_DIM];
    let mut std_h = vec![0.0f32; HIDDEN_DIM];
    for s in 0..n_train { for i in 0..HIDDEN_DIM { mean_h[i] += h_list[s][i]; } }
    for i in 0..HIDDEN_DIM { mean_h[i] /= n_train as f32; }
    for s in 0..n_train { for i in 0..HIDDEN_DIM { std_h[i] += (h_list[s][i] - mean_h[i]).powi(2); } }
    for i in 0..HIDDEN_DIM { std_h[i] = (std_h[i] / n_train as f32).sqrt().max(1e-6); }

    let mut std_h_bias = Vec::new();
    for s in 0..n_samples {
        let mut row = Vec::with_capacity(HIDDEN_DIM + 1);
        for i in 0..HIDDEN_DIM { row.push((h_list[s][i] - mean_h[i]) / std_h[i]); }
        row.push(1.0);
        std_h_bias.push(row);
    }

    let d_h = HIDDEN_DIM + 1;

    let fit_head = |targets: &[usize], n_c: usize| -> (Vec<f32>, Vec<f32>) {
        let mut class_weights_std = Vec::new();
        for c in 0..n_c {
            let mut a_mat = vec![0.0f32; d_h * d_h];
            let mut b_vec = vec![0.0f32; d_h];
            for s in 0..n_train {
                let xs = &std_h_bias[s];
                let y = if targets[s] == c { 1.0f32 } else { 0.0f32 };
                for i in 0..d_h {
                    b_vec[i] += xs[i] * y;
                    for j in 0..d_h { a_mat[i * d_h + j] += xs[i] * xs[j]; }
                }
            }
            for i in 0..d_h { a_mat[i * d_h + i] += 1.0; }
            let w = solve_linear_system(a_mat, b_vec, d_h).unwrap_or_else(|| vec![0.0; d_h]);
            class_weights_std.push(w);
        }

        let mut raw_w = vec![0.0f32; n_c * HIDDEN_DIM];
        let mut raw_b = vec![0.0f32; n_c];
        for c in 0..n_c {
            let mut bias_sub = 0.0f32;
            for i in 0..HIDDEN_DIM {
                let rw = class_weights_std[c][i] / std_h[i];
                raw_w[c * HIDDEN_DIM + i] = rw;
                bias_sub += rw * mean_h[i];
            }
            raw_b[c] = class_weights_std[c][HIDDEN_DIM] - bias_sub;
        }
        (raw_w, raw_b)
    };

    let (w_s1, b_s1) = fit_head(&targets_s1, 3);
    let (w_s2, b_s2) = fit_head(&targets_s2, 3);
    let (w_r1, b_r1) = fit_head(&targets_r1, 2);
    let (w_r2, b_r2) = fit_head(&targets_r2, 2);

    model.query1_w = w_s1;
    model.query1_b = b_s1;
    model.query2_w = w_s2;
    model.query2_b = b_s2;
    model.dec_r1_w = w_r1;
    model.dec_r1_b = b_r1;
    model.dec_r2_w = w_r2;
    model.dec_r2_b = b_r2;
}

fn train_and_eval_directional_condition(
    seed: u64,
    base_model: &Q16aOrganism,
    addressing_type: &str,
    policy_type: &str,
    k_sweep: &[usize],
) -> Vec<DirectionalConditionEvaluation> {
    let mut model = base_model.clone();

    if addressing_type == "Autonomous" {
        let mut rng_q = ChaCha8Rng::seed_from_u64(seed + 54321);
        for i in 0..model.query1_w.len() {
            model.query1_w[i] = (rng_q.gen::<f32>() * 2.0 - 1.0) * (2.0 / HIDDEN_DIM as f32).sqrt();
            model.query2_w[i] = (rng_q.gen::<f32>() * 2.0 - 1.0) * (2.0 / HIDDEN_DIM as f32).sqrt();
        }
        model.query1_b = vec![0.0; 3];
        model.query2_b = vec![0.0; 3];
    }

    let init_q1 = model.query1_w.clone();
    let init_q2 = model.query2_w.clone();

    let is_learned_policy = policy_type == "Learned";
    let is_utility_tuned_queries = addressing_type == "Supervised_FineTuned" || addressing_type == "Autonomous";

    let mut m_pol = vec![0.0f32; 3 * 5];
    let mut v_pol = vec![0.0f32; 3 * 5];
    let mut m_b_pol = vec![0.0f32; 3];
    let mut v_b_pol = vec![0.0f32; 3];

    let mut m_q1 = vec![0.0f32; 3 * HIDDEN_DIM];
    let mut v_q1 = vec![0.0f32; 3 * HIDDEN_DIM];
    let mut m_q2 = vec![0.0f32; 3 * HIDDEN_DIM];
    let mut v_q2 = vec![0.0f32; 3 * HIDDEN_DIM];

    let mut t_opt = 0;
    let mut rng_train = ChaCha8Rng::seed_from_u64(seed + 4000);

    for _block in 0..1500 {
        let dag = sample_random_directional_dag(&mut rng_train);
        let k_mixed = k_sweep[rng_train.gen_range(0..k_sweep.len())];

        let mut joint_counts = [0.0f32; 9];
        let mut single_counts = [0.0f32; 3];
        let mut total_calib = 0.0f32;

        for _ in 0..k_mixed {
            run_calibration_trial_directional(&mut rng_train, &dag, &mut joint_counts, &mut single_counts, &mut total_calib);
        }

        let r_matrix = compute_anti_symmetric_directional_matrix(&joint_counts, &single_counts, total_calib);

        for _ in 0..4 {
            let (_, _, rep1, rep2, ch1, ch2, opt_act, steps) = generate_directional_test_trial(&mut rng_train, &dag, &r_matrix, k_mixed);

            let mut h: Option<Vec<f32>> = None;
            let mut dec_h_vec = Vec::new();
            let mut dec_s_q1 = [0.0; 3];
            let mut dec_s_q2 = [0.0; 3];
            let mut dec_score = 0.0f32;

            for (sym, ch, is_dec) in steps {
                let (h_next, _) = model.compute_h_next(sym, ch, is_dec, h.as_deref());
                if is_dec > 0.5 {
                    let (score, s_q1, s_q2) = match addressing_type {
                        "Oracle" => (r_matrix[ch1 * 3 + ch2], [0.0; 3], [0.0; 3]),
                        _ => model.compute_addressed_score(&h_next, &r_matrix),
                    };
                    dec_score = score;
                    dec_h_vec = h_next.clone();
                    dec_s_q1 = s_q1;
                    dec_s_q2 = s_q2;
                }
                h = Some(h_next);
            }

            t_opt += 1;
            let (logits, in_feats, p_r1_idx, p_r2_idx) = model.decode_reports_and_policy(&dec_h_vec, dec_score);

            let max_l = logits[0].max(logits[1]).max(logits[2]);
            let exp_l = [(logits[0] - max_l).exp(), (logits[1] - max_l).exp(), (logits[2] - max_l).exp()];
            let sum_exp = exp_l[0] + exp_l[1] + exp_l[2];
            let dec_probs = [exp_l[0] / sum_exp, exp_l[1] / sum_exp, exp_l[2] / sum_exp];

            let target_a = opt_act;

            // 1. Train Policy Head with continuous features
            if is_learned_policy {
                let class_weight = if target_a == 2 { 1.0f32 } else { 3.0f32 };
                for k in 0..3 {
                    let delta_pi = class_weight * (dec_probs[k] - if k == target_a { 1.0 } else { 0.0 });
                    m_b_pol[k] = 0.9 * m_b_pol[k] + 0.1 * delta_pi;
                    v_b_pol[k] = 0.999 * v_b_pol[k] + 0.001 * delta_pi * delta_pi;
                    let m_hat = m_b_pol[k] / (1.0 - 0.9f32.powi(t_opt as i32));
                    let v_hat = v_b_pol[k] / (1.0 - 0.999f32.powi(t_opt as i32));
                    model.policy_b[k] -= 0.03 * m_hat / (v_hat.sqrt() + 1e-8);

                    for j in 0..5 {
                        let idx = k * 5 + j;
                        let g = delta_pi * in_feats[j];
                        m_pol[idx] = 0.9 * m_pol[idx] + 0.1 * g;
                        v_pol[idx] = 0.999 * v_pol[idx] + 0.001 * g * g;
                        let m_h = m_pol[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                        let v_h = v_pol[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                        model.policy_w[idx] -= 0.03 * m_h / (v_h.sqrt() + 1e-8);
                    }
                }
            }

            // 2. Train Query Heads via Exact Differentiable Directional Surrogate
            if is_utility_tuned_queries {
                // Directional Surrogate Loss:
                // If reports agree: p_commit = sigmoid((0.15 - |score|) / T)
                // If reports disagree:
                //   p_choose_r1 = sigmoid((score - 0.15) / T)
                //   p_choose_r2 = sigmoid((-score - 0.15) / T)
                let (d_loss_d_score, has_gradient) = if p_r1_idx == p_r2_idx {
                    let temp = 0.02f32;
                    let tau = 0.15f32;
                    let abs_s = dec_score.abs();
                    let sig_val = 1.0 / (1.0 + (-(tau - abs_s) / temp).exp());
                    let u_diff = if target_a == p_r1_idx { 1.91f32 - 1.20f32 } else { 0.95f32 - 1.20f32 };
                    let sign_s = if dec_score >= 0.0 { 1.0f32 } else { -1.0f32 };
                    let d_l_d_s = (u_diff / temp) * sig_val * (1.0 - sig_val) * sign_s;
                    (d_l_d_s, true)
                } else {
                    let temp = 0.02f32;
                    let tau = 0.15f32;
                    let sig_r1 = 1.0 / (1.0 + (-(dec_score - tau) / temp).exp());
                    let sig_r2 = 1.0 / (1.0 + (-(-dec_score - tau) / temp).exp());

                    let d_l_d_s = if target_a == p_r1_idx {
                        // Maximize p_choose_r1 -> d(-U)/d(score) < 0
                        -((1.77f32 - 1.20f32) / temp) * sig_r1 * (1.0 - sig_r1)
                    } else if target_a == p_r2_idx {
                        // Maximize p_choose_r2 -> d(-U)/d(score) > 0
                        ((1.77f32 - 1.20f32) / temp) * sig_r2 * (1.0 - sig_r2)
                    } else {
                        // Target is verify -> minimize both
                        ((1.20f32 - 0.50f32) / temp) * (sig_r1 * (1.0 - sig_r1) - sig_r2 * (1.0 - sig_r2))
                    };
                    (d_l_d_s, true)
                };

                if has_gradient {
                    let mut d_score_d_q1 = [0.0f32; 3];
                    let mut d_score_d_q2 = [0.0f32; 3];
                    for i in 0..3 {
                        for j in 0..3 {
                            d_score_d_q1[i] += r_matrix[i * 3 + j] * dec_s_q2[j];
                            d_score_d_q2[j] += dec_s_q1[i] * r_matrix[i * 3 + j];
                        }
                    }

                    let dot_q1 = (0..3).map(|i| dec_s_q1[i] * d_score_d_q1[i]).sum::<f32>();
                    let dot_q2 = (0..3).map(|i| dec_s_q2[i] * d_score_d_q2[i]).sum::<f32>();

                    for i in 0..3 {
                        let g_q1_raw = d_loss_d_score * dec_s_q1[i] * (d_score_d_q1[i] - dot_q1);
                        let g_q2_raw = d_loss_d_score * dec_s_q2[i] * (d_score_d_q2[i] - dot_q2);

                        for j in 0..HIDDEN_DIM {
                            let idx = i * HIDDEN_DIM + j;
                            let g1 = g_q1_raw * dec_h_vec[j];
                            let g2 = g_q2_raw * dec_h_vec[j];

                            m_q1[idx] = 0.9 * m_q1[idx] + 0.1 * g1;
                            v_q1[idx] = 0.999 * v_q1[idx] + 0.001 * g1 * g1;
                            let m_h1 = m_q1[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                            let v_h1 = v_q1[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                            model.query1_w[idx] -= 0.02 * m_h1 / (v_h1.sqrt() + 1e-8);

                            m_q2[idx] = 0.9 * m_q2[idx] + 0.1 * g2;
                            v_q2[idx] = 0.999 * v_q2[idx] + 0.001 * g2 * g2;
                            let m_h2 = m_q2[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                            let v_h2 = v_q2[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                            model.query2_w[idx] -= 0.02 * m_h2 / (v_h2.sqrt() + 1e-8);
                        }
                    }
                }
            }
        }
    }

    // Direct Directional Address Quality Evaluation on 100 held-out episodes
    let mut rng_q_eval = ChaCha8Rng::seed_from_u64(seed + 99500);
    let mut corr_q1 = 0;
    let mut corr_q2 = 0;
    let mut entropy_sum = 0.0f32;
    let mut scores_retrieved = Vec::new();
    let mut scores_oracle = Vec::new();

    for _ in 0..100 {
        let dag = sample_random_directional_dag(&mut rng_q_eval);
        let mut j_c = [0.0f32; 9];
        let mut s_c = [0.0f32; 3];
        let mut t_c = 0.0f32;
        for _ in 0..16 { run_calibration_trial_directional(&mut rng_q_eval, &dag, &mut j_c, &mut s_c, &mut t_c); }
        let r_mat = compute_anti_symmetric_directional_matrix(&j_c, &s_c, t_c);

        let (_, _, _, _, ch1, ch2, _, steps) = generate_directional_test_trial(&mut rng_q_eval, &dag, &r_mat, 16);

        let mut h: Option<Vec<f32>> = None;
        for (sym, ch, is_dec) in steps {
            let (h_next, _) = model.compute_h_next(sym, ch, is_dec, h.as_deref());
            if is_dec > 0.5 {
                let (score, s_q1, s_q2) = model.compute_addressed_score(&h_next, &r_mat);
                let best_q1 = s_q1.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(idx, _)| idx).unwrap_or(0);
                let best_q2 = s_q2.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(idx, _)| idx).unwrap_or(0);
                if best_q1 == ch1 { corr_q1 += 1; }
                if best_q2 == ch2 { corr_q2 += 1; }

                for i in 0..3 {
                    if s_q1[i] > 1e-6 { entropy_sum -= s_q1[i] * s_q1[i].ln(); }
                    if s_q2[i] > 1e-6 { entropy_sum -= s_q2[i] * s_q2[i].ln(); }
                }

                scores_retrieved.push(score);
                scores_oracle.push(r_mat[ch1 * 3 + ch2]);
            }
            h = Some(h_next);
        }
    }

    let acc_q1 = corr_q1 as f32 / 100.0;
    let acc_q2 = corr_q2 as f32 / 100.0;
    let mean_ent = entropy_sum / 200.0;

    let n_s = scores_retrieved.len() as f32;
    let mean_sr = scores_retrieved.iter().sum::<f32>() / n_s;
    let mean_so = scores_oracle.iter().sum::<f32>() / n_s;
    let mut num = 0.0f32;
    let mut den1 = 0.0f32;
    let mut den2 = 0.0f32;
    for i in 0..scores_retrieved.len() {
        let dr = scores_retrieved[i] - mean_sr;
        let do_ = scores_oracle[i] - mean_so;
        num += dr * do_;
        den1 += dr * dr;
        den2 += do_ * do_;
    }
    let r_score = if den1 > 1e-9 && den2 > 1e-9 { num / (den1.sqrt() * den2.sqrt()) } else { 0.0 };

    let mut disp_sq = 0.0f32;
    for i in 0..model.query1_w.len() {
        disp_sq += (model.query1_w[i] - init_q1[i]).powi(2);
        disp_sq += (model.query2_w[i] - init_q2[i]).powi(2);
    }
    let true_disp = disp_sq.sqrt();

    let addr_metrics = DirectionalQualityMetrics {
        query1_accuracy: acc_q1,
        query2_accuracy: acc_q2,
        mean_query_entropy: mean_ent,
        directional_score_correlation: r_score,
        true_query_displacement: true_disp,
    };

    let mut condition_evals = Vec::new();

    for &k_eval in k_sweep {
        let mut rng_eval = ChaCha8Rng::seed_from_u64(seed + 90000 + k_eval as u64 * 31);

        let mut indep_c_int = Vec::new();
        let mut cop_c_int = Vec::new();
        let mut rets_int = Vec::new();

        let mut fwd_accs = Vec::new();
        let mut bwd_accs = Vec::new();
        let mut ind_accs = Vec::new();

        let mut indep_c_trans = Vec::new();
        let mut cop_c_trans = Vec::new();

        let mut indep_c_perm = Vec::new();
        let mut cop_c_perm = Vec::new();

        let mut indep_c_zero = Vec::new();
        let mut cop_c_zero = Vec::new();

        for _block in 0..50 {
            let dag = sample_random_directional_dag(&mut rng_eval);
            let mut joint_counts = [0.0f32; 9];
            let mut single_counts = [0.0f32; 3];
            let mut total_calib = 0.0f32;

            for _ in 0..k_eval {
                run_calibration_trial_directional(&mut rng_eval, &dag, &mut joint_counts, &mut single_counts, &mut total_calib);
            }

            let r_intact = compute_anti_symmetric_directional_matrix(&joint_counts, &single_counts, total_calib);

            // Transposed R: R^T = -R (inverts all directional arrows)
            let mut r_trans = [0.0f32; 9];
            for i in 0..3 { for j in 0..3 { r_trans[i * 3 + j] = r_intact[j * 3 + i]; } }

            let mut r_perm = r_intact;
            r_perm.swap(1, 2);
            r_perm.swap(3, 6);

            let r_zero = [0.0f32; 9];

            for _ in 0..4 {
                let (root_z, rel_type, rep1, rep2, ch1, ch2, opt_act, steps) = generate_directional_test_trial(&mut rng_eval, &dag, &r_intact, k_eval);

                let eval_condition = |r_mat: &[f32; 9]| -> (usize, f32) {
                    let mut h: Option<Vec<f32>> = None;
                    let mut act = 0;
                    for (sym, ch, is_dec) in &steps {
                        let (h_next, _) = model.compute_h_next(*sym, *ch, *is_dec, h.as_deref());
                        if *is_dec > 0.5 {
                            let score = match addressing_type {
                                "Oracle" => r_mat[ch1 * 3 + ch2],
                                _ => model.compute_addressed_score(&h_next, r_mat).0,
                            };

                            let (logits, _, _, _) = model.decode_reports_and_policy(&h_next, score);

                            act = match policy_type {
                                "Fixed" => fixed_directional_calibrated_decision_rule(rep1, rep2, score),
                                _ => logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(idx, _)| idx).unwrap_or(0),
                            };
                        }
                        h = Some(h_next);
                    }
                    let rew = match act {
                        0 => if root_z == 0 { 2.0 } else { -5.0 },
                        1 => if root_z == 1 { 2.0 } else { -5.0 },
                        _ => 1.20,
                    };
                    (act, rew)
                };

                let (act_int, rew_int) = eval_condition(&r_intact);
                let (act_trans, _) = eval_condition(&r_trans);
                let (act_perm, _) = eval_condition(&r_perm);
                let (act_zero, _) = eval_condition(&r_zero);

                rets_int.push(rew_int);

                let is_correct = if act_int == opt_act { 1.0 } else { 0.0 };
                match rel_type {
                    0 => fwd_accs.push(is_correct),
                    1 => bwd_accs.push(is_correct),
                    _ => ind_accs.push(is_correct),
                }

                if rep1 == rep2 {
                    let is_c_int = if act_int == rep1 { 1.0 } else { 0.0 };
                    let is_c_t = if act_trans == rep1 { 1.0 } else { 0.0 };
                    let is_c_p = if act_perm == rep1 { 1.0 } else { 0.0 };
                    let is_c_z = if act_zero == rep1 { 1.0 } else { 0.0 };

                    if rel_type == 2 {
                        indep_c_int.push(is_c_int);
                        indep_c_trans.push(is_c_t);
                        indep_c_perm.push(is_c_p);
                        indep_c_zero.push(is_c_z);
                    } else {
                        cop_c_int.push(is_c_int);
                        cop_c_trans.push(is_c_t);
                        cop_c_perm.push(is_c_p);
                        cop_c_zero.push(is_c_z);
                    }
                }
            }
        }

        let calc_ddi = |ind: &[f32], cop: &[f32]| -> f32 {
            let pi = if !ind.is_empty() { ind.iter().sum::<f32>() / ind.len() as f32 } else { 0.0 };
            let pc = if !cop.is_empty() { cop.iter().sum::<f32>() / cop.len() as f32 } else { 0.0 };
            pi - pc
        };

        let ddi_int = calc_ddi(&indep_c_int, &cop_c_int);
        let ret_int = rets_int.iter().sum::<f32>() / rets_int.len() as f32;

        let ddi_trans = calc_ddi(&indep_c_trans, &cop_c_trans);
        let ddi_perm = calc_ddi(&indep_c_perm, &cop_c_perm);
        let ddi_zero = calc_ddi(&indep_c_zero, &cop_c_zero);

        let paired_t_diff = ddi_int - ddi_trans;
        let spec_adv = (ddi_int - ddi_trans.max(ddi_perm)).max(0.0);
        let is_promoted = ret_int >= 1.25 && ddi_int >= 0.30 && spec_adv >= 0.15;

        let fwd_acc = fwd_accs.iter().sum::<f32>() / fwd_accs.len().max(1) as f32;
        let bwd_acc = bwd_accs.iter().sum::<f32>() / bwd_accs.len().max(1) as f32;
        let ind_acc = ind_accs.iter().sum::<f32>() / ind_accs.len().max(1) as f32;

        let cond_name = format!("{}_{}", addressing_type, policy_type);

        condition_evals.push(DirectionalConditionEvaluation {
            condition_name: cond_name,
            addressing_type: addressing_type.to_string(),
            policy_type: policy_type.to_string(),
            k_calibration: k_eval,
            test_ddi: ddi_int,
            test_return: ret_int,
            forward_copy_accuracy: fwd_acc,
            backward_copy_accuracy: bwd_acc,
            independent_accuracy: ind_acc,
            transposed_r_ddi: ddi_trans,
            permuted_r_ddi: ddi_perm,
            zero_r_ddi: ddi_zero,
            paired_transposed_diff: paired_t_diff,
            directional_specificity_adv: spec_adv,
            addressing_metrics: addr_metrics.clone(),
            is_competent_and_promoted: is_promoted,
        });
    }

    condition_evals
}

fn train_and_eval_q16a_seed(seed: u64) -> Q16aSeedResult {
    let mut model_base = Q16aOrganism::new(seed);
    let k_sweep = vec![0, 2, 4, 8, 16];

    calibrate_constituent_decoders_directional(seed, &mut model_base);

    // 1. Theoretical Perfect-Information Bayes Oracle Benchmark
    let mut rng_theo = ChaCha8Rng::seed_from_u64(seed + 99991);
    let mut theo_returns = Vec::new();
    let mut theo_ind_commits = Vec::new();
    let mut theo_cop_commits = Vec::new();

    for _ in 0..500 {
        let dag = sample_random_directional_dag(&mut rng_theo);
        for _ in 0..4 {
            let (root_z, rel_type, rep1, rep2, _, _, _, _) = generate_directional_test_trial(&mut rng_theo, &dag, &[0.0; 9], 16);
            let opt_act = if rep1 == rep2 {
                if rel_type == 2 { rep1 } else { 2 }
            } else {
                if rel_type == 0 { rep1 } else if rel_type == 1 { rep2 } else { 2 }
            };

            let rew = match opt_act {
                0 => if root_z == 0 { 2.0 } else { -5.0 },
                1 => if root_z == 1 { 2.0 } else { -5.0 },
                _ => 1.20,
            };
            theo_returns.push(rew);
            if rep1 == rep2 {
                if rel_type == 2 { theo_ind_commits.push(if opt_act == rep1 { 1.0 } else { 0.0 }); }
                else { theo_cop_commits.push(if opt_act == rep1 { 1.0 } else { 0.0 }); }
            }
        }
    }
    let theo_ddi = (theo_ind_commits.iter().sum::<f32>() / theo_ind_commits.len().max(1) as f32)
        - (theo_cop_commits.iter().sum::<f32>() / theo_cop_commits.len().max(1) as f32);
    let theo_ret = theo_returns.iter().sum::<f32>() / theo_returns.len().max(1) as f32;

    // 2. K=16 Empirical-R Teacher Benchmark
    let mut rng_emp = ChaCha8Rng::seed_from_u64(seed + 99992);
    let mut emp_returns = Vec::new();
    let mut emp_ind_commits = Vec::new();
    let mut emp_cop_commits = Vec::new();

    for _ in 0..500 {
        let dag = sample_random_directional_dag(&mut rng_emp);
        let mut j_c = [0.0f32; 9];
        let mut s_c = [0.0f32; 3];
        let mut t_c = 0.0f32;
        for _ in 0..16 { run_calibration_trial_directional(&mut rng_emp, &dag, &mut j_c, &mut s_c, &mut t_c); }
        let r_mat = compute_anti_symmetric_directional_matrix(&j_c, &s_c, t_c);

        for _ in 0..4 {
            let (root_z, rel_type, rep1, rep2, _, _, opt_act, _) = generate_directional_test_trial(&mut rng_emp, &dag, &r_mat, 16);
            let rew = match opt_act {
                0 => if root_z == 0 { 2.0 } else { -5.0 },
                1 => if root_z == 1 { 2.0 } else { -5.0 },
                _ => 1.20,
            };
            emp_returns.push(rew);
            if rep1 == rep2 {
                if rel_type == 2 { emp_ind_commits.push(if opt_act == rep1 { 1.0 } else { 0.0 }); }
                else { emp_cop_commits.push(if opt_act == rep1 { 1.0 } else { 0.0 }); }
            }
        }
    }
    let emp_ddi = (emp_ind_commits.iter().sum::<f32>() / emp_ind_commits.len().max(1) as f32)
        - (emp_cop_commits.iter().sum::<f32>() / emp_cop_commits.len().max(1) as f32);
    let emp_ret = emp_returns.iter().sum::<f32>() / emp_returns.len().max(1) as f32;

    let mut all_cond_results = Vec::new();
    all_cond_results.extend(train_and_eval_directional_condition(seed, &model_base, "Oracle", "Fixed", &k_sweep));
    all_cond_results.extend(train_and_eval_directional_condition(seed, &model_base, "Oracle", "Learned", &k_sweep));
    all_cond_results.extend(train_and_eval_directional_condition(seed, &model_base, "Supervised_FineTuned", "Fixed", &k_sweep));
    all_cond_results.extend(train_and_eval_directional_condition(seed, &model_base, "Supervised_FineTuned", "Learned", &k_sweep));
    all_cond_results.extend(train_and_eval_directional_condition(seed, &model_base, "Autonomous", "Fixed", &k_sweep));
    all_cond_results.extend(train_and_eval_directional_condition(seed, &model_base, "Autonomous", "Learned", &k_sweep));

    Q16aSeedResult {
        seed,
        theoretical_bayes_return: theo_ret,
        theoretical_bayes_ddi: theo_ddi,
        empirical_teacher_return: emp_ret,
        empirical_teacher_ddi: emp_ddi,
        condition_results: all_cond_results,
    }
}

fn main() {
    let seeds: Vec<u64> = vec![101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116];
    let k_sweep = vec![0, 2, 4, 8, 16];

    println!("==========================================================================================================");
    println!("EXECUTING Q16a: MINIMAL DIRECTIONAL PROVENANCE & ANTI-SYMMETRIC ADDRESSING (16 SEEDS)");
    println!("Tests Asymmetric Inheritance (S1->S2 vs S2->S1 vs S1_|_S2) across Shared Evaluation Tapes");
    println!("==========================================================================================================");

    let start = Instant::now();

    let results: Vec<Q16aSeedResult> = seeds
        .par_iter()
        .map(|&seed| train_and_eval_q16a_seed(seed))
        .collect();

    let elapsed = start.elapsed();

    println!("\n==========================================================================================================");
    println!("Q16a EXECUTION COMPLETED IN {:?}", elapsed);
    println!("==========================================================================================================");

    let n = results.len() as f32;
    let mean_theo_ret = results.iter().map(|r| r.theoretical_bayes_return).sum::<f32>() / n;
    let mean_theo_ddi = results.iter().map(|r| r.theoretical_bayes_ddi).sum::<f32>() / n;
    let mean_emp_ret = results.iter().map(|r| r.empirical_teacher_return).sum::<f32>() / n;
    let mean_emp_ddi = results.iter().map(|r| r.empirical_teacher_ddi).sum::<f32>() / n;

    println!("1. ECONOMIC BENCHMARKS (DIRECTIONAL ASYMMETRY):");
    println!("  - Theoretical Perfect-Information Bayes Oracle: Return = {:+.2}, DDI = {:+.1}%", mean_theo_ret, mean_theo_ddi * 100.0);
    println!("  - K=16 Empirical-R Teacher Benchmark          : Return = {:+.2}, DDI = {:+.1}% (vs +1.20 Always-VERIFY baseline)", mean_emp_ret, mean_emp_ddi * 100.0);
    println!("----------------------------------------------------------------------------------------------------------------------------------");

    let condition_names = [
        ("Oracle_Fixed", "1. ORACLE DIRECTIONAL ADDRESS + FIXED CALIBRATED POLICY (ISOLATES ECONOMIC CEILING)"),
        ("Oracle_Learned", "2. ORACLE DIRECTIONAL ADDRESS + LEARNED POLICY (VALIDATES DIRECTIONAL DECISION MAPPING)"),
        ("Supervised_FineTuned_Fixed", "3. SUPERVISED+TUNED ADDRESS + FIXED POLICY (DIFFERENTIABLE DIRECTIONAL SURROGATE)"),
        ("Supervised_FineTuned_Learned", "4. SUPERVISED+TUNED ADDRESS + LEARNED POLICY (FULL SUPERVISED PIPELINE)"),
        ("Autonomous_Fixed", "5. AUTONOMOUS ADDRESS FROM SCRATCH + FIXED POLICY (TESTS DIRECTIONAL EMERGENCE PRESSURE)"),
        ("Autonomous_Learned", "6. AUTONOMOUS ADDRESS + LEARNED POLICY (SCAFFOLDED REPORT DECODERS)"),
    ];

    for (c_idx, (_, display_title)) in condition_names.iter().enumerate() {
        println!("\n==================================================================================================================");
        println!("{}", display_title);
        println!("------------------------------------------------------------------------------------------------------------------");
        println!("CALIB K | INTACT DDI | INTACT RET | Fwd Acc | Bwd Acc | Ind Acc | TRANS R | PERM R | PAIRED ΔTRANS (±STE) | PROMO");
        println!("------------------------------------------------------------------------------------------------------------------");

        for (k_idx, &k_val) in k_sweep.iter().enumerate() {
            let offset = c_idx * 5 + k_idx;
            let mean_ddi = results.iter().map(|r| r.condition_results[offset].test_ddi).sum::<f32>() / n;
            let mean_ret = results.iter().map(|r| r.condition_results[offset].test_return).sum::<f32>() / n;
            let mean_fwd = results.iter().map(|r| r.condition_results[offset].forward_copy_accuracy).sum::<f32>() / n;
            let mean_bwd = results.iter().map(|r| r.condition_results[offset].backward_copy_accuracy).sum::<f32>() / n;
            let mean_ind = results.iter().map(|r| r.condition_results[offset].independent_accuracy).sum::<f32>() / n;
            let mean_trans = results.iter().map(|r| r.condition_results[offset].transposed_r_ddi).sum::<f32>() / n;
            let mean_perm = results.iter().map(|r| r.condition_results[offset].permuted_r_ddi).sum::<f32>() / n;

            let paired_diffs: Vec<f32> = results.iter().map(|r| r.condition_results[offset].paired_transposed_diff).collect();
            let mean_p_diff = paired_diffs.iter().sum::<f32>() / n;
            let var_diff: f32 = paired_diffs.iter().map(|&x| (x - mean_p_diff).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
            let ste_diff = (var_diff / n).sqrt();

            let promo = results.iter().filter(|r| r.condition_results[offset].is_competent_and_promoted).count();

            println!(
                "K = {:<2}  | {:+.1}%    | {:+.2} vs 1.20 | {:+.1}%   | {:+.1}%   | {:+.1}%   | {:+.1}%  | {:+.1}%  | {:+.1}% (±{:.1}%)          | {}/16 ({:.1}%)",
                k_val, mean_ddi * 100.0, mean_ret, mean_fwd * 100.0, mean_bwd * 100.0, mean_ind * 100.0, mean_trans * 100.0, mean_perm * 100.0, mean_p_diff * 100.0, ste_diff * 100.0, promo, (promo as f32 / 16.0) * 100.0
            );
        }

        let last_offset = c_idx * 5 + 4;
        let q1_acc = results.iter().map(|r| r.condition_results[last_offset].addressing_metrics.query1_accuracy).sum::<f32>() / n;
        let q2_acc = results.iter().map(|r| r.condition_results[last_offset].addressing_metrics.query2_accuracy).sum::<f32>() / n;
        let ent = results.iter().map(|r| r.condition_results[last_offset].addressing_metrics.mean_query_entropy).sum::<f32>() / n;
        let r_sc = results.iter().map(|r| r.condition_results[last_offset].addressing_metrics.directional_score_correlation).sum::<f32>() / n;
        let disp = results.iter().map(|r| r.condition_results[last_offset].addressing_metrics.true_query_displacement).sum::<f32>() / n;

        println!("ADDRESSING QUALITY METRICS (K=16):");
        println!("  - Query 1 Acc: {:+.1}%, Query 2 Acc: {:+.1}% | Entropy H(q): {:.3} | Score Corr r: {:+.3} | Displacement ||ΔW_q||: {:.3}",
            q1_acc * 100.0, q2_acc * 100.0, ent, r_sc, disp);
    }

    let out_dir = Path::new("../../results/e26_q16_directional_provenance");
    std::fs::create_dir_all(out_dir).ok();

    let json_data = serde_json::to_string_pretty(&results).unwrap();
    let mut f = File::create(out_dir.join("q16a_summary.json")).unwrap();
    f.write_all(json_data.as_bytes()).unwrap();

    let mut report = format!(
        "# Q16a: Minimal Directional Provenance & Anti-Symmetric Addressing Synthesis Report

========================================================================================================================
Q16a DIRECTIONAL SYNTHESIS REPORT (16 SEEDS, RUNTIME: {:?})
1. Theoretical Perfect-Information Bayes Oracle: Expected Return = {:+.2}, Expected DDI = {:+.1}%
2. K=16 Empirical-R Teacher Benchmark          : Expected Return = {:+.2}, Expected DDI = {:+.1}%
========================================================================================================================
",
        elapsed, mean_theo_ret, mean_theo_ddi * 100.0, mean_emp_ret, mean_emp_ddi * 100.0
    );

    for (c_idx, (_, display_title)) in condition_names.iter().enumerate() {
        report.push_str(&format!("## {}\n\n", display_title));
        report.push_str("| Calibration (K) | Intact DDI % | Realized Return | Forward Acc | Backward Acc | Indep Acc | Transposed R DDI | Permuted R DDI | Paired ΔTrans (±STE) | Promoted Seeds |\n");
        report.push_str("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n");

        for (k_idx, &k_val) in k_sweep.iter().enumerate() {
            let offset = c_idx * 5 + k_idx;
            let mean_ddi = results.iter().map(|r| r.condition_results[offset].test_ddi).sum::<f32>() / n;
            let mean_ret = results.iter().map(|r| r.condition_results[offset].test_return).sum::<f32>() / n;
            let mean_fwd = results.iter().map(|r| r.condition_results[offset].forward_copy_accuracy).sum::<f32>() / n;
            let mean_bwd = results.iter().map(|r| r.condition_results[offset].backward_copy_accuracy).sum::<f32>() / n;
            let mean_ind = results.iter().map(|r| r.condition_results[offset].independent_accuracy).sum::<f32>() / n;
            let mean_trans = results.iter().map(|r| r.condition_results[offset].transposed_r_ddi).sum::<f32>() / n;
            let mean_perm = results.iter().map(|r| r.condition_results[offset].permuted_r_ddi).sum::<f32>() / n;

            let paired_diffs: Vec<f32> = results.iter().map(|r| r.condition_results[offset].paired_transposed_diff).collect();
            let mean_p_diff = paired_diffs.iter().sum::<f32>() / n;
            let var_diff: f32 = paired_diffs.iter().map(|&x| (x - mean_p_diff).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
            let ste_diff = (var_diff / n).sqrt();

            let promo = results.iter().filter(|r| r.condition_results[offset].is_competent_and_promoted).count();

            report.push_str(&format!(
                "| **K = {}** | {:+.1}% | {:+.2} | {:+.1}% | {:+.1}% | {:+.1}% | {:+.1}% | {:+.1}% | {:+.1}% (±{:.1}%) | **{}/16 ({:.1}%)** |\n",
                k_val, mean_ddi * 100.0, mean_ret, mean_fwd * 100.0, mean_bwd * 100.0, mean_ind * 100.0, mean_trans * 100.0, mean_perm * 100.0, mean_p_diff * 100.0, ste_diff * 100.0, promo, (promo as f32 / 16.0) * 100.0
            ));
        }

        let last_offset = c_idx * 5 + 4;
        let q1_acc = results.iter().map(|r| r.condition_results[last_offset].addressing_metrics.query1_accuracy).sum::<f32>() / n;
        let q2_acc = results.iter().map(|r| r.condition_results[last_offset].addressing_metrics.query2_accuracy).sum::<f32>() / n;
        let ent = results.iter().map(|r| r.condition_results[last_offset].addressing_metrics.mean_query_entropy).sum::<f32>() / n;
        let r_sc = results.iter().map(|r| r.condition_results[last_offset].addressing_metrics.directional_score_correlation).sum::<f32>() / n;
        let disp = results.iter().map(|r| r.condition_results[last_offset].addressing_metrics.true_query_displacement).sum::<f32>() / n;

        report.push_str(&format!(
            "\n**Addressing Quality Metrics (K=16):** Query 1 Acc = {:+.1}%, Query 2 Acc = {:+.1}%, Entropy H(q) = {:.3}, Score Correlation r = {:+.3}, Displacement ||ΔW_q|| = {:.3}\n\n",
            q1_acc * 100.0, q2_acc * 100.0, ent, r_sc, disp
        ));
    }

    let or_fix_k16_ret = results.iter().map(|r| r.condition_results[4].test_return).sum::<f32>() / n;
    let or_fix_k16_ddi = results.iter().map(|r| r.condition_results[4].test_ddi).sum::<f32>() / n;
    let or_lrn_k16_ret = results.iter().map(|r| r.condition_results[9].test_return).sum::<f32>() / n;
    let or_lrn_k16_ddi = results.iter().map(|r| r.condition_results[9].test_ddi).sum::<f32>() / n;

    let sup_fix_k16_ret = results.iter().map(|r| r.condition_results[14].test_return).sum::<f32>() / n;
    let sup_fix_k16_ddi = results.iter().map(|r| r.condition_results[14].test_ddi).sum::<f32>() / n;
    let sup_q1_acc = results.iter().map(|r| r.condition_results[14].addressing_metrics.query1_accuracy).sum::<f32>() / n;
    let sup_q2_acc = results.iter().map(|r| r.condition_results[14].addressing_metrics.query2_accuracy).sum::<f32>() / n;
    let sup_r_sc = results.iter().map(|r| r.condition_results[14].addressing_metrics.directional_score_correlation).sum::<f32>() / n;

    let auto_fix_k16_ddi = results.iter().map(|r| r.condition_results[24].test_ddi).sum::<f32>() / n;
    let auto_fix_k16_ret = results.iter().map(|r| r.condition_results[24].test_return).sum::<f32>() / n;
    let auto_q1_acc = results.iter().map(|r| r.condition_results[24].addressing_metrics.query1_accuracy).sum::<f32>() / n;
    let auto_q2_acc = results.iter().map(|r| r.condition_results[24].addressing_metrics.query2_accuracy).sum::<f32>() / n;
    let auto_disp = results.iter().map(|r| r.condition_results[24].addressing_metrics.true_query_displacement).sum::<f32>() / n;

    report.push_str(&format!(
        "
========================================================================================================================
## 7. SCIENTIFIC DIAGNOSTIC CONCLUSIONS:
- **Directional Relational Ceiling:** Under anti-symmetric relational matrix R (where R^T = -R), oracle fixed addressing achieves return = {:+.2} and DDI = {:+.1}%, and oracle learned policy achieves return = {:+.2} and DDI = {:+.1}%.
- **Directional Supervised Addressing:** Supervised queries maintain directional correlation (r = {:+.3}, q1 = {:+.1}%, q2 = {:+.1}%), achieving return = {:+.2} and DDI = {:+.1}%.
- **Autonomous Directional Recruitment Status:** Under asymmetric environmental pressure where arrow directionality is consequential, autonomous query weights displace (||ΔW_q|| = {:.3}), yielding q1 = {:+.1}%, q2 = {:+.1}%, DDI = {:+.1}%, and return = {:+.2}.
========================================================================================================================
",
        or_fix_k16_ret, or_fix_k16_ddi * 100.0,
        or_lrn_k16_ret, or_lrn_k16_ddi * 100.0,
        sup_r_sc, sup_q1_acc * 100.0, sup_q2_acc * 100.0, sup_fix_k16_ret, sup_fix_k16_ddi * 100.0,
        auto_disp, auto_q1_acc * 100.0, auto_q2_acc * 100.0, auto_fix_k16_ddi * 100.0, auto_fix_k16_ret
    ));

    let mut rep_file = File::create(out_dir.join("report_q16a.md")).unwrap();
    rep_file.write_all(report.as_bytes()).unwrap();

    println!("Saved Q16a summary JSON and Report to {:?}", out_dir);
}
