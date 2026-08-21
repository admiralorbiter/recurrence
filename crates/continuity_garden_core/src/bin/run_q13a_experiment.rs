//! Q13a: Signed Source Relational Binding Diagnostic.
//! Investigates whether the recurrent substrate represents source identity, content, conjunction binding, and signed evidence over delay.

use continuity_garden_core::provenance_kernel::{ProvenanceEvent, ProvenanceEventTape, ProvenanceGardenEnv, SourceNode, SourceType};
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

#[derive(Debug, Clone)]
pub struct Q13aOrganism {
    pub symbol_embed: Vec<f32>,
    pub sensor_w: Vec<f32>, // 16 x 5
    pub sensor_b: Vec<f32>, // 16
    pub gru_w_ih: Vec<f32>, // 192 x 48
    pub gru_w_hh: Vec<f32>, // 192 x 64
    pub gru_b: Vec<f32>,    // 192
    pub policy_w: Vec<f32>, // 2 x COMBINED_DIM
    pub policy_b: Vec<f32>, // 2
}

impl Q13aOrganism {
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
            policy_w: rand_vec(2 * COMBINED_DIM, 0.01),
            policy_b: vec![0.0; 2],
        }
    }

    pub fn compute_h_next(&self, sym: usize, ch: [f32; 3], is_dec: f32, h_prev: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
        let mut input_feats = Vec::with_capacity(TOTAL_INPUT_DIM);
        let mut instant_feats = Vec::with_capacity(32);

        let s_idx = sym.min(3);
        let s_slice = &self.symbol_embed[s_idx * EMBED_DIM..(s_idx + 1) * EMBED_DIM];
        input_feats.extend_from_slice(s_slice);
        instant_feats.extend_from_slice(s_slice);

        input_feats.extend_from_slice(&vec![0.0; EMBED_DIM]); // dummy last action

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

    pub fn compute_logits(&self, h: &[f32], instant_feats: &[f32]) -> [f32; 2] {
        let mut comb = Vec::with_capacity(COMBINED_DIM);
        comb.extend_from_slice(h);
        comb.extend_from_slice(instant_feats);

        let mut logits = [0.0; 2];
        for k in 0..2 {
            let mut sum = self.policy_b[k];
            for j in 0..COMBINED_DIM { sum += self.policy_w[k * COMBINED_DIM + j] * comb[j]; }
            logits[k] = sum;
        }
        logits
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayConditionResult {
    pub delay_steps: usize,
    pub r2_source_identity: f32,
    pub r2_report_content: f32,
    pub r2_conjunction_binding: f32,
    pub r2_signed_evidence: f32,
    pub helpful_following_rate: f32,
    pub opposite_inversion_rate: f32,
    pub empirical_inversion_effect: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q13aSeedResult {
    pub seed: u64,
    pub delay_results: Vec<DelayConditionResult>,
    pub plastic_delay4_inversion_effect: f32,
}

fn generate_signed_episode(
    seed: u64,
    ep_idx: usize,
    delay_steps: usize,
) -> (usize, usize, usize, f32, Vec<(usize, [f32; 3], f32)>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed + ep_idx as u64 * 37);
    let root_z = if rng.gen::<f64>() < 0.50 { 0 } else { 1 };
    let s_id = rng.gen_range(0..3); // 0: Helpful (0.85), 1: Random (0.50), 2: Opposite (0.15)
    let p_acc = match s_id {
        0 => 0.85f32,
        1 => 0.50f32,
        _ => 0.15f32,
    };
    let rep = if rng.gen::<f32>() < p_acc { root_z } else { 1 - root_z };

    // Signed evidence: e = (2r - 1) * ln(rel / (1 - rel))
    let log_lr = (p_acc / (1.0 - p_acc)).ln();
    let signed_evidence = (2.0 * rep as f32 - 1.0) * log_lr;

    // Episode steps:
    // Step 0: Blank
    // Step 1: Acquisition (symbol = rep + 1, neutral channel active)
    // Steps 2..=1+delay: Blank delay (symbol = 0, channel = 0)
    // Step 2+delay: Decision (symbol = 2 [candidate proposition], is_dec = 1.0)
    let mut steps = Vec::new();
    steps.push((0, [0.0, 0.0, 0.0], 0.0)); // Step 0

    let mut ch = [0.0; 3];
    ch[s_id] = 1.0;
    steps.push((rep + 1, ch, 0.0)); // Step 1 Acquisition

    for _ in 0..delay_steps {
        steps.push((0, [0.0, 0.0, 0.0], 0.0)); // Delay steps
    }

    steps.push((2, [0.0, 0.0, 0.0], 1.0)); // Decision window

    (root_z, s_id, rep, signed_evidence, steps)
}

fn evaluate_delay_condition(
    model: &mut Q13aOrganism,
    seed: u64,
    delay_steps: usize,
) -> DelayConditionResult {
    // 1. Train linear policy head on optimal action a*(h) = argmax_a Q(h, a)
    let mut m_pol = vec![0.0; 2 * COMBINED_DIM];
    let mut v_pol = vec![0.0; 2 * COMBINED_DIM];
    let mut t_opt = 0;

    for ep in 1..=800 {
        let (root_z, s_id, rep, signed_ev, steps) = generate_signed_episode(seed, ep, delay_steps);
        let opt_act = if signed_ev > 0.0 { 1 } else if signed_ev < 0.0 { 0 } else { rep };

        let mut h: Option<Vec<f32>> = None;
        let mut dec_comb = Vec::new();
        let mut dec_probs = [0.0; 2];

        for (sym, ch, is_dec) in steps {
            let (h_next, instant_feats) = model.compute_h_next(sym, ch, is_dec, h.as_deref());
            if is_dec > 0.5 {
                let logits = model.compute_logits(&h_next, &instant_feats);
                let max_l = logits[0].max(logits[1]);
                let exp_l = [(logits[0] - max_l).exp(), (logits[1] - max_l).exp()];
                let sum_exp = exp_l[0] + exp_l[1];
                dec_probs = [exp_l[0] / sum_exp, exp_l[1] / sum_exp];

                let mut comb = Vec::with_capacity(COMBINED_DIM);
                comb.extend_from_slice(&h_next);
                comb.extend_from_slice(&instant_feats);
                dec_comb = comb;
            }
            h = Some(h_next);
        }

        t_opt += 1;
        let target_a = opt_act;
        for k in 0..2 {
            let delta_pi = (if k == target_a { 1.0 } else { 0.0 }) - dec_probs[k];
            for j in 0..COMBINED_DIM {
                let idx = k * COMBINED_DIM + j;
                let g = -delta_pi * dec_comb[j];
                m_pol[idx] = 0.9 * m_pol[idx] + 0.1 * g;
                v_pol[idx] = 0.999 * v_pol[idx] + 0.001 * g * g;
                let m_hat = m_pol[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                let v_hat = v_pol[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                model.policy_w[idx] -= 0.02 * m_hat / (v_hat.sqrt() + 1e-8);
            }
        }
    }

    // 2. Probing & Behavioral Evaluation on 200 Held-out Episodes
    let mut dec_h_list = Vec::new();
    let mut target_source = Vec::new();
    let mut target_content = Vec::new();
    let mut target_conjunction = Vec::new();
    let mut target_signed_ev = Vec::new();

    let mut helpful_follows = Vec::new();
    let mut opposite_inverts = Vec::new();
    let mut random_inverts = Vec::new();

    for ep in 0..200 {
        let (root_z, s_id, rep, signed_ev, steps) = generate_signed_episode(seed + 90000, ep, delay_steps);
        let mut h: Option<Vec<f32>> = None;

        for (sym, ch, is_dec) in steps {
            let (h_next, instant_feats) = model.compute_h_next(sym, ch, is_dec, h.as_deref());
            if is_dec > 0.5 {
                let mut h_vec = h_next.clone();
                h_vec.push(1.0);
                dec_h_list.push(h_vec);

                target_source.push(s_id as f32);
                target_content.push(rep as f32);
                target_conjunction.push((s_id * 2 + rep) as f32);
                target_signed_ev.push(signed_ev);

                let logits = model.compute_logits(&h_next, &instant_feats);
                let act = if logits[1] > logits[0] { 1 } else { 0 };

                if s_id == 0 {
                    helpful_follows.push(if act == rep { 1.0 } else { 0.0 });
                } else if s_id == 2 {
                    opposite_inverts.push(if act == 1 - rep { 1.0 } else { 0.0 });
                } else {
                    random_inverts.push(if act == 1 - rep { 1.0 } else { 0.0 });
                }
            }
            h = Some(h_next);
        }
    }

    let n_split = dec_h_list.len() / 2;
    let eval_probe = |targets: &[f32]| -> f32 {
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

    let r2_s = eval_probe(&target_source);
    let r2_c = eval_probe(&target_content);
    let r2_conj = eval_probe(&target_conjunction);
    let r2_ev = eval_probe(&target_signed_ev);

    let p_help = if !helpful_follows.is_empty() { helpful_follows.iter().sum::<f32>() / helpful_follows.len() as f32 } else { 0.0 };
    let p_opp = if !opposite_inverts.is_empty() { opposite_inverts.iter().sum::<f32>() / opposite_inverts.len() as f32 } else { 0.0 };
    let p_rand = if !random_inverts.is_empty() { random_inverts.iter().sum::<f32>() / random_inverts.len() as f32 } else { 0.0 };

    DelayConditionResult {
        delay_steps,
        r2_source_identity: r2_s,
        r2_report_content: r2_c,
        r2_conjunction_binding: r2_conj,
        r2_signed_evidence: r2_ev,
        helpful_following_rate: p_help,
        opposite_inversion_rate: p_opp,
        empirical_inversion_effect: p_opp - p_rand,
    }
}

fn main() {
    let seeds: Vec<u64> = vec![101, 102, 103, 104, 105, 106, 107, 108];
    let delays = vec![0, 1, 2, 4, 8];

    println!("==========================================================================================================");
    println!("EXECUTING Q13a: SIGNED SOURCE RELATIONAL BINDING DIAGNOSTIC (8 SEEDS ACROSS DELAY SWEEP)");
    println!("Probing: Source Class (s) | Report Content (r) | Conjunction (s x r) | Signed Evidence (e)");
    println!("==========================================================================================================");

    let start = Instant::now();

    let results: Vec<Q13aSeedResult> = seeds
        .par_iter()
        .map(|&seed| {
            let mut model = Q13aOrganism::new(seed);
            let mut delay_res = Vec::new();

            for &d in &delays {
                let mut cond_model = model.clone();
                let res = evaluate_delay_condition(&mut cond_model, seed, d);
                delay_res.push(res);
            }

            Q13aSeedResult {
                seed,
                delay_results: delay_res,
                plastic_delay4_inversion_effect: 0.0,
            }
        })
        .collect();

    let elapsed = start.elapsed();

    println!("\n==========================================================================================================");
    println!("Q13a EXECUTION FINISHED IN {:?}", elapsed);
    println!("==========================================================================================================");

    println!("------------------------------------------------------------------------------------------------------------------");
    println!("DELAY | R^2(Source s) | R^2(Content r) | R^2(Binding s x r) | R^2(Signed e) | Helpful % | Invert % | Inversion Effect");
    println!("------------------------------------------------------------------------------------------------------------------");

    let n = results.len() as f32;

    for (d_idx, &d) in delays.iter().enumerate() {
        let mean_r2_s: f32 = results.iter().map(|r| r.delay_results[d_idx].r2_source_identity).sum::<f32>() / n;
        let mean_r2_c: f32 = results.iter().map(|r| r.delay_results[d_idx].r2_report_content).sum::<f32>() / n;
        let mean_r2_conj: f32 = results.iter().map(|r| r.delay_results[d_idx].r2_conjunction_binding).sum::<f32>() / n;
        let mean_r2_ev: f32 = results.iter().map(|r| r.delay_results[d_idx].r2_signed_evidence).sum::<f32>() / n;
        let mean_help: f32 = results.iter().map(|r| r.delay_results[d_idx].helpful_following_rate).sum::<f32>() / n;
        let mean_opp: f32 = results.iter().map(|r| r.delay_results[d_idx].opposite_inversion_rate).sum::<f32>() / n;
        let mean_inv_eff: f32 = results.iter().map(|r| r.delay_results[d_idx].empirical_inversion_effect).sum::<f32>() / n;

        println!(
            "d = {:<2} | {:+.3}         | {:+.3}          | {:+.3}              | {:+.3}         | {:+.1}%     | {:+.1}%    | {:+.1}%",
            d, mean_r2_s, mean_r2_c, mean_r2_conj, mean_r2_ev, mean_help * 100.0, mean_opp * 100.0, mean_inv_eff * 100.0
        );
    }

    println!("------------------------------------------------------------------------------------------------------------------");

    let out_dir = Path::new("../../results/e24_q13_signed_source_binding");
    std::fs::create_dir_all(out_dir).ok();

    let json_data = serde_json::to_string_pretty(&results).unwrap();
    let mut f = File::create(out_dir.join("q13a_summary.json")).unwrap();
    f.write_all(json_data.as_bytes()).unwrap();

    let mut report_body = format!(
        "# Q13a: Signed Source Relational Binding Diagnostic Report

========================================================================================================================
Q13a DIAGNOSTIC SYNTHESIS: RELATIONAL BINDING OVER DELAYS (8 SEEDS, RUNTIME: {:?})
========================================================================================================================

| Delay (Steps) | R²(Source Identity) | R²(Report Content) | R²(Binding s × r) | R²(Signed Evidence e) | Helpful Following | Opposite Inversion | Net Inversion Effect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
",
        elapsed
    );

    for (d_idx, &d) in delays.iter().enumerate() {
        let mean_r2_s: f32 = results.iter().map(|r| r.delay_results[d_idx].r2_source_identity).sum::<f32>() / n;
        let mean_r2_c: f32 = results.iter().map(|r| r.delay_results[d_idx].r2_report_content).sum::<f32>() / n;
        let mean_r2_conj: f32 = results.iter().map(|r| r.delay_results[d_idx].r2_conjunction_binding).sum::<f32>() / n;
        let mean_r2_ev: f32 = results.iter().map(|r| r.delay_results[d_idx].r2_signed_evidence).sum::<f32>() / n;
        let mean_help: f32 = results.iter().map(|r| r.delay_results[d_idx].helpful_following_rate).sum::<f32>() / n;
        let mean_opp: f32 = results.iter().map(|r| r.delay_results[d_idx].opposite_inversion_rate).sum::<f32>() / n;
        let mean_inv_eff: f32 = results.iter().map(|r| r.delay_results[d_idx].empirical_inversion_effect).sum::<f32>() / n;

        report_body.push_str(&format!(
            "| **d = {}** | {:+.3} | {:+.3} | {:+.3} | {:+.3} | {:+.1}% | {:+.1}% | **{:+.1}%** |\n",
            d, mean_r2_s, mean_r2_c, mean_r2_conj, mean_r2_ev, mean_help * 100.0, mean_opp * 100.0, mean_inv_eff * 100.0
        ));
    }

    report_body.push_str("\n========================================================================================================================\n");

    let mut rep_file = File::create(out_dir.join("report.md")).unwrap();
    rep_file.write_all(report_body.as_bytes()).unwrap();

    println!("Saved Q13a summary JSON and Report to {:?}", out_dir);
}
