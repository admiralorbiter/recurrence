//! High-Speed Parallel Developmental Trainer & Representation Assay in Native Rust.

use crate::environment::{DualLocusRegulatorEnv, GroundTruthStateV2, ObservationV2};
use crate::organism::{DualLocusOrganism, COMBINED_DIM, EMBED_DIM, HIDDEN_DIM, TOTAL_INPUT_DIM};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const CHECKPOINT_EPISODES: [usize; 9] = [0, 25, 50, 100, 200, 400, 800, 1600, 3200];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetrics {
    pub mean_return: f32,
    pub std_return: f32,
    pub motor_competence_baseline: f32,
    pub motor_competence_pass: bool,
    pub ladder_r2_h_bayesian_q: f32,
    pub ladder_r2_current_obs: f32,
    pub ladder_r2_short_window: f32,
    pub ladder_r2_full_history: f32,
    pub delta_r2_vs_current: f32,
    pub delta_r2_vs_short_window: f32,
    pub p_maint_severe_risk: f32,
    pub p_maint_safe_risk: f32,
    pub maint_specificity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedResult {
    pub seed: u64,
    pub t_representation: Option<usize>,
    pub t_recruitment: Option<usize>,
    pub developmental_battery_a: HashMap<String, CheckpointMetrics>,
}

/// Solves linear system A w = b via Gaussian elimination with partial pivoting.
pub fn solve_linear_system(mut a: Vec<f32>, mut b: Vec<f32>, n: usize) -> Option<Vec<f32>> {
    for i in 0..n {
        let mut max_row = i;
        let mut max_val = a[i * n + i].abs();
        for k in (i + 1)..n {
            let val = a[k * n + i].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        if max_row != i {
            for col in 0..n {
                let tmp = a[i * n + col];
                a[i * n + col] = a[max_row * n + col];
                a[max_row * n + col] = tmp;
            }
            let tmp_b = b[i];
            b[i] = b[max_row];
            b[max_row] = tmp_b;
        }

        let pivot = a[i * n + i];
        for col in i..n {
            a[i * n + col] /= pivot;
        }
        b[i] /= pivot;

        for k in 0..n {
            if k != i {
                let factor = a[k * n + i];
                for col in i..n {
                    a[k * n + col] -= factor * a[i * n + col];
                }
                b[k] -= factor * b[i];
            }
        }
    }
    Some(b)
}

/// Computes R^2 of Ridge Regression (X w -> y).
pub fn fit_and_eval_ridge(x_train: &[Vec<f32>], y_train: &[f32], x_test: &[Vec<f32>], y_test: &[f32], lambda: f32) -> f32 {
    let n_samples = x_train.len();
    if n_samples == 0 || x_test.is_empty() {
        return 0.0;
    }
    let d = x_train[0].len();

    let mut a = vec![0.0; d * d];
    let mut b = vec![0.0; d];

    for s in 0..n_samples {
        let xs = &x_train[s];
        let ys = y_train[s];
        for i in 0..d {
            b[i] += xs[i] * ys;
            for j in 0..d {
                a[i * d + j] += xs[i] * xs[j];
            }
        }
    }
    for i in 0..d {
        a[i * d + i] += lambda;
    }

    let weights = match solve_linear_system(a, b, d) {
        Some(w) => w,
        None => return 0.0,
    };

    let mean_y = y_test.iter().sum::<f32>() / y_test.len() as f32;
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;

    for (xt, &yt) in x_test.iter().zip(y_test.iter()) {
        let mut pred = 0.0;
        for i in 0..d {
            pred += weights[i] * xt[i];
        }
        ss_res += (yt - pred).powi(2);
        ss_tot += (yt - mean_y).powi(2);
    }

    if ss_tot < 1e-8 {
        0.0
    } else {
        (1.0 - (ss_res / ss_tot)).max(-1.0)
    }
}

pub fn evaluate_motor_competence_rust(model: &DualLocusOrganism, seed: u64, num_episodes: usize) -> f32 {
    let mut env = DualLocusRegulatorEnv::new(seed, false);
    let mut hits = 0;
    let mut total = 0;

    for ep in 0..num_episodes {
        let mut tape = env.generate_deterministic_tape(env.episode_len, seed + ep as u64 * 10);
        tape.shock_steps.clear();
        tape.precursor_start_steps.clear();
        tape.decision_window_steps.clear();

        let (mut obs, _) = env.reset(Some(tape));
        let mut h: Option<Vec<f32>> = None;
        let mut done = false;

        while !done {
            let (h_next, logits, _) = model.step(&obs, h.as_deref());
            let act = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let (next_obs, rew, is_done, _) = env.step(act);
            if rew > 0.5 {
                hits += 1;
            }
            total += 1;
            done = is_done;
            obs = next_obs;
            h = Some(h_next);
        }
    }
    if total > 0 { hits as f32 / total as f32 } else { 0.0 }
}

pub fn evaluate_checkpoint_rust(
    model: &DualLocusOrganism,
    seed: u64,
    num_episodes: usize,
) -> CheckpointMetrics {
    let mut env = DualLocusRegulatorEnv::new(seed + 50000, false);

    let mut returns = Vec::with_capacity(num_episodes);
    let mut maint_severe = Vec::new();
    let mut maint_safe = Vec::new();

    let mut dec_h = Vec::new();
    let mut dec_curr = Vec::new();
    let mut dec_short = Vec::new();
    let mut dec_full = Vec::new();
    let mut dec_q = Vec::new();

    for ep in 0..num_episodes {
        let tape = env.generate_deterministic_tape(env.episode_len, seed + 70000 + ep as u64 * 10);
        let (mut obs, mut gt) = env.reset(Some(tape));
        let mut h: Option<Vec<f32>> = None;
        let mut done = false;
        let mut ep_ret = 0.0;
        let mut ep_obs_hist = Vec::new();

        while !done {
            let curr_vec = vec![obs.symbol as f32, obs.sensor_a, obs.sensor_b, obs.warning_cue, obs.is_decision_window as f32];
            ep_obs_hist.push(curr_vec.clone());

            let (h_next, logits, _) = model.step(&obs, h.as_deref());
            let act = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if obs.is_decision_window == 1 {
                dec_h.push(h_next.clone());
                dec_curr.push(vec![obs.symbol as f32, obs.sensor_a, obs.sensor_b, obs.warning_cue, 1.0]);

                let prev = if ep_obs_hist.len() >= 2 { ep_obs_hist[ep_obs_hist.len() - 2].clone() } else { vec![0.0; 5] };
                let mut short = prev;
                short.extend_from_slice(&curr_vec);
                short.push(1.0);
                dec_short.push(short);

                let mut full = vec![0.0; 24 * 5 + 1];
                for (k, o) in ep_obs_hist.iter().enumerate() {
                    if k < 24 {
                        for j in 0..5 {
                            full[k * 5 + j] = o[j];
                        }
                    }
                }
                full[24 * 5] = 1.0;
                dec_full.push(full);

                let q = gt.bayesian_risk_q;
                dec_q.push(q);

                if q >= 0.50 {
                    maint_severe.push(if act == 2 { 1.0 } else { 0.0 });
                } else {
                    maint_safe.push(if act == 2 { 1.0 } else { 0.0 });
                }
            }

            let (next_obs, rew, is_done, next_gt) = env.step(act);
            ep_ret += rew;
            done = is_done;
            obs = next_obs;
            gt = next_gt;
            h = Some(h_next);
        }
        returns.push(ep_ret);
    }

    let mean_ret: f32 = returns.iter().sum::<f32>() / num_episodes as f32;
    let var: f32 = returns.iter().map(|&r| (r - mean_ret).powi(2)).sum::<f32>() / num_episodes as f32;

    let n_total = dec_q.len();
    let n_split = n_total / 2;

    let (r2_h, r2_curr, r2_short, r2_full) = if n_split >= 10 {
        let r2_h = fit_and_eval_ridge(&dec_h[..n_split], &dec_q[..n_split], &dec_h[n_split..], &dec_q[n_split..], 1.0);
        let r2_curr = fit_and_eval_ridge(&dec_curr[..n_split], &dec_q[..n_split], &dec_curr[n_split..], &dec_q[n_split..], 1.0);
        let r2_short = fit_and_eval_ridge(&dec_short[..n_split], &dec_q[..n_split], &dec_short[n_split..], &dec_q[n_split..], 1.0);
        let r2_full = fit_and_eval_ridge(&dec_full[..n_split], &dec_q[..n_split], &dec_full[n_split..], &dec_q[n_split..], 1.0);
        (r2_h, r2_curr, r2_short, r2_full)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    let p_sev: f32 = if !maint_severe.is_empty() { maint_severe.iter().sum::<f32>() / maint_severe.len() as f32 } else { 0.0 };
    let p_saf: f32 = if !maint_safe.is_empty() { maint_safe.iter().sum::<f32>() / maint_safe.len() as f32 } else { 0.0 };

    let motor_comp = evaluate_motor_competence_rust(model, seed + 90000, 20);

    CheckpointMetrics {
        mean_return: mean_ret,
        std_return: var.sqrt(),
        motor_competence_baseline: motor_comp,
        motor_competence_pass: motor_comp >= 0.75,
        ladder_r2_h_bayesian_q: r2_h,
        ladder_r2_current_obs: r2_curr,
        ladder_r2_short_window: r2_short,
        ladder_r2_full_history: r2_full,
        delta_r2_vs_current: r2_h - r2_curr,
        delta_r2_vs_short_window: r2_h - r2_short,
        p_maint_severe_risk: p_sev,
        p_maint_safe_risk: p_saf,
        maint_specificity: p_sev - p_saf,
    }
}

pub fn train_duallocus_organism_rust(
    model: &mut DualLocusOrganism,
    num_episodes: usize,
    warmup_episodes: usize,
    lr: f32,
    gamma: f32,
    is_decorative: bool,
    seed: u64,
) -> HashMap<usize, DualLocusOrganism> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut env = DualLocusRegulatorEnv::new(seed, is_decorative);

    let mut saved_checkpoints = HashMap::new();
    saved_checkpoints.insert(0, model.clone());

    // Adam optimizer moments
    let mut m_pol = vec![0.0; 4 * COMBINED_DIM];
    let mut v_pol = vec![0.0; 4 * COMBINED_DIM];
    let mut m_val = vec![0.0; COMBINED_DIM];
    let mut v_val = vec![0.0; COMBINED_DIM];
    let mut t_opt = 0;

    // Sensorimotor Grounding Warmup
    for ep in 0..warmup_episodes {
        let mut tape = env.generate_deterministic_tape(env.episode_len, seed + ep as u64);
        tape.shock_steps.clear();
        tape.precursor_start_steps.clear();
        tape.decision_window_steps.clear();

        let (mut obs, _) = env.reset(Some(tape));
        let mut h: Option<Vec<f32>> = None;
        let mut done = false;

        let mut ep_combined = Vec::new();
        let mut ep_actions = Vec::new();
        let mut ep_values = Vec::new();
        let mut ep_rewards = Vec::new();

        while !done {
            let (h_next, logits, val) = model.step(&obs, h.as_deref());
            let (act, _) = model.sample_action(&logits, &mut rng);

            let (_, instant_feats) = model.forward_features(&obs);
            let mut comb = Vec::with_capacity(COMBINED_DIM);
            comb.extend_from_slice(&h_next);
            comb.extend_from_slice(&instant_feats);

            ep_combined.push(comb);
            ep_actions.push(act);
            ep_values.push(val);

            let (next_obs, rew, is_done, _) = env.step(act);
            ep_rewards.push(rew);
            done = is_done;
            obs = next_obs;
            h = Some(h_next);
        }

        // TD update
        t_opt += 1;
        let t_steps = ep_rewards.len();
        for t in 0..t_steps {
            let nv = if t + 1 < t_steps { ep_values[t + 1] } else { 0.0 };
            let td_err = ep_rewards[t] + gamma * nv - ep_values[t];
            let a = ep_actions[t];
            let comb = &ep_combined[t];

            // Policy gradient step
            for j in 0..COMBINED_DIM {
                let idx = a * COMBINED_DIM + j;
                let g = -td_err * comb[j];
                m_pol[idx] = 0.9 * m_pol[idx] + 0.1 * g;
                v_pol[idx] = 0.999 * v_pol[idx] + 0.001 * g * g;
                let m_hat = m_pol[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                let v_hat = v_pol[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                model.policy_w[idx] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            }

            // Value gradient step
            for j in 0..COMBINED_DIM {
                let g = -td_err * comb[j];
                m_val[j] = 0.9 * m_val[j] + 0.1 * g;
                v_val[j] = 0.999 * v_val[j] + 0.001 * g * g;
                let m_hat = m_val[j] / (1.0 - 0.9f32.powi(t_opt as i32));
                let v_hat = v_val[j] / (1.0 - 0.999f32.powi(t_opt as i32));
                model.value_w[j] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            }
        }
    }

    // Full Developmental Training
    for ep in 1..=num_episodes {
        let tape = env.generate_deterministic_tape(env.episode_len, seed + 1000 + ep as u64 * 10);
        let (mut obs, _) = env.reset(Some(tape));
        let mut h: Option<Vec<f32>> = None;
        let mut done = false;

        let mut ep_combined = Vec::new();
        let mut ep_actions = Vec::new();
        let mut ep_values = Vec::new();
        let mut ep_rewards = Vec::new();

        while !done {
            let (h_next, logits, val) = model.step(&obs, h.as_deref());
            let (act, _) = model.sample_action(&logits, &mut rng);

            let (_, instant_feats) = model.forward_features(&obs);
            let mut comb = Vec::with_capacity(COMBINED_DIM);
            comb.extend_from_slice(&h_next);
            comb.extend_from_slice(&instant_feats);

            ep_combined.push(comb);
            ep_actions.push(act);
            ep_values.push(val);

            let (next_obs, rew, is_done, _) = env.step(act);
            ep_rewards.push(rew);
            done = is_done;
            obs = next_obs;
            h = Some(h_next);
        }

        // TD update
        t_opt += 1;
        let t_steps = ep_rewards.len();
        for t in 0..t_steps {
            let nv = if t + 1 < t_steps { ep_values[t + 1] } else { 0.0 };
            let td_err = ep_rewards[t] + gamma * nv - ep_values[t];
            let a = ep_actions[t];
            let comb = &ep_combined[t];

            for j in 0..COMBINED_DIM {
                let idx = a * COMBINED_DIM + j;
                let g = -td_err * comb[j];
                m_pol[idx] = 0.9 * m_pol[idx] + 0.1 * g;
                v_pol[idx] = 0.999 * v_pol[idx] + 0.001 * g * g;
                let m_hat = m_pol[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                let v_hat = v_pol[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                model.policy_w[idx] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            }

            for j in 0..COMBINED_DIM {
                let g = -td_err * comb[j];
                m_val[j] = 0.9 * m_val[j] + 0.1 * g;
                v_val[j] = 0.999 * v_val[j] + 0.001 * g * g;
                let m_hat = m_val[j] / (1.0 - 0.9f32.powi(t_opt as i32));
                let v_hat = v_val[j] / (1.0 - 0.999f32.powi(t_opt as i32));
                model.value_w[j] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            }
        }

        if CHECKPOINT_EPISODES.contains(&ep) {
            saved_checkpoints.insert(ep, model.clone());
        }
    }

    saved_checkpoints
}
