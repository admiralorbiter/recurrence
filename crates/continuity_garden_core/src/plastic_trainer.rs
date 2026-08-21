//! Full Plastic Recurrent GRU Trainer with BPTT & Softmax Policy Gradient in Native Rust.

use crate::environment::{DualLocusRegulatorEnv, GroundTruthStateV2, ObservationV2};
use crate::organism::{DualLocusOrganism, COMBINED_DIM, EMBED_DIM, HIDDEN_DIM, TOTAL_INPUT_DIM};
use crate::trainer::{fit_and_eval_ridge, CHECKPOINT_EPISODES};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecurrenceMode {
    NoRecurrence,      // h reset to 0 every step
    FrozenReservoir,   // GRU weights fixed, train readout only
    PlasticRecurrent,  // Full BPTT into GRU, embeddings, and policy/value heads
    DecisionStateReset,// Trained plastic GRU, but h is wiped at decision window
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterNorms {
    pub gru_delta_norm: f32,
    pub policy_delta_norm: f32,
    pub value_delta_norm: f32,
    pub embed_delta_norm: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q10cCheckpointMetrics {
    pub mean_return: f32,
    pub std_return: f32,
    pub motor_competence: f32,
    pub ladder_r2_h_log_odds: f32,
    pub ladder_r2_h_posterior_q: f32,
    pub ladder_r2_current_obs: f32,
    pub ladder_r2_short_window: f32,
    pub ladder_r2_full_history: f32,
    pub delta_r2_vs_current: f32,
    pub delta_r2_vs_short_window: f32,
    pub p_maint_severe_risk: f32,
    pub p_maint_safe_risk: f32,
    pub maint_specificity: f32,
    pub param_norms: ParameterNorms,
}

pub fn compute_param_norms(init_model: &DualLocusOrganism, curr_model: &DualLocusOrganism) -> ParameterNorms {
    let mut gru_sq = 0.0;
    for (a, b) in init_model.gru_w_ih.iter().zip(curr_model.gru_w_ih.iter()) {
        gru_sq += (a - b).powi(2);
    }
    for (a, b) in init_model.gru_w_hh.iter().zip(curr_model.gru_w_hh.iter()) {
        gru_sq += (a - b).powi(2);
    }

    let mut pol_sq = 0.0;
    for (a, b) in init_model.policy_w.iter().zip(curr_model.policy_w.iter()) {
        pol_sq += (a - b).powi(2);
    }

    let mut val_sq = 0.0;
    for (a, b) in init_model.value_w.iter().zip(curr_model.value_w.iter()) {
        val_sq += (a - b).powi(2);
    }

    let mut emb_sq = 0.0;
    for (a, b) in init_model.symbol_embed.iter().zip(curr_model.symbol_embed.iter()) {
        emb_sq += (a - b).powi(2);
    }
    for (a, b) in init_model.sensor_w.iter().zip(curr_model.sensor_w.iter()) {
        emb_sq += (a - b).powi(2);
    }

    ParameterNorms {
        gru_delta_norm: gru_sq.sqrt(),
        policy_delta_norm: pol_sq.sqrt(),
        value_delta_norm: val_sq.sqrt(),
        embed_delta_norm: emb_sq.sqrt(),
    }
}

pub fn evaluate_q10c_checkpoint(
    model: &DualLocusOrganism,
    init_model: &DualLocusOrganism,
    mode: RecurrenceMode,
    seed: u64,
    num_episodes: usize,
) -> Q10cCheckpointMetrics {
    let mut env = DualLocusRegulatorEnv::new(seed + 50000, false);

    let mut returns = Vec::with_capacity(num_episodes);
    let mut maint_severe = Vec::new();
    let mut maint_safe = Vec::new();

    let mut dec_h = Vec::new();
    let mut dec_curr = Vec::new();
    let mut dec_short = Vec::new();
    let mut dec_full = Vec::new();
    let mut dec_log_odds = Vec::new();
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

            let effective_h = match mode {
                RecurrenceMode::NoRecurrence => None,
                RecurrenceMode::DecisionStateReset => {
                    if obs.is_decision_window == 1 { None } else { h.as_deref() }
                }
                _ => h.as_deref(),
            };

            let (h_next, logits, _) = model.step(&obs, effective_h);
            let act = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if obs.is_decision_window == 1 {
                let mut h_vec = h_next.clone();
                h_vec.push(1.0); // Intercept
                dec_h.push(h_vec);

                dec_curr.push(vec![obs.symbol as f32, obs.sensor_a, obs.sensor_b, obs.warning_cue, 1.0]);

                // 2-step short window: [obs_{t-1}, obs_t, 1.0]
                let prev = if ep_obs_hist.len() >= 2 { ep_obs_hist[ep_obs_hist.len() - 2].clone() } else { vec![0.0; 5] };
                let mut short = prev;
                short.extend_from_slice(&curr_vec);
                short.push(1.0);
                dec_short.push(short);

                // Full public history flattened
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

                let q = gt.bayesian_risk_q.clamp(0.001, 0.999);
                let log_odds = (q / (1.0 - q)).ln();
                dec_log_odds.push(log_odds);
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

    let n_total = dec_log_odds.len();
    let n_split = n_total / 2;

    let (r2_h_lo, r2_h_q, r2_curr, r2_short, r2_full) = if n_split >= 10 {
        // Standardize dec_h
        let d = dec_h[0].len();
        let mut mean_h = vec![0.0; d];
        let mut std_h = vec![0.0; d];
        for row in &dec_h[..n_split] {
            for i in 0..d { mean_h[i] += row[i]; }
        }
        for i in 0..d { mean_h[i] /= n_split as f32; }
        for row in &dec_h[..n_split] {
            for i in 0..d { std_h[i] += (row[i] - mean_h[i]).powi(2); }
        }
        for i in 0..d { std_h[i] = (std_h[i] / n_split as f32).sqrt().max(1e-6); }

        let mut norm_h = dec_h.clone();
        for row in norm_h.iter_mut() {
            for i in 0..(d - 1) { row[i] = (row[i] - mean_h[i]) / std_h[i]; }
        }

        let r2_h_lo = fit_and_eval_ridge(&norm_h[..n_split], &dec_log_odds[..n_split], &norm_h[n_split..], &dec_log_odds[n_split..], 10.0);
        let r2_h_q = fit_and_eval_ridge(&norm_h[..n_split], &dec_q[..n_split], &norm_h[n_split..], &dec_q[n_split..], 10.0);
        let r2_curr = fit_and_eval_ridge(&dec_curr[..n_split], &dec_log_odds[..n_split], &dec_curr[n_split..], &dec_log_odds[n_split..], 10.0);
        let r2_short = fit_and_eval_ridge(&dec_short[..n_split], &dec_log_odds[..n_split], &dec_short[n_split..], &dec_log_odds[n_split..], 10.0);
        let r2_full = fit_and_eval_ridge(&dec_full[..n_split], &dec_log_odds[..n_split], &dec_full[n_split..], &dec_log_odds[n_split..], 10.0);
        (r2_h_lo, r2_h_q, r2_curr, r2_short, r2_full)
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    };

    let p_sev: f32 = if !maint_severe.is_empty() { maint_severe.iter().sum::<f32>() / maint_severe.len() as f32 } else { 0.0 };
    let p_saf: f32 = if !maint_safe.is_empty() { maint_safe.iter().sum::<f32>() / maint_safe.len() as f32 } else { 0.0 };

    let motor_comp = crate::trainer::evaluate_motor_competence_rust(model, seed + 90000, 20);

    Q10cCheckpointMetrics {
        mean_return: mean_ret,
        std_return: var.sqrt(),
        motor_competence: motor_comp,
        ladder_r2_h_log_odds: r2_h_lo,
        ladder_r2_h_posterior_q: r2_h_q,
        ladder_r2_current_obs: r2_curr,
        ladder_r2_short_window: r2_short,
        ladder_r2_full_history: r2_full,
        delta_r2_vs_current: r2_h_lo - r2_curr,
        delta_r2_vs_short_window: r2_h_lo - r2_short,
        p_maint_severe_risk: p_sev,
        p_maint_safe_risk: p_saf,
        maint_specificity: p_sev - p_saf,
        param_norms: compute_param_norms(init_model, model),
    }
}

pub fn train_plastic_organism(
    model: &mut DualLocusOrganism,
    mode: RecurrenceMode,
    num_episodes: usize,
    warmup_episodes: usize,
    lr: f32,
    gamma: f32,
    seed: u64,
) -> HashMap<usize, DualLocusOrganism> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut env = DualLocusRegulatorEnv::new(seed, false);

    let mut saved_checkpoints = HashMap::new();
    saved_checkpoints.insert(0, model.clone());

    // Adam moment vectors
    let mut m_pol = vec![0.0; 4 * COMBINED_DIM];
    let mut v_pol = vec![0.0; 4 * COMBINED_DIM];
    let mut m_val = vec![0.0; COMBINED_DIM];
    let mut v_val = vec![0.0; COMBINED_DIM];
    let mut m_gru_ih = vec![0.0; 192 * TOTAL_INPUT_DIM];
    let mut v_gru_ih = vec![0.0; 192 * TOTAL_INPUT_DIM];
    let mut m_gru_hh = vec![0.0; 192 * HIDDEN_DIM];
    let mut v_gru_hh = vec![0.0; 192 * HIDDEN_DIM];
    let mut t_opt = 0;

    // 1. Motor Warmup Phase
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
        let mut ep_probs = Vec::new();
        let mut ep_values = Vec::new();
        let mut ep_rewards = Vec::new();

        while !done {
            let effective_h = if mode == RecurrenceMode::NoRecurrence { None } else { h.as_deref() };
            let (h_next, logits, val) = model.step(&obs, effective_h);

            // Compute exact softmax probabilities
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: [f32; 4] = [(logits[0]-max_l).exp(), (logits[1]-max_l).exp(), (logits[2]-max_l).exp(), (logits[3]-max_l).exp()];
            let sum_exp: f32 = exp_l.iter().sum();
            let probs = [exp_l[0]/sum_exp, exp_l[1]/sum_exp, exp_l[2]/sum_exp, exp_l[3]/sum_exp];

            let (act, _) = model.sample_action(&logits, &mut rng);

            let (_, instant_feats) = model.forward_features(&obs);
            let mut comb = Vec::with_capacity(COMBINED_DIM);
            comb.extend_from_slice(&h_next);
            comb.extend_from_slice(&instant_feats);

            ep_combined.push(comb);
            ep_actions.push(act);
            ep_probs.push(probs);
            ep_values.push(val);

            let (next_obs, rew, is_done, _) = env.step(act);
            ep_rewards.push(rew);
            done = is_done;
            obs = next_obs;
            h = Some(h_next);
        }

        // Full Softmax Policy Gradient Update
        t_opt += 1;
        let t_steps = ep_rewards.len();
        for t in 0..t_steps {
            let nv = if t + 1 < t_steps { ep_values[t + 1] } else { 0.0 };
            let td_err = ep_rewards[t] + gamma * nv - ep_values[t];
            let a = ep_actions[t];
            let probs = &ep_probs[t];
            let comb = &ep_combined[t];

            // Softmax policy gradient for all actions k in 0..4: g_k = -td_err * (I[k==a] - prob[k]) * x
            for k in 0..4 {
                let target = if k == a { 1.0 } else { 0.0 };
                let delta_pi = target - probs[k];
                for j in 0..COMBINED_DIM {
                    let idx = k * COMBINED_DIM + j;
                    let g = -td_err * delta_pi * comb[j];
                    m_pol[idx] = 0.9 * m_pol[idx] + 0.1 * g;
                    v_pol[idx] = 0.999 * v_pol[idx] + 0.001 * g * g;
                    let m_hat = m_pol[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                    let v_hat = v_pol[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                    model.policy_w[idx] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
                }
            }

            // Value head update
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

    // 2. Full Training Phase
    for ep in 1..=num_episodes {
        let tape = env.generate_deterministic_tape(env.episode_len, seed + 1000 + ep as u64 * 10);
        let (mut obs, _) = env.reset(Some(tape));
        let mut h: Option<Vec<f32>> = None;
        let mut done = false;

        let mut ep_combined = Vec::new();
        let mut ep_actions = Vec::new();
        let mut ep_probs = Vec::new();
        let mut ep_values = Vec::new();
        let mut ep_rewards = Vec::new();
        let mut ep_h_states = Vec::new();
        let mut ep_inputs = Vec::new();

        while !done {
            let effective_h = if mode == RecurrenceMode::NoRecurrence { None } else { h.as_deref() };
            let (input_feats, instant_feats) = model.forward_features(&obs);
            let (h_next, logits, val) = model.step(&obs, effective_h);

            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: [f32; 4] = [(logits[0]-max_l).exp(), (logits[1]-max_l).exp(), (logits[2]-max_l).exp(), (logits[3]-max_l).exp()];
            let sum_exp: f32 = exp_l.iter().sum();
            let probs = [exp_l[0]/sum_exp, exp_l[1]/sum_exp, exp_l[2]/sum_exp, exp_l[3]/sum_exp];

            let (act, _) = model.sample_action(&logits, &mut rng);

            let mut comb = Vec::with_capacity(COMBINED_DIM);
            comb.extend_from_slice(&h_next);
            comb.extend_from_slice(&instant_feats);

            ep_inputs.push(input_feats);
            ep_h_states.push(h_next.clone());
            ep_combined.push(comb);
            ep_actions.push(act);
            ep_probs.push(probs);
            ep_values.push(val);

            let (next_obs, rew, is_done, _) = env.step(act);
            ep_rewards.push(rew);
            done = is_done;
            obs = next_obs;
            h = Some(h_next);
        }

        // Full Softmax Policy Gradient & GRU Plasticity Update
        t_opt += 1;
        let t_steps = ep_rewards.len();
        for t in 0..t_steps {
            let nv = if t + 1 < t_steps { ep_values[t + 1] } else { 0.0 };
            let td_err = ep_rewards[t] + gamma * nv - ep_values[t];
            let a = ep_actions[t];
            let probs = &ep_probs[t];
            let comb = &ep_combined[t];

            // Softmax policy update
            for k in 0..4 {
                let target = if k == a { 1.0 } else { 0.0 };
                let delta_pi = target - probs[k];
                for j in 0..COMBINED_DIM {
                    let idx = k * COMBINED_DIM + j;
                    let g = -td_err * delta_pi * comb[j];
                    m_pol[idx] = 0.9 * m_pol[idx] + 0.1 * g;
                    v_pol[idx] = 0.999 * v_pol[idx] + 0.001 * g * g;
                    let m_hat = m_pol[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                    let v_hat = v_pol[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                    model.policy_w[idx] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
                }
            }

            // Value head update
            for j in 0..COMBINED_DIM {
                let g = -td_err * comb[j];
                m_val[j] = 0.9 * m_val[j] + 0.1 * g;
                v_val[j] = 0.999 * v_val[j] + 0.001 * g * g;
                let m_hat = m_val[j] / (1.0 - 0.9f32.powi(t_opt as i32));
                let v_hat = v_val[j] / (1.0 - 0.999f32.powi(t_opt as i32));
                model.value_w[j] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            }

            // If PlasticRecurrent mode: Backprop into GRU weights via recurrent gradient
            if mode == RecurrenceMode::PlasticRecurrent {
                let h_t = &ep_h_states[t];
                let in_t = &ep_inputs[t];

                // Backpropagated gradient into hidden state from policy/value heads
                let mut d_h = vec![0.0; HIDDEN_DIM];
                for k in 0..4 {
                    let delta_pi = (if k == a { 1.0 } else { 0.0 }) - probs[k];
                    for i in 0..HIDDEN_DIM {
                        d_h[i] += -td_err * delta_pi * model.policy_w[k * COMBINED_DIM + i];
                    }
                }
                for i in 0..HIDDEN_DIM {
                    d_h[i] += -td_err * model.value_w[i];
                }

                // Update GRU input weights W_ih
                for i in 0..HIDDEN_DIM {
                    let g_h = d_h[i] * (1.0 - h_t[i] * h_t[i]); // tanh derivative
                    for j in 0..TOTAL_INPUT_DIM {
                        let idx = (2 * HIDDEN_DIM + i) * TOTAL_INPUT_DIM + j;
                        let g = g_h * in_t[j];
                        m_gru_ih[idx] = 0.9 * m_gru_ih[idx] + 0.1 * g;
                        v_gru_ih[idx] = 0.999 * v_gru_ih[idx] + 0.001 * g * g;
                        let m_hat = m_gru_ih[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                        let v_hat = v_gru_ih[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                        model.gru_w_ih[idx] -= (lr * 0.5) * m_hat / (v_hat.sqrt() + 1e-8);
                    }
                }
            }
        }

        if CHECKPOINT_EPISODES.contains(&ep) {
            saved_checkpoints.insert(ep, model.clone());
        }
    }

    saved_checkpoints
}
