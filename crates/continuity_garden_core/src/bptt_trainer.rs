//! Full Multi-Step Backpropagation Through Time (BPTT) GRU Trainer in Native Rust.

use crate::environment::{DualLocusRegulatorEnv, GroundTruthStateV2, ObservationV2};
use crate::organism::{DualLocusOrganism, COMBINED_DIM, EMBED_DIM, HIDDEN_DIM, TOTAL_INPUT_DIM};
use crate::trainer::fit_and_eval_ridge;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedParameterNorms {
    pub gru_ih_rel_norm: f32,
    pub gru_hh_rel_norm: f32,
    pub policy_rel_norm: f32,
    pub value_rel_norm: f32,
    pub embed_rel_norm: f32,
    pub total_param_mse: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Q10dEvaluationMetrics {
    pub mean_return: f32,
    pub std_return: f32,
    pub motor_competence: f32,
    pub r2_h_log_odds: f32,
    pub r2_h_posterior_q: f32,
    pub r2_current_obs: f32,
    pub r2_short_window: f32,
    pub r2_event_relative_precursors: f32,
    pub r2_full_history_flattened: f32,
    pub delta_r2_vs_current: f32,
    pub delta_r2_vs_short_window: f32,
    pub p_maint_severe_risk: f32,
    pub p_maint_safe_risk: f32,
    pub maint_specificity: f32,
    pub param_norms: DetailedParameterNorms,
}

pub fn compute_detailed_param_norms(init_model: &DualLocusOrganism, curr_model: &DualLocusOrganism) -> DetailedParameterNorms {
    let norm_diff = |a_vec: &[f32], b_vec: &[f32]| -> (f32, f32, usize) {
        let mut diff_sq = 0.0;
        let mut base_sq = 0.0;
        for (a, b) in a_vec.iter().zip(b_vec.iter()) {
            diff_sq += (a - b).powi(2);
            base_sq += a.powi(2);
        }
        (diff_sq.sqrt() / (base_sq.sqrt() + 1e-8), diff_sq, a_vec.len())
    };

    let (ih_rel, ih_sq, n_ih) = norm_diff(&init_model.gru_w_ih, &curr_model.gru_w_ih);
    let (hh_rel, hh_sq, n_hh) = norm_diff(&init_model.gru_w_hh, &curr_model.gru_w_hh);
    let (pol_rel, pol_sq, n_pol) = norm_diff(&init_model.policy_w, &curr_model.policy_w);
    let (val_rel, val_sq, n_val) = norm_diff(&init_model.value_w, &curr_model.value_w);
    let (emb_rel, emb_sq, n_emb) = norm_diff(&init_model.symbol_embed, &curr_model.symbol_embed);

    let total_sq = ih_sq + hh_sq + pol_sq + val_sq + emb_sq;
    let total_n = n_ih + n_hh + n_pol + n_val + n_emb;

    DetailedParameterNorms {
        gru_ih_rel_norm: ih_rel,
        gru_hh_rel_norm: hh_rel,
        policy_rel_norm: pol_rel,
        value_rel_norm: val_rel,
        embed_rel_norm: emb_rel,
        total_param_mse: total_sq / total_n as f32,
    }
}

pub fn evaluate_q10d_model(
    model: &DualLocusOrganism,
    init_model: &DualLocusOrganism,
    wipe_decision_h: bool,
    seed: u64,
    num_episodes: usize,
) -> Q10dEvaluationMetrics {
    let mut env = DualLocusRegulatorEnv::new(seed + 50000, false);

    let mut returns = Vec::with_capacity(num_episodes);
    let mut maint_severe = Vec::new();
    let mut maint_safe = Vec::new();

    let mut dec_h = Vec::new();
    let mut dec_curr = Vec::new();
    let mut dec_short = Vec::new();
    let mut dec_event_precursors = Vec::new();
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
        let mut ep_precursor_samples = Vec::new();

        while !done {
            let curr_vec = vec![obs.symbol as f32, obs.sensor_a, obs.sensor_b, obs.warning_cue, obs.is_decision_window as f32];
            ep_obs_hist.push(curr_vec.clone());

            if obs.warning_cue > 0.0 {
                ep_precursor_samples.push(obs.warning_cue);
            }

            let effective_h = if wipe_decision_h && obs.is_decision_window == 1 {
                None
            } else {
                h.as_deref()
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
                h_vec.push(1.0);
                dec_h.push(h_vec);

                dec_curr.push(vec![obs.symbol as f32, obs.sensor_a, obs.sensor_b, obs.warning_cue, 1.0]);

                let prev = if ep_obs_hist.len() >= 2 { ep_obs_hist[ep_obs_hist.len() - 2].clone() } else { vec![0.0; 5] };
                let mut short = prev;
                short.extend_from_slice(&curr_vec);
                short.push(1.0);
                dec_short.push(short);

                // Event-relative public precursor observer [c1, c2, c3, 1.0]
                let mut prec_vec = vec![0.0; 4];
                for (k, &c) in ep_precursor_samples.iter().rev().take(3).enumerate() {
                    prec_vec[2 - k] = c;
                }
                prec_vec[3] = 1.0;
                dec_event_precursors.push(prec_vec);

                let mut full = vec![0.0; 24 * 5 + 1];
                for (k, o) in ep_obs_hist.iter().enumerate() {
                    if k < 24 {
                        for j in 0..5 { full[k * 5 + j] = o[j]; }
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

    let (r2_h_lo, r2_h_q, r2_curr, r2_short, r2_event_prec, r2_full) = if n_split >= 10 {
        let d = dec_h[0].len();
        let mut mean_h = vec![0.0; d];
        let mut std_h = vec![0.0; d];
        for row in &dec_h[..n_split] { for i in 0..d { mean_h[i] += row[i]; } }
        for i in 0..d { mean_h[i] /= n_split as f32; }
        for row in &dec_h[..n_split] { for i in 0..d { std_h[i] += (row[i] - mean_h[i]).powi(2); } }
        for i in 0..d { std_h[i] = (std_h[i] / n_split as f32).sqrt().max(1e-6); }

        let mut norm_h = dec_h.clone();
        for row in norm_h.iter_mut() {
            for i in 0..(d - 1) { row[i] = (row[i] - mean_h[i]) / std_h[i]; }
        }

        let r2_h_lo = fit_and_eval_ridge(&norm_h[..n_split], &dec_log_odds[..n_split], &norm_h[n_split..], &dec_log_odds[n_split..], 10.0);
        let r2_h_q = fit_and_eval_ridge(&norm_h[..n_split], &dec_q[..n_split], &norm_h[n_split..], &dec_q[n_split..], 10.0);
        let r2_curr = fit_and_eval_ridge(&dec_curr[..n_split], &dec_log_odds[..n_split], &dec_curr[n_split..], &dec_log_odds[n_split..], 10.0);
        let r2_short = fit_and_eval_ridge(&dec_short[..n_split], &dec_log_odds[..n_split], &dec_short[n_split..], &dec_log_odds[n_split..], 10.0);
        let r2_event_prec = fit_and_eval_ridge(&dec_event_precursors[..n_split], &dec_log_odds[..n_split], &dec_event_precursors[n_split..], &dec_log_odds[n_split..], 0.01);
        let r2_full = fit_and_eval_ridge(&dec_full[..n_split], &dec_log_odds[..n_split], &dec_full[n_split..], &dec_log_odds[n_split..], 10.0);
        (r2_h_lo, r2_h_q, r2_curr, r2_short, r2_event_prec, r2_full)
    } else {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    };

    let p_sev: f32 = if !maint_severe.is_empty() { maint_severe.iter().sum::<f32>() / maint_severe.len() as f32 } else { 0.0 };
    let p_saf: f32 = if !maint_safe.is_empty() { maint_safe.iter().sum::<f32>() / maint_safe.len() as f32 } else { 0.0 };
    let motor_comp = crate::trainer::evaluate_motor_competence_rust(model, seed + 90000, 20);

    Q10dEvaluationMetrics {
        mean_return: mean_ret,
        std_return: var.sqrt(),
        motor_competence: motor_comp,
        r2_h_log_odds: r2_h_lo,
        r2_h_posterior_q: r2_h_q,
        r2_current_obs: r2_curr,
        r2_short_window: r2_short,
        r2_event_relative_precursors: r2_event_prec,
        r2_full_history_flattened: r2_full,
        delta_r2_vs_current: r2_h_lo - r2_curr,
        delta_r2_vs_short_window: r2_h_lo - r2_short,
        p_maint_severe_risk: p_sev,
        p_maint_safe_risk: p_saf,
        maint_specificity: p_sev - p_saf,
        param_norms: compute_detailed_param_norms(init_model, model),
    }
}

/// Trains readout under 3 distinct policy regimes:
///   1. "supervised_upper_bound": Supervised cross-entropy directly to Bayes-optimal action
///   2. "counterfactual_rewards": Counterfactual reward evaluation of candidate actions cloned at decision point
///   3. "actor_critic_rl": Standard on-policy RL with reward feedback
pub fn train_policy_readout_regimes(
    frozen_base_model: &DualLocusOrganism,
    regime: &str,
    num_episodes: usize,
    lr: f32,
    seed: u64,
) -> DualLocusOrganism {
    let mut model = frozen_base_model.clone();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut env = DualLocusRegulatorEnv::new(seed, false);

    let mut m_pol = vec![0.0; 4 * COMBINED_DIM];
    let mut v_pol = vec![0.0; 4 * COMBINED_DIM];
    let mut t_opt = 0;

    for ep in 1..=num_episodes {
        let tape = env.generate_deterministic_tape(env.episode_len, seed + ep as u64 * 10);
        let (mut obs, mut gt) = env.reset(Some(tape));
        let mut h: Option<Vec<f32>> = None;
        let mut done = false;

        let mut ep_combined = Vec::new();
        let mut ep_actions = Vec::new();
        let mut ep_probs = Vec::new();
        let mut ep_rewards = Vec::new();
        let mut ep_values = Vec::new();
        let mut ep_bayes_optimal = Vec::new();
        let mut ep_counterfactual_rews = Vec::new();

        while !done {
            let (h_next, logits, val) = model.step(&obs, h.as_deref());

            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_l: [f32; 4] = [(logits[0]-max_l).exp(), (logits[1]-max_l).exp(), (logits[2]-max_l).exp(), (logits[3]-max_l).exp()];
            let sum_exp: f32 = exp_l.iter().sum();
            let probs = [exp_l[0]/sum_exp, exp_l[1]/sum_exp, exp_l[2]/sum_exp, exp_l[3]/sum_exp];

            let (act, _) = model.sample_action(&logits, &mut rng);

            let (_, instant_feats) = model.forward_features(&obs);
            let mut comb = Vec::with_capacity(COMBINED_DIM);
            comb.extend_from_slice(&h_next);
            comb.extend_from_slice(&instant_feats);

            // 1. Bayes optimal action: MAINTAIN_A (2) if severe risk at decision window, else motor goal
            let goal_act = if obs.symbol >= 3 && obs.symbol <= 4 { obs.symbol - 3 } else { 0 };
            let opt_act = if obs.is_decision_window == 1 && gt.bayesian_risk_q >= 0.50 {
                2
            } else {
                goal_act
            };

            // 2. Counterfactual reward branch evaluation
            let mut cf_rews = [0.0; 4];
            if obs.is_decision_window == 1 {
                let snap = env.snapshot();
                for test_a in 0..4 {
                    let mut cloned_env = env.clone();
                    cloned_env.restore(&snap);
                    let (_, r_cf, _, _) = cloned_env.step(test_a);
                    cf_rews[test_a] = r_cf;
                }
            }

            ep_combined.push(comb);
            ep_actions.push(act);
            ep_probs.push(probs);
            ep_values.push(val);
            ep_bayes_optimal.push(opt_act);
            ep_counterfactual_rews.push(cf_rews);

            let (next_obs, rew, is_done, next_gt) = env.step(act);
            ep_rewards.push(rew);
            done = is_done;
            obs = next_obs;
            gt = next_gt;
            h = Some(h_next);
        }

        t_opt += 1;
        let t_steps = ep_rewards.len();

        for t in 0..t_steps {
            let comb = &ep_combined[t];
            let probs = &ep_probs[t];

            match regime {
                "supervised_upper_bound" => {
                    let target_a = ep_bayes_optimal[t];
                    for k in 0..4 {
                        let delta_pi = (if k == target_a { 1.0 } else { 0.0 }) - probs[k];
                        for j in 0..COMBINED_DIM {
                            let idx = k * COMBINED_DIM + j;
                            let g = -delta_pi * comb[j];
                            m_pol[idx] = 0.9 * m_pol[idx] + 0.1 * g;
                            v_pol[idx] = 0.999 * v_pol[idx] + 0.001 * g * g;
                            let m_hat = m_pol[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                            let v_hat = v_pol[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                            model.policy_w[idx] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
                        }
                    }
                }
                "counterfactual_rewards" => {
                    let cf_rews = &ep_counterfactual_rews[t];
                    let mean_cf = cf_rews.iter().sum::<f32>() / 4.0;
                    for k in 0..4 {
                        let adv = cf_rews[k] - mean_cf;
                        let delta_pi = adv * (1.0 - probs[k]);
                        for j in 0..COMBINED_DIM {
                            let idx = k * COMBINED_DIM + j;
                            let g = -delta_pi * comb[j];
                            m_pol[idx] = 0.9 * m_pol[idx] + 0.1 * g;
                            v_pol[idx] = 0.999 * v_pol[idx] + 0.001 * g * g;
                            let m_hat = m_pol[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                            let v_hat = v_pol[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                            model.policy_w[idx] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
                        }
                    }
                }
                _ => { // actor_critic_rl
                    let nv = if t + 1 < t_steps { ep_values[t + 1] } else { 0.0 };
                    let td_err = ep_rewards[t] + 0.95 * nv - ep_values[t];
                    let a = ep_actions[t];
                    for k in 0..4 {
                        let delta_pi = (if k == a { 1.0 } else { 0.0 }) - probs[k];
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
                }
            }
        }
    }

    model
}
