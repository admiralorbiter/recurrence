//! Hardened Q10d Policy Readout Trainer & Causal Behavioral Necessity Assay in Native Rust.

use crate::environment::{DualLocusRegulatorEnv, GroundTruthStateV2, ObservationV2};
use crate::oracle::{ObservationBeliefOracle, ReactiveSensorDropPolicy, Policy};
use crate::organism::{DualLocusOrganism, COMBINED_DIM, EMBED_DIM, HIDDEN_DIM, TOTAL_INPUT_DIM};
use crate::trainer::fit_and_eval_ridge;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

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
    pub paired_belief_oracle_return: f32,
    pub paired_heuristic_return: f32,
    pub return_advantage_over_heuristic: f32,
    pub motor_competence_intact: f32,
    pub motor_competence_non_decision_steps: f32,
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
    let mut paired_belief_returns = Vec::with_capacity(num_episodes);
    let mut paired_heuristic_returns = Vec::with_capacity(num_episodes);

    let mut maint_severe = Vec::new();
    let mut maint_safe = Vec::new();

    let mut non_decision_hits = 0;
    let mut non_decision_total = 0;

    let mut dec_h = Vec::new();
    let mut dec_curr = Vec::new();
    let mut dec_short = Vec::new();
    let mut dec_event_precursors = Vec::new();
    let mut dec_full = Vec::new();
    let mut dec_log_odds = Vec::new();
    let mut dec_q = Vec::new();

    let mut belief_oracle = ObservationBeliefOracle { threshold: 0.60, precursor_noise_std: 0.35, precursor_history: Vec::new() };
    let mut reactive_heuristic = ReactiveSensorDropPolicy { threshold: 0.60 };

    for ep in 0..num_episodes {
        let tape = env.generate_deterministic_tape(env.episode_len, seed + 70000 + ep as u64 * 10);

        // 1. Paired Belief Oracle Return on identical tape
        let (mut o_b, mut g_b) = env.reset(Some(tape.clone()));
        belief_oracle.reset();
        let mut done_b = false;
        let mut ret_b = 0.0;
        while !done_b {
            let a = belief_oracle.act(&o_b, Some(&g_b));
            let (no, r, d, ng) = env.step(a);
            ret_b += r;
            done_b = d;
            o_b = no;
            g_b = ng;
        }
        paired_belief_returns.push(ret_b);

        // 2. Paired Reactive Heuristic Return on identical tape
        let (mut o_h, mut g_h) = env.reset(Some(tape.clone()));
        let mut done_h = false;
        let mut ret_h = 0.0;
        while !done_h {
            let a = reactive_heuristic.act(&o_h, Some(&g_h));
            let (no, r, d, ng) = env.step(a);
            ret_h += r;
            done_h = d;
            o_h = no;
            g_h = ng;
        }
        paired_heuristic_returns.push(ret_h);

        // 3. Paired Decision-Window Always-Maintain Baseline Return
        let (mut o_m, mut g_m) = env.reset(Some(tape.clone()));
        let mut done_m = false;
        let mut ret_m = 0.0;
        while !done_m {
            let a = if o_m.is_decision_window == 1 {
                2 // Always maintain in designated decision window
            } else {
                if o_m.symbol >= 3 && o_m.symbol <= 4 { o_m.symbol - 3 } else { 0 }
            };
            let (no, r, d, ng) = env.step(a);
            ret_m += r;
            done_m = d;
            o_m = no;
            g_m = ng;
        }

        // 3. Model Evaluation on identical tape
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

            // Lesion history ONLY at decision window if wipe_decision_h is true
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
            } else {
                // Non-decision window steps: evaluate if motor accuracy is preserved!
                let expected_goal = if obs.symbol >= 3 && obs.symbol <= 4 { obs.symbol - 3 } else { 0 };
                if act == expected_goal {
                    non_decision_hits += 1;
                }
                non_decision_total += 1;
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

    let mean_belief_ret: f32 = paired_belief_returns.iter().sum::<f32>() / num_episodes as f32;
    let mean_heur_ret: f32 = paired_heuristic_returns.iter().sum::<f32>() / num_episodes as f32;

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
    let non_dec_comp = if non_decision_total > 0 { non_decision_hits as f32 / non_decision_total as f32 } else { 0.0 };

    Q10dEvaluationMetrics {
        mean_return: mean_ret,
        std_return: var.sqrt(),
        paired_belief_oracle_return: mean_belief_ret,
        paired_heuristic_return: mean_heur_ret,
        return_advantage_over_heuristic: mean_ret - mean_heur_ret,
        motor_competence_intact: motor_comp,
        motor_competence_non_decision_steps: non_dec_comp,
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

/// Hardened Policy Readout Regimes:
///   1. "supervised_risk_conditional": Supervised cross-entropy directly to risk-conditional optimal action
///   2. "downstream_counterfactual_return": Clones at decision window, rolls out through shock impact to episode end
///   3. "trained_actor_critic_rl": Full Monte-Carlo policy gradient with concurrently trained value critic
pub fn train_policy_readout_regimes(
    frozen_base_model: &DualLocusOrganism,
    regime: &str,
    num_episodes: usize,
    lr: f32,
    gamma: f32,
    seed: u64,
) -> DualLocusOrganism {
    let mut model = frozen_base_model.clone();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut env = DualLocusRegulatorEnv::new(seed, false);

    let mut m_pol = vec![0.0; 4 * COMBINED_DIM];
    let mut v_pol = vec![0.0; 4 * COMBINED_DIM];
    let mut m_val = vec![0.0; COMBINED_DIM];
    let mut v_val = vec![0.0; COMBINED_DIM];
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
        let mut ep_supervised_target = Vec::new();
        let mut ep_downstream_q = Vec::new();

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

            // 1. Supervised Risk-Conditional Target (Maintain if severe risk q >= 0.50 at decision window, else motor goal)
            let goal_act = if obs.symbol >= 3 && obs.symbol <= 4 { obs.symbol - 3 } else { 0 };
            let opt_act = if obs.is_decision_window == 1 && gt.bayesian_risk_q >= 0.50 {
                2
            } else {
                goal_act
            };

            // 2. Downstream Counterfactual Return Rollout:
            // Roll each candidate action through shock impact to episode end
            let mut downstream_q = [0.0; 4];
            if regime == "downstream_counterfactual_return" && obs.is_decision_window == 1 {
                let snap = env.snapshot();
                for test_a in 0..4 {
                    let mut branch_env = env.clone();
                    branch_env.restore(&snap);

                    let (mut b_obs, mut b_r, mut b_done, _) = branch_env.step(test_a);
                    let mut b_tot_ret = b_r;
                    let mut b_gamma_accum = gamma;
                    let mut b_h: Option<Vec<f32>> = Some(h_next.clone());

                    while !b_done {
                        let (b_h_next, b_logits, _) = model.step(&b_obs, b_h.as_deref());
                        let b_act = if b_obs.symbol >= 3 && b_obs.symbol <= 4 { b_obs.symbol - 3 } else { 0 };
                        let (nb_obs, nb_r, nb_done, _) = branch_env.step(b_act);
                        b_tot_ret += b_gamma_accum * nb_r;
                        b_gamma_accum *= gamma;
                        b_done = nb_done;
                        b_obs = nb_obs;
                        b_h = Some(b_h_next);
                    }
                    downstream_q[test_a] = b_tot_ret;
                }
            }

            ep_combined.push(comb);
            ep_actions.push(act);
            ep_probs.push(probs);
            ep_values.push(val);
            ep_supervised_target.push(opt_act);
            ep_downstream_q.push(downstream_q);

            let (next_obs, rew, is_done, next_gt) = env.step(act);
            ep_rewards.push(rew);
            done = is_done;
            obs = next_obs;
            gt = next_gt;
            h = Some(h_next);
        }

        t_opt += 1;
        let t_steps = ep_rewards.len();

        // Compute discounted returns-to-go for trained RL critic
        let mut returns_to_go = vec![0.0; t_steps];
        let mut running_g = 0.0;
        for t in (0..t_steps).rev() {
            running_g = ep_rewards[t] + gamma * running_g;
            returns_to_go[t] = running_g;
        }

        for t in 0..t_steps {
            let comb = &ep_combined[t];
            let probs = &ep_probs[t];

            match regime {
                "supervised_risk_conditional" => {
                    let target_a = ep_supervised_target[t];
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
                "downstream_counterfactual_return" => {
                    let q_vals = &ep_downstream_q[t];
                    let is_dec = q_vals.iter().any(|&v| v != 0.0);
                    if is_dec {
                        // Expected value E_pi[Q]
                        let exp_q: f32 = (0..4).map(|k| probs[k] * q_vals[k]).sum();
                        // Gradient of expected reward: sum_k (Q_k - E[Q]) * nabla pi_k = sum_k probs[k]*(Q_k - E[Q]) * x
                        for k in 0..4 {
                            let adv = q_vals[k] - exp_q;
                            let grad_logit = probs[k] * adv;
                            for j in 0..COMBINED_DIM {
                                let idx = k * COMBINED_DIM + j;
                                let g = -grad_logit * comb[j];
                                m_pol[idx] = 0.9 * m_pol[idx] + 0.1 * g;
                                v_pol[idx] = 0.999 * v_pol[idx] + 0.001 * g * g;
                                let m_hat = m_pol[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                                let v_hat = v_pol[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                                model.policy_w[idx] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
                            }
                        }
                    } else {
                        // Standard motor reward gradient
                        let a = ep_actions[t];
                        let adv = returns_to_go[t] - ep_values[t];
                        for k in 0..4 {
                            let delta_pi = (if k == a { 1.0 } else { 0.0 }) - probs[k];
                            for j in 0..COMBINED_DIM {
                                let idx = k * COMBINED_DIM + j;
                                let g = -adv * delta_pi * comb[j];
                                m_pol[idx] = 0.9 * m_pol[idx] + 0.1 * g;
                                v_pol[idx] = 0.999 * v_pol[idx] + 0.001 * g * g;
                                let m_hat = m_pol[idx] / (1.0 - 0.9f32.powi(t_opt as i32));
                                let v_hat = v_pol[idx] / (1.0 - 0.999f32.powi(t_opt as i32));
                                model.policy_w[idx] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
                            }
                        }
                    }
                }
                _ => { // trained_actor_critic_rl
                    let a = ep_actions[t];
                    let adv = returns_to_go[t] - ep_values[t];

                    // 1. Value critic update
                    for j in 0..COMBINED_DIM {
                        let g = -adv * comb[j];
                        m_val[j] = 0.9 * m_val[j] + 0.1 * g;
                        v_val[j] = 0.999 * v_val[j] + 0.001 * g * g;
                        let m_hat = m_val[j] / (1.0 - 0.9f32.powi(t_opt as i32));
                        let v_hat = v_val[j] / (1.0 - 0.999f32.powi(t_opt as i32));
                        model.value_w[j] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
                    }

                    // 2. Policy actor update
                    for k in 0..4 {
                        let delta_pi = (if k == a { 1.0 } else { 0.0 }) - probs[k];
                        for j in 0..COMBINED_DIM {
                            let idx = k * COMBINED_DIM + j;
                            let g = -adv * delta_pi * comb[j];
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
