//! Gate E Ideal Learned-Source Oracle: Evaluates optimal Bayesian decision rules for ideal observers with learned source statistics.

use crate::provenance_kernel::{ProvenanceEventTape, ProvenanceGardenEnv, SourceType};

#[derive(Debug, Clone)]
pub struct IdealLearnedSourceOracle;

impl IdealLearnedSourceOracle {
    /// Ideal Learned-Source Oracle: Computes exact posterior P(z=1 | O_1..t) given learned source reliability and dependency structure.
    pub fn compute_ideal_bayesian_posterior(tape: &ProvenanceEventTape) -> f32 {
        let mut total_log_odds = 0.0f32;
        let mut processed_dependency_groups = Vec::new();

        for ev in &tape.events {
            // For dependency groups (e.g. S1 copying S0), independent root evidence is only counted ONCE!
            if processed_dependency_groups.contains(&ev.dependency_group_id) {
                continue;
            }
            processed_dependency_groups.push(ev.dependency_group_id);

            let s_node = &tape.sources[ev.immediate_source_id.min(tape.sources.len() - 1)];
            let rel = match s_node.source_type {
                SourceType::DomainExpert => s_node.domain_competence[tape.domain_id],
                SourceType::Opposite => 0.15,
                SourceType::Random => 0.50,
                SourceType::CopiedRelay => 0.5 + 0.4 * (0.80f32).powi(ev.transmission_depth as i32),
                _ => s_node.reliability,
            };

            let lr = if ev.reported_content == 1 {
                rel / (1.0 - rel).max(1e-4)
            } else {
                (1.0 - rel) / rel.max(1e-4)
            };
            total_log_odds += lr.ln();
        }

        1.0 / (1.0 + (-total_log_odds).exp())
    }

    /// Evaluates economic calibration for a given scout config:
    /// Compares Privileged Ceiling vs Observational Oracle vs Source-Blind Baseline.
    pub fn calibrate_scout_economy(config_name: &str, num_episodes: usize, seed: u64) -> (f32, f32, f32, bool) {
        let mut env = ProvenanceGardenEnv::new(seed, config_name);

        let mut privileged_returns = Vec::new();
        let mut obs_oracle_returns = Vec::new();
        let mut blind_returns = Vec::new();

        for ep in 0..num_episodes {
            let tape = env.generate_tape_for_scout(config_name, seed + ep as u64 * 31);

            // 1. Privileged Oracle (Always knows root_truth_z)
            let (mut _o, _) = env.reset(Some(tape.clone()));
            let mut ep_ret_priv = 0.0;
            let mut done = false;
            while !done {
                let (_, rew, is_done, _) = env.step(tape.root_truth_z);
                ep_ret_priv += rew;
                done = is_done;
            }
            privileged_returns.push(ep_ret_priv);

            // 2. Observational Belief Oracle (Uses Bayesian posterior derived from observations)
            let (mut _o, _) = env.reset(Some(tape.clone()));
            let mut ep_ret_obs = 0.0;
            let p_bayes = Self::compute_ideal_bayesian_posterior(&tape);
            let opt_action = if p_bayes >= 0.50 { 1 } else { 0 };
            done = false;
            while !done {
                let (_, rew, is_done, _) = env.step(opt_action);
                ep_ret_obs += rew;
                done = is_done;
            }
            obs_oracle_returns.push(ep_ret_obs);

            // 3. Source-Blind / Content-Only Baseline (Follows surface reported content without knowing source)
            let (mut _o, _) = env.reset(Some(tape.clone()));
            let mut ep_ret_blind = 0.0;
            let last_reported_content = tape.events.last().map(|e| e.reported_content).unwrap_or(1);
            done = false;
            while !done {
                let (_, rew, is_done, _) = env.step(last_reported_content);
                ep_ret_blind += rew;
                done = is_done;
            }
            blind_returns.push(ep_ret_blind);
        }

        let mean_priv = privileged_returns.iter().sum::<f32>() / num_episodes as f32;
        let mean_obs = obs_oracle_returns.iter().sum::<f32>() / num_episodes as f32;
        let mean_blind = blind_returns.iter().sum::<f32>() / num_episodes as f32;

        let is_valid = mean_obs > mean_blind + 0.05;

        (mean_priv, mean_obs, mean_blind, is_valid)
    }
}
