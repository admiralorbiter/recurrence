//! Gate E Bayesian Provenance Oracle & Economic Calibration.
//! Computes exact Bayesian posterior over root state z given DAG dependency structure and calibrates economic value.

use crate::provenance_kernel::{ProvenanceEventTape, ProvenanceGardenEnv, SourceType};

#[derive(Debug, Clone)]
pub struct ProvenanceOracle;

impl ProvenanceOracle {
    /// Computes the exact Bayesian posterior log-odds given DAG structure:
    /// Handles independent corroboration, copied evidence, domain expertise, and signed reliability.
    pub fn compute_exact_bayesian_posterior(tape: &ProvenanceEventTape) -> f32 {
        let mut total_log_odds = 0.0f32;
        let mut processed_dependency_groups = Vec::new();

        for ev in &tape.events {
            // For dependency groups, copied evidence is only counted ONCE for root evidence!
            if processed_dependency_groups.contains(&ev.dependency_group_id) {
                continue;
            }
            processed_dependency_groups.push(ev.dependency_group_id);

            let s_node = &tape.sources[ev.immediate_source_id.min(tape.sources.len() - 1)];
            let rel = match s_node.source_type {
                SourceType::DomainExpert => s_node.domain_competence[tape.domain_id],
                SourceType::Opposite => 0.15,
                SourceType::Random => 0.50,
                SourceType::CopiedNode => 0.85 * 0.90f32.powi(ev.transmission_depth as i32),
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
    /// Asserts R_provenance_aware > R_content_only_blind.
    pub fn calibrate_scout_economy(config_name: &str, num_episodes: usize, seed: u64) -> (f32, f32, bool) {
        let mut env = ProvenanceGardenEnv::new(seed, config_name);

        let mut oracle_returns = Vec::new();
        let mut blind_returns = Vec::new();

        for ep in 0..num_episodes {
            let tape = env.generate_tape_for_scout(config_name, seed + ep as u64 * 31);

            // 1. Oracle (Provenance-Aware)
            let (mut obs, _) = env.reset(Some(tape.clone()));
            let mut done = false;
            let mut ep_ret_oracle = 0.0;
            let p_bayes = Self::compute_exact_bayesian_posterior(&tape);
            let opt_action = if p_bayes >= 0.50 { 1 } else { 0 };

            while !done {
                let (next_obs, rew, is_done, _) = env.step(opt_action);
                ep_ret_oracle += rew;
                done = is_done;
                obs = next_obs;
            }
            oracle_returns.push(ep_ret_oracle);

            // 2. Source-Blind / Content-Only Baseline (Always follows surface reported content or defaults)
            let (mut obs_b, _) = env.reset(Some(tape.clone()));
            let mut done_b = false;
            let mut ep_ret_blind = 0.0;
            // Blind follows the surface symbol without knowing whether source was opposite, copied, or untrusted
            let last_reported_content = tape.events.last().map(|e| e.reported_content).unwrap_or(1);

            while !done_b {
                let (next_obs, rew, is_done, _) = env.step(last_reported_content);
                ep_ret_blind += rew;
                done_b = is_done;
                obs_b = next_obs;
            }
            blind_returns.push(ep_ret_blind);
        }

        let mean_oracle = oracle_returns.iter().sum::<f32>() / num_episodes as f32;
        let mean_blind = blind_returns.iter().sum::<f32>() / num_episodes as f32;
        let is_valid = mean_oracle > mean_blind;

        (mean_oracle, mean_blind, is_valid)
    }
}
