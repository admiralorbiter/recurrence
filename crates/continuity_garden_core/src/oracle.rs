//! Garden v2 Oracle & Gate D0a Observability Calibration in Native Rust.

use crate::environment::{DualLocusRegulatorEnv, GroundTruthStateV2, ObservationV2};

pub trait Policy {
    fn reset(&mut self) {}
    fn act(&mut self, obs: &ObservationV2, gt: Option<&GroundTruthStateV2>) -> usize;
}

pub struct NeverMaintainPolicy;
impl Policy for NeverMaintainPolicy {
    fn act(&mut self, obs: &ObservationV2, _gt: Option<&GroundTruthStateV2>) -> usize {
        if obs.symbol >= 3 && obs.symbol <= 4 { obs.symbol - 3 } else { 0 }
    }
}

pub struct AlwaysMaintainPolicy;
impl Policy for AlwaysMaintainPolicy {
    fn act(&mut self, _obs: &ObservationV2, _gt: Option<&GroundTruthStateV2>) -> usize {
        2 // MAINTAIN_A
    }
}

pub struct ReactiveSensorDropPolicy {
    pub threshold: f32,
}
impl Policy for ReactiveSensorDropPolicy {
    fn act(&mut self, obs: &ObservationV2, _gt: Option<&GroundTruthStateV2>) -> usize {
        let goal = if obs.symbol >= 3 && obs.symbol <= 4 { obs.symbol - 3 } else { 0 };
        if obs.sensor_a < self.threshold {
            2
        } else {
            goal
        }
    }
}

pub struct WarningReflexPolicy {
    pub already_maintained: bool,
}
impl Policy for WarningReflexPolicy {
    fn reset(&mut self) {
        self.already_maintained = false;
    }
    fn act(&mut self, obs: &ObservationV2, _gt: Option<&GroundTruthStateV2>) -> usize {
        let goal = if obs.symbol >= 3 && obs.symbol <= 4 { obs.symbol - 3 } else { 0 };
        if obs.warning_cue > 0.0 {
            if !self.already_maintained {
                self.already_maintained = true;
                return 2;
            }
        } else {
            self.already_maintained = false;
        }
        goal
    }
}

pub struct ShortHistoryWindowPolicy;
impl Policy for ShortHistoryWindowPolicy {
    fn act(&mut self, obs: &ObservationV2, _gt: Option<&GroundTruthStateV2>) -> usize {
        if obs.symbol >= 3 && obs.symbol <= 4 { obs.symbol - 3 } else { 0 }
    }
}

pub struct ObservationBeliefOracle {
    pub threshold: f32,
    pub precursor_noise_std: f32,
    pub precursor_history: Vec<f32>,
}
impl Policy for ObservationBeliefOracle {
    fn reset(&mut self) {
        self.precursor_history.clear();
    }
    fn act(&mut self, obs: &ObservationV2, _gt: Option<&GroundTruthStateV2>) -> usize {
        let goal = if obs.symbol >= 3 && obs.symbol <= 4 { obs.symbol - 3 } else { 0 };

        if obs.warning_cue > 0.0 {
            self.precursor_history.push(obs.warning_cue);
        }

        if obs.is_decision_window == 1 {
            let q_severe = DualLocusRegulatorEnv::compute_exact_bayesian_posterior(
                &self.precursor_history,
                0.55,
                0.80,
                0.20,
                self.precursor_noise_std,
            );
            self.precursor_history.clear();
            if q_severe >= self.threshold {
                return 2;
            }
        }

        if obs.sensor_a < 0.40 {
            return 2;
        }

        goal
    }
}

pub struct PrivilegedGroundTruthOracle;
impl Policy for PrivilegedGroundTruthOracle {
    fn act(&mut self, obs: &ObservationV2, gt: Option<&GroundTruthStateV2>) -> usize {
        let goal = if obs.symbol >= 3 && obs.symbol <= 4 { obs.symbol - 3 } else { 0 };
        if let Some(gt) = gt {
            if obs.is_decision_window == 1 && gt.pending_shock_magnitude >= 0.50 {
                return 2;
            }
        }
        if obs.sensor_a < 0.40 {
            return 2;
        }
        goal
    }
}

pub struct PolicyEvalMetrics {
    pub mean_return: f32,
    pub std_return: f32,
    pub mean_maintenance_count: f32,
    pub mean_target_hits: f32,
}

pub fn evaluate_policy_on_env<P: Policy>(
    mut policy: P,
    env: &mut DualLocusRegulatorEnv,
    num_episodes: usize,
    seed: u64,
) -> PolicyEvalMetrics {
    let mut returns = Vec::with_capacity(num_episodes);
    let mut maint_counts = Vec::with_capacity(num_episodes);
    let mut hits_counts = Vec::with_capacity(num_episodes);

    for ep in 0..num_episodes {
        let tape = env.generate_deterministic_tape(env.episode_len, seed + ep as u64 * 100);
        let (mut obs, mut gt) = env.reset(Some(tape));
        policy.reset();

        let mut done = false;
        let mut ep_ret = 0.0;
        let mut maint = 0;
        let mut hits = 0;

        while !done {
            let act = policy.act(&obs, Some(&gt));
            if act == 2 || act == 3 {
                maint += 1;
            }
            let (next_obs, rew, is_done, next_gt) = env.step(act);
            ep_ret += rew;
            if rew > 0.5 {
                hits += 1;
            }
            done = is_done;
            obs = next_obs;
            gt = next_gt;
        }

        returns.push(ep_ret);
        maint_counts.push(maint as f32);
        hits_counts.push(hits as f32);
    }

    let mean_ret: f32 = returns.iter().sum::<f32>() / num_episodes as f32;
    let var: f32 = returns.iter().map(|&r| (r - mean_ret).powi(2)).sum::<f32>() / num_episodes as f32;

    PolicyEvalMetrics {
        mean_return: mean_ret,
        std_return: var.sqrt(),
        mean_maintenance_count: maint_counts.iter().sum::<f32>() / num_episodes as f32,
        mean_target_hits: hits_counts.iter().sum::<f32>() / num_episodes as f32,
    }
}
