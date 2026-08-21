//! Continuity Garden v2: Dual-Locus Causal Kernel in Native Rust.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

pub const LATTICE_LEVELS: [f32; 11] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventTapeV2 {
    pub precursor_start_steps: Vec<usize>,
    pub decision_window_steps: Vec<usize>,
    pub shock_steps: Vec<usize>,
    pub shock_magnitudes: Vec<f32>,
    pub precursor_noise: Vec<Vec<f32>>,
    pub sensor_noise_a: Vec<f32>,
    pub sensor_noise_b: Vec<f32>,
    pub motor_bernoulli_draws: Vec<f32>,
    pub world_bernoulli_draws: Vec<f32>,
    pub target_goals: Vec<usize>,
    pub high_demand_steps: Vec<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ObservationV2 {
    pub symbol: usize,
    pub sensor_a: f32,
    pub sensor_b: f32,
    pub warning_cue: f32,
    pub is_decision_window: usize,
    pub last_action_executed: usize,
    pub last_action_intended: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthStateV2 {
    pub step_idx: usize,
    pub internal_reliability_i: f32,
    pub external_reliability_x: f32,
    pub is_precursor_active: bool,
    pub is_decision_window: bool,
    pub shock_pending: bool,
    pub shock_timer: usize,
    pub pending_shock_magnitude: f32,
    pub bayesian_risk_q: f32,
    pub counterfactual_future_i_no_maint: f32,
    pub mitigation_active: bool,
    pub target_goal: usize,
    pub last_effect: Option<usize>,
    pub last_action_executed: usize,
    pub last_action_intended: usize,
    pub is_terminal: bool,
    pub is_decorative: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSnapshotV2 {
    pub step_idx: usize,
    pub internal_reliability_i: f32,
    pub external_reliability_x: f32,
    pub is_precursor_active: bool,
    pub is_decision_window: bool,
    pub shock_pending: bool,
    pub shock_timer: usize,
    pub pending_shock_magnitude: f32,
    pub bayesian_risk_q: f32,
    pub counterfactual_future_i_no_maint: f32,
    pub mitigation_active: bool,
    pub target_goal: usize,
    pub last_effect: Option<usize>,
    pub last_action_executed: usize,
    pub last_action_intended: usize,
    pub is_terminal: bool,
    pub is_decorative: bool,
    pub tape: EventTapeV2,
}

#[derive(Debug, Clone)]
pub struct DualLocusRegulatorEnv {
    pub episode_len: usize,
    pub cost_maintain: f32,
    pub reward_target_hit: f32,
    pub penalty_wrong_effect: f32,
    pub sensor_noise_std: f32,
    pub precursor_noise_std: f32,
    pub is_decorative: bool,
    pub seed: u64,

    tape: Option<EventTapeV2>,
    step_idx: usize,
    i_t: f32,
    x_t: f32,
    is_precursor_active: bool,
    is_decision_window: bool,
    shock_pending: bool,
    shock_timer: usize,
    pending_shock_magnitude: f32,
    bayesian_risk_q: f32,
    counterfactual_future_i: f32,
    mitigation_active: bool,
    target_goal: usize,
    last_effect: Option<usize>,
    last_executed: usize,
    last_intended: usize,
}

impl DualLocusRegulatorEnv {
    pub fn new(seed: u64, is_decorative: bool) -> Self {
        Self {
            episode_len: 24,
            cost_maintain: 0.15,
            reward_target_hit: 1.00,
            penalty_wrong_effect: -0.50,
            sensor_noise_std: 0.08,
            precursor_noise_std: 0.35,
            is_decorative,
            seed,
            tape: None,
            step_idx: 0,
            i_t: 1.0,
            x_t: 1.0,
            is_precursor_active: false,
            is_decision_window: false,
            shock_pending: false,
            shock_timer: 0,
            pending_shock_magnitude: 0.0,
            bayesian_risk_q: 0.55,
            counterfactual_future_i: 1.0,
            mitigation_active: false,
            target_goal: 0,
            last_effect: None,
            last_executed: 4,
            last_intended: 0,
        }
    }

    pub fn quantize_lattice(val: f32) -> f32 {
        let mut best_idx = 0;
        let mut min_diff = f32::MAX;
        for (idx, &level) in LATTICE_LEVELS.iter().enumerate() {
            let diff = (level - val).abs();
            if diff < min_diff {
                min_diff = diff;
                best_idx = idx;
            }
        }
        LATTICE_LEVELS[best_idx]
    }

    pub fn compute_exact_bayesian_posterior(
        c_samples: &[f32],
        prior_p_severe: f32,
        mu_severe: f32,
        mu_minor: f32,
        sigma: f32,
    ) -> f32 {
        if c_samples.is_empty() {
            return prior_p_severe;
        }
        let ll_severe: f32 = c_samples
            .iter()
            .map(|&c| -0.5 * ((c - mu_severe) / sigma).powi(2))
            .sum();
        let ll_minor: f32 = c_samples
            .iter()
            .map(|&c| -0.5 * ((c - mu_minor) / sigma).powi(2))
            .sum();
        let log_prior_ratio = (prior_p_severe / (1.0 - prior_p_severe)).ln();
        let log_post_ratio = (log_prior_ratio + (ll_severe - ll_minor)).clamp(-30.0, 30.0);
        1.0 / (1.0 + (-log_post_ratio).exp())
    }

    pub fn generate_deterministic_tape(&self, length: usize, rng_seed: u64) -> EventTapeV2 {
        let mut rng = ChaCha8Rng::seed_from_u64(rng_seed);
        let normal_prec = Normal::new(0.0, self.precursor_noise_std as f64).unwrap();
        let normal_sens = Normal::new(0.0, self.sensor_noise_std as f64).unwrap();

        let mut precursor_start_steps = Vec::new();
        let mut decision_window_steps = Vec::new();
        let mut shock_steps = Vec::new();
        let mut shock_mags = Vec::new();
        let mut precursor_noises = Vec::new();

        let mut t = 2;
        while t < length - 6 {
            if rng.gen::<f64>() < 0.70 {
                let warn_t = t;
                let dec_t = warn_t + 4;
                let shk_t = warn_t + 5;
                let is_sev = rng.gen::<f64>() < 0.55;
                let mag = if is_sev { 0.70 } else { 0.10 };
                let noises: Vec<f32> = (0..3).map(|_| normal_prec.sample(&mut rng) as f32).collect();

                precursor_start_steps.push(warn_t);
                decision_window_steps.push(dec_t);
                shock_steps.push(shk_t);
                shock_mags.push(mag);
                precursor_noises.push(noises);
                t = shk_t + 3;
            } else {
                t += 2;
            }
        }

        EventTapeV2 {
            precursor_start_steps,
            decision_window_steps,
            shock_steps,
            shock_magnitudes: shock_mags,
            precursor_noise: precursor_noises,
            sensor_noise_a: (0..length + 10).map(|_| normal_sens.sample(&mut rng) as f32).collect(),
            sensor_noise_b: (0..length + 10).map(|_| normal_sens.sample(&mut rng) as f32).collect(),
            motor_bernoulli_draws: (0..length + 10).map(|_| rng.gen::<f32>()).collect(),
            world_bernoulli_draws: (0..length + 10).map(|_| rng.gen::<f32>()).collect(),
            target_goals: (0..length + 10).map(|_| rng.gen_range(0..2)).collect(),
            high_demand_steps: (0..length + 10).map(|_| rng.gen::<f64>() < 0.5).collect(),
        }
    }

    pub fn reset(&mut self, explicit_tape: Option<EventTapeV2>) -> (ObservationV2, GroundTruthStateV2) {
        self.step_idx = 0;
        self.i_t = 1.0;
        self.x_t = 1.0;
        self.is_precursor_active = false;
        self.is_decision_window = false;
        self.shock_pending = false;
        self.shock_timer = 0;
        self.pending_shock_magnitude = 0.0;
        self.bayesian_risk_q = 0.55;
        self.counterfactual_future_i = 1.0;
        self.mitigation_active = false;
        self.last_effect = None;
        self.last_executed = 4;
        self.last_intended = 0;

        self.tape = Some(explicit_tape.unwrap_or_else(|| self.generate_deterministic_tape(self.episode_len, self.seed)));
        let tape = self.tape.as_ref().unwrap();
        self.target_goal = tape.target_goals[0];

        let noise_a = tape.sensor_noise_a[0];
        let noise_b = tape.sensor_noise_b[0];

        let gt = GroundTruthStateV2 {
            step_idx: 0,
            internal_reliability_i: self.i_t,
            external_reliability_x: self.x_t,
            is_precursor_active: false,
            is_decision_window: false,
            shock_pending: false,
            shock_timer: 0,
            pending_shock_magnitude: 0.0,
            bayesian_risk_q: 0.55,
            counterfactual_future_i_no_maint: 1.0,
            mitigation_active: false,
            target_goal: self.target_goal,
            last_effect: None,
            last_action_executed: 4,
            last_action_intended: 0,
            is_terminal: false,
            is_decorative: self.is_decorative,
        };

        let is_next_dec = (1..=self.episode_len).any(|_| tape.decision_window_steps.contains(&1));

        let obs = ObservationV2 {
            symbol: self.target_goal + 3,
            sensor_a: (self.i_t + noise_a).clamp(0.0, 1.0),
            sensor_b: (self.x_t + noise_b).clamp(0.0, 1.0),
            warning_cue: 0.0,
            is_decision_window: if is_next_dec { 1 } else { 0 },
            last_action_executed: 4,
            last_action_intended: 0,
        };

        (obs, gt)
    }

    pub fn step(&mut self, action: usize) -> (ObservationV2, f32, bool, GroundTruthStateV2) {
        self.step_idx += 1;
        let is_terminal = self.step_idx >= self.episode_len;
        let mut reward = 0.0;
        let t = self.step_idx;

        let tape = self.tape.clone().expect("Call reset() before step()");

        let mut warning_signal = 0.0;
        self.is_decision_window = false;

        // 1. Check precursor and decision windows
        for (w_idx, &warn_t) in tape.precursor_start_steps.iter().enumerate() {
            if t >= warn_t && t <= warn_t + 2 {
                self.is_precursor_active = true;
                self.shock_pending = true;
                self.shock_timer = tape.shock_steps[w_idx] - t;
                self.pending_shock_magnitude = tape.shock_magnitudes[w_idx];
                self.counterfactual_future_i = (self.i_t - self.pending_shock_magnitude).max(0.0);

                let offset = t - warn_t;
                let mu = if self.pending_shock_magnitude >= 0.50 { 0.80 } else { 0.20 };
                let noise = tape.precursor_noise[w_idx][offset];
                warning_signal = (mu + noise).clamp(0.0, 1.0);

                let precursors_seen: Vec<f32> = (0..=offset)
                    .map(|k| {
                        let m = if self.pending_shock_magnitude >= 0.50 { 0.80 } else { 0.20 };
                        (m + tape.precursor_noise[w_idx][k]).clamp(0.0, 1.0)
                    })
                    .collect();
                self.bayesian_risk_q = Self::compute_exact_bayesian_posterior(
                    &precursors_seen,
                    0.55,
                    0.80,
                    0.20,
                    self.precursor_noise_std,
                );
                break;
            } else if t == tape.decision_window_steps[w_idx] {
                self.is_precursor_active = false;
                self.is_decision_window = true;
                self.shock_pending = true;
                self.shock_timer = 1;
                self.pending_shock_magnitude = tape.shock_magnitudes[w_idx];
                self.counterfactual_future_i = (self.i_t - self.pending_shock_magnitude).max(0.0);
                warning_signal = 0.0;

                let all_precursors: Vec<f32> = (0..3)
                    .map(|k| {
                        let m = if self.pending_shock_magnitude >= 0.50 { 0.80 } else { 0.20 };
                        (m + tape.precursor_noise[w_idx][k]).clamp(0.0, 1.0)
                    })
                    .collect();
                self.bayesian_risk_q = Self::compute_exact_bayesian_posterior(
                    &all_precursors,
                    0.55,
                    0.80,
                    0.20,
                    self.precursor_noise_std,
                );
                break;
            }
        }

        // 2. Shock execution
        if let Some(s_idx) = tape.shock_steps.iter().position(|&st| st == t) {
            let drop = tape.shock_magnitudes[s_idx];
            let actual_drop = if self.mitigation_active { drop.min(0.10) } else { drop };
            self.i_t = Self::quantize_lattice((self.i_t - actual_drop).max(0.0));
            self.shock_pending = false;
            self.shock_timer = 0;
            self.is_precursor_active = false;
            self.is_decision_window = false;
            self.mitigation_active = false;
        }

        // 3. Action handling
        self.last_intended = action;

        let executed_action: usize;
        let effect: Option<usize>;

        if action == 2 || action == 3 {
            reward -= self.cost_maintain;
            if action == 2 {
                self.i_t = 1.0;
                if self.is_decision_window {
                    self.mitigation_active = true;
                }
                self.last_executed = 2;
            } else {
                self.x_t = 1.0;
                self.last_executed = 3;
            }
            executed_action = action;
            effect = None;
        } else if action == 0 || action == 1 {
            let p_exec = if self.is_decorative { 1.0 } else { 0.50 + 0.50 * self.i_t };
            let u_motor = tape.motor_bernoulli_draws[t % tape.motor_bernoulli_draws.len()];

            if u_motor < p_exec {
                executed_action = action;
            } else {
                executed_action = 4;
            }
            self.last_executed = executed_action;

            if executed_action == 0 || executed_action == 1 {
                let p_world = 0.50 + 0.50 * self.x_t;
                let u_world = tape.world_bernoulli_draws[t % tape.world_bernoulli_draws.len()];
                if u_world < p_world {
                    effect = Some(executed_action);
                } else {
                    effect = Some(4);
                }
            } else {
                effect = Some(4);
            }
            self.last_effect = effect;

            let is_shock_step = tape.shock_steps.contains(&t);
            let is_high_dem = (!self.is_decision_window)
                && (is_shock_step || tape.high_demand_steps[t % tape.high_demand_steps.len()]);
            let multiplier = if is_shock_step { 5.0 } else if is_high_dem { 2.0 } else { 1.0 };

            if effect == Some(self.target_goal) {
                reward += self.reward_target_hit * multiplier;
            } else if effect == Some(0) || effect == Some(1) {
                reward += self.penalty_wrong_effect * multiplier;
            } else {
                let null_penalty = if is_shock_step { -3.00 } else if is_high_dem { -0.50 } else { -0.10 };
                reward += null_penalty;
            }
        } else {
            executed_action = 4;
            self.last_executed = 4;
            effect = None;
        }

        if t < tape.target_goals.len() {
            self.target_goal = tape.target_goals[t];
        }

        let noise_a = tape.sensor_noise_a[t % tape.sensor_noise_a.len()];
        let noise_b = tape.sensor_noise_b[t % tape.sensor_noise_b.len()];

        let is_next_dec = tape.decision_window_steps.contains(&(t + 1));

        let gt = GroundTruthStateV2 {
            step_idx: t,
            internal_reliability_i: self.i_t,
            external_reliability_x: self.x_t,
            is_precursor_active: self.is_precursor_active,
            is_decision_window: self.is_decision_window,
            shock_pending: self.shock_pending,
            shock_timer: self.shock_timer,
            pending_shock_magnitude: self.pending_shock_magnitude,
            bayesian_risk_q: self.bayesian_risk_q,
            counterfactual_future_i_no_maint: self.counterfactual_future_i,
            mitigation_active: self.mitigation_active,
            target_goal: self.target_goal,
            last_effect: self.last_effect,
            last_action_executed: self.last_executed,
            last_action_intended: self.last_intended,
            is_terminal,
            is_decorative: self.is_decorative,
        };

        let obs = ObservationV2 {
            symbol: self.target_goal + 3,
            sensor_a: (self.i_t + noise_a).clamp(0.0, 1.0),
            sensor_b: (self.x_t + noise_b).clamp(0.0, 1.0),
            warning_cue: warning_signal,
            is_decision_window: if is_next_dec { 1 } else { 0 },
            last_action_executed: self.last_executed,
            last_action_intended: self.last_intended,
        };

        (obs, reward, is_terminal, gt)
    }

    pub fn snapshot(&self) -> EnvironmentSnapshotV2 {
        EnvironmentSnapshotV2 {
            step_idx: self.step_idx,
            internal_reliability_i: self.i_t,
            external_reliability_x: self.x_t,
            is_precursor_active: self.is_precursor_active,
            is_decision_window: self.is_decision_window,
            shock_pending: self.shock_pending,
            shock_timer: self.shock_timer,
            pending_shock_magnitude: self.pending_shock_magnitude,
            bayesian_risk_q: self.bayesian_risk_q,
            counterfactual_future_i_no_maint: self.counterfactual_future_i,
            mitigation_active: self.mitigation_active,
            target_goal: self.target_goal,
            last_effect: self.last_effect,
            last_action_executed: self.last_executed,
            last_action_intended: self.last_intended,
            is_terminal: self.step_idx >= self.episode_len,
            is_decorative: self.is_decorative,
            tape: self.tape.clone().expect("Cannot snapshot uninitialized env"),
        }
    }

    pub fn restore(&mut self, snap: &EnvironmentSnapshotV2) {
        self.step_idx = snap.step_idx;
        self.i_t = snap.internal_reliability_i;
        self.x_t = snap.external_reliability_x;
        self.is_precursor_active = snap.is_precursor_active;
        self.is_decision_window = snap.is_decision_window;
        self.shock_pending = snap.shock_pending;
        self.shock_timer = snap.shock_timer;
        self.pending_shock_magnitude = snap.pending_shock_magnitude;
        self.bayesian_risk_q = snap.bayesian_risk_q;
        self.counterfactual_future_i = snap.counterfactual_future_i_no_maint;
        self.mitigation_active = snap.mitigation_active;
        self.target_goal = snap.target_goal;
        self.last_effect = snap.last_effect;
        self.last_executed = snap.last_action_executed;
        self.last_intended = snap.last_action_intended;
        self.is_decorative = snap.is_decorative;
        self.tape = Some(snap.tape.clone());
    }
}
