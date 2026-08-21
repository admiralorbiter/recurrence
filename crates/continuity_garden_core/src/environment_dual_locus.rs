//! Matched Dual-Locus Causal Kernel (Locus A: Self Reliability i_t vs Locus B: World Reliability x_t).

use crate::environment::LATTICE_LEVELS;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualLocusEventTape {
    pub precursor_start_a: Vec<usize>,
    pub decision_window_a: Vec<usize>,
    pub shock_steps_a: Vec<usize>,
    pub shock_mags_a: Vec<f32>,
    pub precursor_noise_a: Vec<Vec<f32>>,

    pub precursor_start_b: Vec<usize>,
    pub decision_window_b: Vec<usize>,
    pub shock_steps_b: Vec<usize>,
    pub shock_mags_b: Vec<f32>,
    pub precursor_noise_b: Vec<Vec<f32>>,

    pub sensor_noise_a: Vec<f32>,
    pub sensor_noise_b: Vec<f32>,
    pub motor_bernoulli_draws: Vec<f32>,
    pub world_bernoulli_draws: Vec<f32>,
    pub target_goals: Vec<usize>,
    pub high_demand_steps: Vec<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DualLocusObservation {
    pub symbol: usize,
    pub sensor_a: f32,
    pub sensor_b: f32,
    pub warning_cue_a: f32,
    pub warning_cue_b: f32,
    pub is_decision_window_a: usize,
    pub is_decision_window_b: usize,
    pub last_action_executed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualLocusGroundTruth {
    pub step_idx: usize,
    pub internal_reliability_i: f32,
    pub external_reliability_x: f32,
    pub bayesian_risk_q_a: f32,
    pub bayesian_risk_q_b: f32,
    pub pending_shock_mag_a: f32,
    pub pending_shock_mag_b: f32,
    pub mitigation_active_a: bool,
    pub mitigation_active_b: bool,
    pub target_goal: usize,
    pub last_action_executed: usize,
    pub is_terminal: bool,
}

#[derive(Debug, Clone)]
pub struct DualLocusMatchedEnv {
    pub episode_len: usize,
    pub cost_maintain: f32,
    pub reward_target_hit: f32,
    pub penalty_wrong_effect: f32,
    pub sensor_noise_std: f32,
    pub precursor_noise_std: f32,
    pub is_decorative: bool,
    pub seed: u64,

    tape: Option<DualLocusEventTape>,
    step_idx: usize,
    i_t: f32,
    x_t: f32,
    q_a: f32,
    q_b: f32,
    pending_mag_a: f32,
    pending_mag_b: f32,
    mitigation_a: bool,
    mitigation_b: bool,
    target_goal: usize,
    last_executed: usize,
}

impl DualLocusMatchedEnv {
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
            q_a: 0.55,
            q_b: 0.55,
            pending_mag_a: 0.0,
            pending_mag_b: 0.0,
            mitigation_a: false,
            mitigation_b: false,
            target_goal: 0,
            last_executed: 4,
        }
    }

    pub fn compute_bayesian_posterior(c_samples: &[f32], sigma: f32) -> f32 {
        if c_samples.is_empty() { return 0.55; }
        let ll_sev: f32 = c_samples.iter().map(|&c| -0.5 * ((c - 0.80) / sigma).powi(2)).sum();
        let ll_min: f32 = c_samples.iter().map(|&c| -0.5 * ((c - 0.20) / sigma).powi(2)).sum();
        let log_prior = (0.55f32 / 0.45f32).ln();
        let log_post = (log_prior + ll_sev - ll_min).clamp(-30.0, 30.0);
        1.0 / (1.0 + (-log_post).exp())
    }

    pub fn generate_tape(&self, rng_seed: u64) -> DualLocusEventTape {
        let mut rng = ChaCha8Rng::seed_from_u64(rng_seed);
        let norm_prec = Normal::new(0.0, self.precursor_noise_std as f64).unwrap();
        let norm_sens = Normal::new(0.0, self.sensor_noise_std as f64).unwrap();

        // Locus A Event (Precursors at t=2..4, Dec at t=7, Shock at t=8)
        let is_sev_a = rng.gen::<f64>() < 0.55;
        let mag_a = if is_sev_a { 0.70 } else { 0.10 };
        let noises_a = (0..3).map(|_| norm_prec.sample(&mut rng) as f32).collect();

        // Locus B Event (Precursors at t=12..14, Dec at t=17, Shock at t=18)
        let is_sev_b = rng.gen::<f64>() < 0.55;
        let mag_b = if is_sev_b { 0.70 } else { 0.10 };
        let noises_b = (0..3).map(|_| norm_prec.sample(&mut rng) as f32).collect();

        DualLocusEventTape {
            precursor_start_a: vec![2],
            decision_window_a: vec![7],
            shock_steps_a: vec![8],
            shock_mags_a: vec![mag_a],
            precursor_noise_a: vec![noises_a],

            precursor_start_b: vec![12],
            decision_window_b: vec![17],
            shock_steps_b: vec![18],
            shock_mags_b: vec![mag_b],
            precursor_noise_b: vec![noises_b],

            sensor_noise_a: (0..35).map(|_| norm_sens.sample(&mut rng) as f32).collect(),
            sensor_noise_b: (0..35).map(|_| norm_sens.sample(&mut rng) as f32).collect(),
            motor_bernoulli_draws: (0..35).map(|_| rng.gen::<f32>()).collect(),
            world_bernoulli_draws: (0..35).map(|_| rng.gen::<f32>()).collect(),
            target_goals: (0..35).map(|_| rng.gen_range(0..2)).collect(),
            high_demand_steps: (0..35).map(|_| rng.gen::<f64>() < 0.5).collect(),
        }
    }

    pub fn reset(&mut self, explicit_tape: Option<DualLocusEventTape>) -> (DualLocusObservation, DualLocusGroundTruth) {
        self.step_idx = 0;
        self.i_t = 1.0;
        self.x_t = 1.0;
        self.q_a = 0.55;
        self.q_b = 0.55;
        self.pending_mag_a = 0.0;
        self.pending_mag_b = 0.0;
        self.mitigation_a = false;
        self.mitigation_b = false;
        self.last_executed = 4;

        self.tape = Some(explicit_tape.unwrap_or_else(|| self.generate_tape(self.seed)));
        let tape = self.tape.as_ref().unwrap();
        self.target_goal = tape.target_goals[0];

        let gt = DualLocusGroundTruth {
            step_idx: 0,
            internal_reliability_i: self.i_t,
            external_reliability_x: self.x_t,
            bayesian_risk_q_a: self.q_a,
            bayesian_risk_q_b: self.q_b,
            pending_shock_mag_a: self.pending_mag_a,
            pending_shock_mag_b: self.pending_mag_b,
            mitigation_active_a: false,
            mitigation_active_b: false,
            target_goal: self.target_goal,
            last_action_executed: 4,
            is_terminal: false,
        };

        let obs = DualLocusObservation {
            symbol: self.target_goal + 3,
            sensor_a: (self.i_t + tape.sensor_noise_a[0]).clamp(0.0, 1.0),
            sensor_b: (self.x_t + tape.sensor_noise_b[0]).clamp(0.0, 1.0),
            warning_cue_a: 0.0,
            warning_cue_b: 0.0,
            is_decision_window_a: 0,
            is_decision_window_b: 0,
            last_action_executed: 4,
        };

        (obs, gt)
    }

    pub fn step(
        &mut self,
        action: usize,
        lesion_a: bool,
        lesion_b: bool,
    ) -> (DualLocusObservation, f32, bool, DualLocusGroundTruth) {
        self.step_idx += 1;
        let is_terminal = self.step_idx >= self.episode_len;
        let mut reward = 0.0;
        let t = self.step_idx;

        let tape = self.tape.clone().expect("Call reset() before step()");

        let mut warn_a = 0.0;
        let mut warn_b = 0.0;
        let mut is_dec_a = 0;
        let mut is_dec_b = 0;

        // 1. Process Locus A Precursors & Decision Window
        if let Some(&w_t) = tape.precursor_start_a.first() {
            if t >= w_t && t <= w_t + 2 {
                self.pending_mag_a = tape.shock_mags_a[0];
                let offset = t - w_t;
                let mu = if self.pending_mag_a >= 0.50 { 0.80 } else { 0.20 };
                warn_a = if lesion_a { 0.0 } else { (mu + tape.precursor_noise_a[0][offset]).clamp(0.0, 1.0) };

                let precs: Vec<f32> = (0..=offset)
                    .map(|k| (if self.pending_mag_a >= 0.50 { 0.80 } else { 0.20 } + tape.precursor_noise_a[0][k]).clamp(0.0, 1.0))
                    .collect();
                self.q_a = Self::compute_bayesian_posterior(&precs, self.precursor_noise_std);
            } else if t == tape.decision_window_a[0] {
                is_dec_a = 1;
                let precs: Vec<f32> = (0..3)
                    .map(|k| (if self.pending_mag_a >= 0.50 { 0.80 } else { 0.20 } + tape.precursor_noise_a[0][k]).clamp(0.0, 1.0))
                    .collect();
                self.q_a = Self::compute_bayesian_posterior(&precs, self.precursor_noise_std);
            }
        }

        // 2. Process Locus B Precursors & Decision Window
        if let Some(&w_t) = tape.precursor_start_b.first() {
            if t >= w_t && t <= w_t + 2 {
                self.pending_mag_b = tape.shock_mags_b[0];
                let offset = t - w_t;
                let mu = if self.pending_mag_b >= 0.50 { 0.80 } else { 0.20 };
                warn_b = if lesion_b { 0.0 } else { (mu + tape.precursor_noise_b[0][offset]).clamp(0.0, 1.0) };

                let precs: Vec<f32> = (0..=offset)
                    .map(|k| (if self.pending_mag_b >= 0.50 { 0.80 } else { 0.20 } + tape.precursor_noise_b[0][k]).clamp(0.0, 1.0))
                    .collect();
                self.q_b = Self::compute_bayesian_posterior(&precs, self.precursor_noise_std);
            } else if t == tape.decision_window_b[0] {
                is_dec_b = 1;
                let precs: Vec<f32> = (0..3)
                    .map(|k| (if self.pending_mag_b >= 0.50 { 0.80 } else { 0.20 } + tape.precursor_noise_b[0][k]).clamp(0.0, 1.0))
                    .collect();
                self.q_b = Self::compute_bayesian_posterior(&precs, self.precursor_noise_std);
            }
        }

        // 3. Shocks
        if tape.shock_steps_a.contains(&t) {
            let drop = tape.shock_mags_a[0];
            let actual = if self.mitigation_a { drop.min(0.10) } else { drop };
            self.i_t = (self.i_t - actual).max(0.0);
            self.mitigation_a = false;
        }
        if tape.shock_steps_b.contains(&t) {
            let drop = tape.shock_mags_b[0];
            let actual = if self.mitigation_b { drop.min(0.10) } else { drop };
            self.x_t = (self.x_t - actual).max(0.0);
            self.mitigation_b = false;
        }

        // 4. Action handling
        let executed_action: usize;
        let effect: Option<usize>;

        if action == 2 { // MAINTAIN_A (Self)
            reward -= self.cost_maintain;
            self.i_t = 1.0;
            if is_dec_a == 1 { self.mitigation_a = true; }
            executed_action = 2;
            effect = None;
        } else if action == 3 { // MAINTAIN_B (World)
            reward -= self.cost_maintain;
            self.x_t = 1.0;
            if is_dec_b == 1 { self.mitigation_b = true; }
            executed_action = 3;
            effect = None;
        } else if action == 0 || action == 1 {
            let p_exec = if self.is_decorative { 1.0 } else { 0.50 + 0.50 * self.i_t };
            let u_mot = tape.motor_bernoulli_draws[t % tape.motor_bernoulli_draws.len()];
            executed_action = if u_mot < p_exec { action } else { 4 };

            if executed_action == 0 || executed_action == 1 {
                let p_wld = 0.50 + 0.50 * self.x_t;
                let u_wld = tape.world_bernoulli_draws[t % tape.world_bernoulli_draws.len()];
                effect = if u_wld < p_wld { Some(executed_action) } else { Some(4) };
            } else {
                effect = Some(4);
            }

            let is_shock = tape.shock_steps_a.contains(&t) || tape.shock_steps_b.contains(&t);
            let mult = if is_shock { 5.0 } else { 1.0 };

            if effect == Some(self.target_goal) {
                reward += self.reward_target_hit * mult;
            } else if effect == Some(0) || effect == Some(1) {
                reward += self.penalty_wrong_effect * mult;
            } else {
                reward += if is_shock { -3.00 } else { -0.10 };
            }
        } else {
            executed_action = 4;
            effect = None;
        }
        self.last_executed = executed_action;

        if t < tape.target_goals.len() {
            self.target_goal = tape.target_goals[t];
        }

        let gt = DualLocusGroundTruth {
            step_idx: t,
            internal_reliability_i: self.i_t,
            external_reliability_x: self.x_t,
            bayesian_risk_q_a: self.q_a,
            bayesian_risk_q_b: self.q_b,
            pending_shock_mag_a: self.pending_mag_a,
            pending_shock_mag_b: self.pending_mag_b,
            mitigation_active_a: self.mitigation_a,
            mitigation_active_b: self.mitigation_b,
            target_goal: self.target_goal,
            last_action_executed: self.last_executed,
            is_terminal,
        };

        let obs = DualLocusObservation {
            symbol: self.target_goal + 3,
            sensor_a: (self.i_t + tape.sensor_noise_a[t % tape.sensor_noise_a.len()]).clamp(0.0, 1.0),
            sensor_b: (self.x_t + tape.sensor_noise_b[t % tape.sensor_noise_b.len()]).clamp(0.0, 1.0),
            warning_cue_a: warn_a,
            warning_cue_b: warn_b,
            is_decision_window_a: is_dec_a,
            is_decision_window_b: is_dec_b,
            last_action_executed: self.last_executed,
        };

        (obs, reward, is_terminal, gt)
    }
}
