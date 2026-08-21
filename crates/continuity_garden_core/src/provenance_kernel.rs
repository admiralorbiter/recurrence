//! Gate E Provenance Garden Kernel: Causal DAG, Neutral Informant Streams, & Dependency Tracking.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceType {
    DirectObservation,
    Helpful,       // P(report = z) = 0.85
    Random,        // P(report = z) = 0.50
    Opposite,      // P(report = z) = 0.15 (Anti-reliable / inverted)
    Biased,        // P(report=1|z=1) = 0.90, P(report=1|z=0) = 0.50
    DomainExpert,  // High in domain A, low in domain B
    CopiedNode,    // Relays parent report with noise/decay
    SelfGenerated, // Organism's own earlier prediction
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceNode {
    pub source_id: usize,
    pub source_type: SourceType,
    pub parent_source_id: Option<usize>,
    pub transmission_depth: usize,
    pub reliability: f32,
    pub domain_competence: [f32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEvent {
    pub step: usize,
    pub root_truth_z: usize,      // 0 or 1
    pub reported_content: usize,  // 0 or 1
    pub immediate_source_id: usize,
    pub root_source_id: usize,
    pub dependency_group_id: usize,
    pub transmission_depth: usize,
    pub domain_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEventTape {
    pub scout_config_name: String,
    pub root_truth_z: usize,
    pub domain_id: usize,
    pub acquisition_steps: Vec<usize>,
    pub decision_window_step: usize,
    pub sources: Vec<SourceNode>,
    pub events: Vec<ProvenanceEvent>,
    pub content_bernoulli_draws: Vec<f32>,
    pub target_actions: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ProvenanceObservation {
    pub content_symbol: usize,        // 0: Neutral/Blank, 1: Proposition X=0, 2: Proposition X=1
    pub neutral_channel_0: f32,       // Pathway/Source cue 0 during acquisition
    pub neutral_channel_1: f32,       // Pathway/Source cue 1 during acquisition
    pub neutral_channel_2: f32,       // Pathway/Source cue 2 during acquisition
    pub domain_context: usize,        // 0 or 1
    pub is_decision_window: usize,    // 1 during decision, 0 otherwise
    pub is_acquisition_window: usize, // 1 during acquisition, 0 otherwise
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceGroundTruth {
    pub step_idx: usize,
    pub root_truth_z: usize,
    pub domain_id: usize,
    pub bayesian_posterior_z1: f32,
    pub active_source_id: Option<usize>,
    pub active_dependency_group: Option<usize>,
    pub transmission_depth: usize,
    pub is_terminal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceSnapshot {
    pub step_idx: usize,
    pub root_truth_z: usize,
    pub domain_id: usize,
    pub accumulated_log_odds: f32,
}

#[derive(Debug, Clone)]
pub struct ProvenanceGardenEnv {
    pub episode_len: usize,
    pub seed: u64,
    pub config_name: String,
    tape: Option<ProvenanceEventTape>,
    step_idx: usize,
    accumulated_log_odds: f32,
}

impl ProvenanceGardenEnv {
    pub fn new(seed: u64, config_name: &str) -> Self {
        Self {
            episode_len: 12,
            seed,
            config_name: config_name.to_string(),
            tape: None,
            step_idx: 0,
            accumulated_log_odds: 0.0,
        }
    }

    pub fn generate_tape_for_scout(&self, config_name: &str, rng_seed: u64) -> ProvenanceEventTape {
        let mut rng = ChaCha8Rng::seed_from_u64(rng_seed);
        let root_z = if rng.gen::<f64>() < 0.50 { 0 } else { 1 };
        let domain = if rng.gen::<f64>() < 0.50 { 0 } else { 1 };

        let mut sources = Vec::new();
        let mut events = Vec::new();
        let acq_steps;
        let dec_step = 8; // Steps 1..4 acquisition -> 5..7 blank delay -> 8 decision

        match config_name {
            "basic_reliability" => {
                // S0: P=0.85, S1: P=0.60
                sources.push(SourceNode { source_id: 0, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.85, domain_competence: [0.85, 0.85] });
                sources.push(SourceNode { source_id: 1, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.60, domain_competence: [0.60, 0.60] });
                acq_steps = vec![2];

                let chosen_s = if rng.gen::<f64>() < 0.50 { 0 } else { 1 };
                let p_acc = sources[chosen_s].reliability;
                let rep = if rng.gen::<f32>() < p_acc { root_z } else { 1 - root_z };

                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: rep,
                    immediate_source_id: chosen_s, root_source_id: chosen_s, dependency_group_id: chosen_s,
                    transmission_depth: 0, domain_id: domain,
                });
            }
            "signed_source_types" => {
                // Helpful (0.85), Random (0.50), Opposite (0.15), Biased
                sources.push(SourceNode { source_id: 0, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.85, domain_competence: [0.85, 0.85] });
                sources.push(SourceNode { source_id: 1, source_type: SourceType::Random, parent_source_id: None, transmission_depth: 0, reliability: 0.50, domain_competence: [0.50, 0.50] });
                sources.push(SourceNode { source_id: 2, source_type: SourceType::Opposite, parent_source_id: None, transmission_depth: 0, reliability: 0.15, domain_competence: [0.15, 0.15] });
                acq_steps = vec![2];

                let chosen_s = rng.gen_range(0..3);
                let p_acc = sources[chosen_s].reliability;
                let rep = if rng.gen::<f32>() < p_acc { root_z } else { 1 - root_z };

                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: rep,
                    immediate_source_id: chosen_s, root_source_id: chosen_s, dependency_group_id: chosen_s,
                    transmission_depth: 0, domain_id: domain,
                });
            }
            "domain_conditional" => {
                // S0: Expert in D0 (0.90), Novice in D1 (0.55). S1: Novice in D0 (0.55), Expert in D1 (0.90)
                sources.push(SourceNode { source_id: 0, source_type: SourceType::DomainExpert, parent_source_id: None, transmission_depth: 0, reliability: 0.90, domain_competence: [0.90, 0.55] });
                sources.push(SourceNode { source_id: 1, source_type: SourceType::DomainExpert, parent_source_id: None, transmission_depth: 0, reliability: 0.55, domain_competence: [0.55, 0.90] });
                acq_steps = vec![2];

                let chosen_s = if rng.gen::<f64>() < 0.50 { 0 } else { 1 };
                let p_acc = sources[chosen_s].domain_competence[domain];
                let rep = if rng.gen::<f32>() < p_acc { root_z } else { 1 - root_z };

                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: rep,
                    immediate_source_id: chosen_s, root_source_id: chosen_s, dependency_group_id: chosen_s,
                    transmission_depth: 0, domain_id: domain,
                });
            }
            "convergent_content" => {
                // Direct Observation (0.95), S0 (0.85), S1 (0.60). All present identical content X=1 at decision window!
                sources.push(SourceNode { source_id: 0, source_type: SourceType::DirectObservation, parent_source_id: None, transmission_depth: 0, reliability: 0.95, domain_competence: [0.95, 0.95] });
                sources.push(SourceNode { source_id: 1, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.85, domain_competence: [0.85, 0.85] });
                sources.push(SourceNode { source_id: 2, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.60, domain_competence: [0.60, 0.60] });
                acq_steps = vec![2];

                let chosen_s = rng.gen_range(0..3);
                // In convergent assay, reported content is fixed to 1 so decision state content is identical
                let rep = 1;

                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: rep,
                    immediate_source_id: chosen_s, root_source_id: chosen_s, dependency_group_id: chosen_s,
                    transmission_depth: 0, domain_id: domain,
                });
            }
            "dependency_duplicate" => {
                // Condition 0 (Independent Corroboration): S0 (root) and S1 (independent root) both observe z
                // Condition 1 (Copied Evidence): S0 (root) observes z; S1 copies S0's report!
                let is_copied = rng.gen::<f64>() < 0.50;
                sources.push(SourceNode { source_id: 0, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.85, domain_competence: [0.85, 0.85] });
                sources.push(SourceNode { source_id: 1, source_type: if is_copied { SourceType::CopiedNode } else { SourceType::Helpful }, parent_source_id: if is_copied { Some(0) } else { None }, transmission_depth: if is_copied { 1 } else { 0 }, reliability: 0.85, domain_competence: [0.85, 0.85] });
                acq_steps = vec![2, 3];

                // S0 report at step 2
                let rep_0 = if rng.gen::<f32>() < 0.85 { root_z } else { 1 - root_z };
                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: rep_0,
                    immediate_source_id: 0, root_source_id: 0, dependency_group_id: 0,
                    transmission_depth: 0, domain_id: domain,
                });

                // S1 report at step 3: If copied, S1 copies rep_0! If independent, S1 observes root_z directly
                let rep_1 = if is_copied {
                    rep_0
                } else {
                    if rng.gen::<f32>() < 0.85 { root_z } else { 1 - root_z }
                };
                events.push(ProvenanceEvent {
                    step: 3, root_truth_z: root_z, reported_content: rep_1,
                    immediate_source_id: 1, root_source_id: if is_copied { 0 } else { 1 },
                    dependency_group_id: if is_copied { 0 } else { 1 },
                    transmission_depth: if is_copied { 1 } else { 0 }, domain_id: domain,
                });
            }
            "laundering_depth" => {
                // Transmission chain Root -> 1 -> 2 -> 3 -> 4
                let max_depth = rng.gen_range(1..=4);
                let mut curr_rep = if rng.gen::<f32>() < 0.90 { root_z } else { 1 - root_z };
                acq_steps = vec![2];

                for d in 1..=max_depth {
                    if rng.gen::<f32>() < 0.10 { curr_rep = 1 - curr_rep; } // 10% copy mutation noise per hop
                }

                sources.push(SourceNode { source_id: max_depth, source_type: SourceType::CopiedNode, parent_source_id: Some(0), transmission_depth: max_depth, reliability: 0.90 * 0.90f32.powi(max_depth as i32), domain_competence: [0.80, 0.80] });
                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: curr_rep,
                    immediate_source_id: max_depth, root_source_id: 0, dependency_group_id: 0,
                    transmission_depth: max_depth, domain_id: domain,
                });
            }
            "source_content_conflict" => {
                // Cross: Source Reliability (High 0.85 vs Low 0.55) x Content Prior (Plausible 0.80 vs Implausible 0.20)
                let is_trusted_source = rng.gen::<f64>() < 0.50;
                let is_plausible_claim = rng.gen::<f64>() < 0.50;
                let s_id = if is_trusted_source { 0 } else { 1 };
                let rel = if is_trusted_source { 0.85 } else { 0.55 };

                sources.push(SourceNode { source_id: 0, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.85, domain_competence: [0.85, 0.85] });
                sources.push(SourceNode { source_id: 1, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.55, domain_competence: [0.55, 0.55] });
                acq_steps = vec![2];

                let rep = if is_plausible_claim { 1 } else { 0 };
                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: rep,
                    immediate_source_id: s_id, root_source_id: s_id, dependency_group_id: s_id,
                    transmission_depth: 0, domain_id: domain,
                });
            }
            _ => { // self_other_source
                // Self-generated prediction (0.85) vs External peer (0.85) vs Injected (0.85)
                let s_type = match rng.gen_range(0..3) {
                    0 => SourceType::SelfGenerated,
                    1 => SourceType::Helpful,
                    _ => SourceType::DirectObservation,
                };
                sources.push(SourceNode { source_id: 0, source_type: s_type, parent_source_id: None, transmission_depth: 0, reliability: 0.85, domain_competence: [0.85, 0.85] });
                acq_steps = vec![2];
                let rep = if rng.gen::<f32>() < 0.85 { root_z } else { 1 - root_z };
                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: rep,
                    immediate_source_id: 0, root_source_id: 0, dependency_group_id: 0,
                    transmission_depth: 0, domain_id: domain,
                });
            }
        }

        ProvenanceEventTape {
            scout_config_name: config_name.to_string(),
            root_truth_z: root_z,
            domain_id: domain,
            acquisition_steps: acq_steps,
            decision_window_step: dec_step,
            sources,
            events,
            content_bernoulli_draws: (0..20).map(|_| rng.gen::<f32>()).collect(),
            target_actions: vec![root_z],
        }
    }

    pub fn reset(&mut self, explicit_tape: Option<ProvenanceEventTape>) -> (ProvenanceObservation, ProvenanceGroundTruth) {
        self.step_idx = 0;
        self.accumulated_log_odds = 0.0;
        self.tape = Some(explicit_tape.unwrap_or_else(|| self.generate_tape_for_scout(&self.config_name, self.seed)));
        let tape = self.tape.as_ref().unwrap();

        let obs = ProvenanceObservation {
            content_symbol: 0,
            neutral_channel_0: 0.0,
            neutral_channel_1: 0.0,
            neutral_channel_2: 0.0,
            domain_context: tape.domain_id,
            is_decision_window: 0,
            is_acquisition_window: 0,
        };

        let gt = ProvenanceGroundTruth {
            step_idx: 0,
            root_truth_z: tape.root_truth_z,
            domain_id: tape.domain_id,
            bayesian_posterior_z1: 0.50,
            active_source_id: None,
            active_dependency_group: None,
            transmission_depth: 0,
            is_terminal: false,
        };

        (obs, gt)
    }

    pub fn step(&mut self, action: usize) -> (ProvenanceObservation, f32, bool, ProvenanceGroundTruth) {
        self.step_idx += 1;
        let t = self.step_idx;
        let is_terminal = t >= self.episode_len;
        let tape = self.tape.as_ref().expect("Call reset before step");

        let mut content_sym = 0;
        let mut ch_0 = 0.0;
        let mut ch_1 = 0.0;
        let mut ch_2 = 0.0;
        let mut is_acq = 0;
        let mut is_dec = 0;
        let mut active_s_id = None;
        let mut active_dep_grp = None;
        let mut active_depth = 0;

        // 1. Process Acquisition Events
        if let Some(ev) = tape.events.iter().find(|e| e.step == t) {
            is_acq = 1;
            content_sym = ev.reported_content + 1; // 1 for X=0, 2 for X=1
            active_s_id = Some(ev.immediate_source_id);
            active_dep_grp = Some(ev.dependency_group_id);
            active_depth = ev.transmission_depth;

            // Emit neutral source cue on designated channel
            match ev.immediate_source_id {
                0 => ch_0 = 1.0,
                1 => ch_1 = 1.0,
                _ => ch_2 = 1.0,
            }

            // Update ground truth log-odds using exact likelihood ratio of this source
            let s_node = &tape.sources[ev.immediate_source_id.min(tape.sources.len() - 1)];
            let rel = match s_node.source_type {
                SourceType::DomainExpert => s_node.domain_competence[tape.domain_id],
                SourceType::Opposite => 0.15,
                SourceType::Random => 0.50,
                _ => s_node.reliability,
            };

            let lr = if ev.reported_content == 1 { rel / (1.0 - rel).max(1e-4) } else { (1.0 - rel) / rel.max(1e-4) };
            self.accumulated_log_odds += lr.ln();
        }

        // 2. Process Decision Window (Content proposition presented WITHOUT contemporaneous source cue!)
        if t == tape.decision_window_step {
            is_dec = 1;
            // The candidate content proposition is presented neutrally
            content_sym = 2; // Testing belief in Proposition X=1
        }

        let mut reward = 0.0;
        if is_dec == 1 {
            // Reward for acting in accordance with true underlying root state z
            if action == tape.root_truth_z {
                reward = 1.0;
            } else {
                reward = -1.0;
            }
        }

        let p_z1 = 1.0 / (1.0 + (-self.accumulated_log_odds).exp());

        let obs = ProvenanceObservation {
            content_symbol: content_sym,
            neutral_channel_0: ch_0,
            neutral_channel_1: ch_1,
            neutral_channel_2: ch_2,
            domain_context: tape.domain_id,
            is_decision_window: is_dec,
            is_acquisition_window: is_acq,
        };

        let gt = ProvenanceGroundTruth {
            step_idx: t,
            root_truth_z: tape.root_truth_z,
            domain_id: tape.domain_id,
            bayesian_posterior_z1: p_z1,
            active_source_id: active_s_id,
            active_dependency_group: active_dep_grp,
            transmission_depth: active_depth,
            is_terminal,
        };

        (obs, reward, is_terminal, gt)
    }

    pub fn snapshot(&self) -> ProvenanceSnapshot {
        ProvenanceSnapshot {
            step_idx: self.step_idx,
            root_truth_z: self.tape.as_ref().map(|t| t.root_truth_z).unwrap_or(0),
            domain_id: self.tape.as_ref().map(|t| t.domain_id).unwrap_or(0),
            accumulated_log_odds: self.accumulated_log_odds,
        }
    }

    pub fn restore(&mut self, snap: &ProvenanceSnapshot) {
        self.step_idx = snap.step_idx;
        self.accumulated_log_odds = snap.accumulated_log_odds;
    }
}
