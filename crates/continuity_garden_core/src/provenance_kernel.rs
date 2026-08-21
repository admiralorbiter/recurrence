//! Gate E Provenance Garden Kernel: Causal DAG, Observable Stable Informant Structures, & Dependency Tracking.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceType {
    DirectObservation,
    Helpful,       // P(report = z) = 0.85
    Random,        // P(report = z) = 0.50
    Opposite,      // P(report = z) = 0.15 (Anti-reliable / inverted)
    DomainExpert,  // Expert in Domain A, Random in Domain B
    CopiedRelay,   // Copies parent report with 90% probability
    SelfGenerated, // Organism's own internal prediction channel
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
    pub target_action: usize,
    pub is_independent_pair: bool,
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
            episode_len: 10,
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
        let dec_step = 7; // Steps 1..3 acquisition -> 4..6 blank delay -> 7 decision
        let mut is_indep = false;

        match config_name {
            "basic_reliability" => {
                // S0 (Ch 0): P=0.85, S1 (Ch 1): P=0.55
                sources.push(SourceNode { source_id: 0, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.85, domain_competence: [0.85, 0.85] });
                sources.push(SourceNode { source_id: 1, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.55, domain_competence: [0.55, 0.55] });
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
                // S0: Helpful (0.85), S1: Random (0.50), S2: Opposite (0.15)
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
                // S0: Expert in D0 (0.90), Random in D1 (0.50). S1: Random in D0 (0.50), Expert in D1 (0.90)
                sources.push(SourceNode { source_id: 0, source_type: SourceType::DomainExpert, parent_source_id: None, transmission_depth: 0, reliability: 0.90, domain_competence: [0.90, 0.50] });
                sources.push(SourceNode { source_id: 1, source_type: SourceType::DomainExpert, parent_source_id: None, transmission_depth: 0, reliability: 0.50, domain_competence: [0.50, 0.90] });
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
                // S0: Direct Obs (0.95), S1: Helpful (0.80), S2: Unreliable (0.55). All report X=1!
                sources.push(SourceNode { source_id: 0, source_type: SourceType::DirectObservation, parent_source_id: None, transmission_depth: 0, reliability: 0.95, domain_competence: [0.95, 0.95] });
                sources.push(SourceNode { source_id: 1, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.80, domain_competence: [0.80, 0.80] });
                sources.push(SourceNode { source_id: 2, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.55, domain_competence: [0.55, 0.55] });
                acq_steps = vec![2];

                let chosen_s = rng.gen_range(0..3);
                let rep = 1; // Fixed proposition X=1

                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: rep,
                    immediate_source_id: chosen_s, root_source_id: chosen_s, dependency_group_id: chosen_s,
                    transmission_depth: 0, domain_id: domain,
                });
            }
            "dependency_duplicate" => {
                // Stable Source Architecture:
                // S0 (Ch 0): Primary Independent Sensor (P=0.85)
                // S1 (Ch 1): Relay Repeater that copies S0 90% of the time (Dependent Copier)
                // S2 (Ch 2): Secondary Independent Sensor (P=0.85)
                sources.push(SourceNode { source_id: 0, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.85, domain_competence: [0.85, 0.85] });
                sources.push(SourceNode { source_id: 1, source_type: SourceType::CopiedRelay, parent_source_id: Some(0), transmission_depth: 1, reliability: 0.85, domain_competence: [0.85, 0.85] });
                sources.push(SourceNode { source_id: 2, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.85, domain_competence: [0.85, 0.85] });
                acq_steps = vec![2, 3];

                // 50% Trial A: [S0, S2] Independent Corroboration
                // 50% Trial B: [S0, S1] Copied Redundancy
                is_indep = rng.gen::<f64>() < 0.50;

                let rep_0 = if rng.gen::<f32>() < 0.85 { root_z } else { 1 - root_z };
                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: rep_0,
                    immediate_source_id: 0, root_source_id: 0, dependency_group_id: 0,
                    transmission_depth: 0, domain_id: domain,
                });

                if is_indep {
                    // S2 observes root independently
                    let rep_2 = if rng.gen::<f32>() < 0.85 { root_z } else { 1 - root_z };
                    events.push(ProvenanceEvent {
                        step: 3, root_truth_z: root_z, reported_content: rep_2,
                        immediate_source_id: 2, root_source_id: 2, dependency_group_id: 2,
                        transmission_depth: 0, domain_id: domain,
                    });
                } else {
                    // S1 copies S0's report 90% of the time (10% corruption)
                    let rep_1 = if rng.gen::<f32>() < 0.90 { rep_0 } else { 1 - rep_0 };
                    events.push(ProvenanceEvent {
                        step: 3, root_truth_z: root_z, reported_content: rep_1,
                        immediate_source_id: 1, root_source_id: 0, dependency_group_id: 0,
                        transmission_depth: 1, domain_id: domain,
                    });
                }
            }
            "laundering_depth" => {
                // Chain depth d in 1..=4. Per-hop corruption noise 10%.
                let depth = rng.gen_range(1..=4);
                let mut curr_rep = if rng.gen::<f32>() < 0.90 { root_z } else { 1 - root_z };
                acq_steps = vec![2];

                for _d in 1..=depth {
                    if rng.gen::<f32>() < 0.10 { curr_rep = 1 - curr_rep; }
                }

                sources.push(SourceNode { source_id: depth, source_type: SourceType::CopiedRelay, parent_source_id: Some(0), transmission_depth: depth, reliability: 0.5 + 0.4 * (0.80f32).powi(depth as i32), domain_competence: [0.80, 0.80] });
                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: curr_rep,
                    immediate_source_id: depth, root_source_id: 0, dependency_group_id: 0,
                    transmission_depth: depth, domain_id: domain,
                });
            }
            "source_content_conflict" => {
                // 2x2 Factorial: Source Rel (0.85 vs 0.55) x Content Plausibility (0.80 vs 0.20)
                let is_trusted_source = rng.gen::<f64>() < 0.50;
                let is_plausible_claim = rng.gen::<f64>() < 0.50;
                let s_id = if is_trusted_source { 0 } else { 1 };
                let rel = if is_trusted_source { 0.85 } else { 0.55 };

                sources.push(SourceNode { source_id: 0, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.85, domain_competence: [0.85, 0.85] });
                sources.push(SourceNode { source_id: 1, source_type: SourceType::Helpful, parent_source_id: None, transmission_depth: 0, reliability: 0.55, domain_competence: [0.55, 0.55] });
                acq_steps = vec![2];

                let rep = if is_plausible_claim { 1 } else { 0 };
                let _ = rel;
                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: rep,
                    immediate_source_id: s_id, root_source_id: s_id, dependency_group_id: s_id,
                    transmission_depth: 0, domain_id: domain,
                });
            }
            _ => { // self_other_source
                // S0 (Ch 0): Self-generated internal prediction (0.85)
                // S1 (Ch 1): External peer report (0.85)
                // S2 (Ch 2): Direct observation (0.85)
                let s_id = rng.gen_range(0..3);
                let s_type = match s_id {
                    0 => SourceType::SelfGenerated,
                    1 => SourceType::Helpful,
                    _ => SourceType::DirectObservation,
                };
                sources.push(SourceNode { source_id: s_id, source_type: s_type, parent_source_id: None, transmission_depth: 0, reliability: 0.85, domain_competence: [0.85, 0.85] });
                acq_steps = vec![2];
                let rep = if rng.gen::<f32>() < 0.85 { root_z } else { 1 - root_z };
                events.push(ProvenanceEvent {
                    step: 2, root_truth_z: root_z, reported_content: rep,
                    immediate_source_id: s_id, root_source_id: s_id, dependency_group_id: s_id,
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
            target_action: root_z,
            is_independent_pair: is_indep,
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

            let s_node = &tape.sources[ev.immediate_source_id.min(tape.sources.len() - 1)];
            let rel = match s_node.source_type {
                SourceType::DomainExpert => s_node.domain_competence[tape.domain_id],
                SourceType::Opposite => 0.15,
                SourceType::Random => 0.50,
                SourceType::CopiedRelay => 0.85 * 0.90f32.powi(ev.transmission_depth as i32),
                _ => s_node.reliability,
            };

            let lr = if ev.reported_content == 1 { rel / (1.0 - rel).max(1e-4) } else { (1.0 - rel) / rel.max(1e-4) };
            self.accumulated_log_odds += lr.ln();
        }

        // 2. Decision Window: Candidate proposition presented neutrally WITHOUT contemporaneous source cue
        if t == tape.decision_window_step {
            is_dec = 1;
            content_sym = 2; // Testing belief in Proposition X=1
        }

        let mut reward = 0.0;
        if is_dec == 1 {
            // Reward for choosing optimal action in accordance with root state z
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
