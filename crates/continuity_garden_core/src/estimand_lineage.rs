//! Estimand Lineage Schema (Provenance v1.3): Strongly-typed provenance metadata contracts.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimandLineage {
    pub scout_id: String,
    pub construct_name: String,
    pub ground_truth_origin: String,
    pub behavioral_target: String,
    pub conditioned_on: Vec<String>,
    pub predictor_state: String,
    pub intervention_type: String,
    pub controls_applied: Vec<String>,
}

impl EstimandLineage {
    pub fn new_for_scout(scout_id: &str) -> Self {
        match scout_id {
            "scout_1_basic_reliability" => Self {
                scout_id: scout_id.to_string(),
                construct_name: "Source Sensitivity Index (SSI)".to_string(),
                ground_truth_origin: "env.tape.sources[s].reliability".to_string(),
                behavioral_target: "P(follow_report | S0) - P(follow_report | S1)".to_string(),
                conditioned_on: vec!["decision_window_content = 2".to_string(), "neutral_cues_absent".to_string()],
                predictor_state: "h_decision".to_string(),
                intervention_type: "source_contrast_patch".to_string(),
                controls_applied: vec!["current_content_matched".to_string(), "blank_delay_steps = 3".to_string()],
            },
            "scout_2_signed_source_types" => Self {
                scout_id: scout_id.to_string(),
                construct_name: "Negative Evidence Weighting / Inversion".to_string(),
                ground_truth_origin: "env.tape.sources[opposite].reliability = 0.15".to_string(),
                behavioral_target: "P(action = 1 - r | Opposite_Source)".to_string(),
                conditioned_on: vec!["source_type = Opposite".to_string()],
                predictor_state: "h_decision".to_string(),
                intervention_type: "signed_source_ablation".to_string(),
                controls_applied: vec!["random_source_50_null_control".to_string()],
            },
            "scout_3_domain_conditional" => Self {
                scout_id: scout_id.to_string(),
                construct_name: "Context-Dependent Epistemic Routing".to_string(),
                ground_truth_origin: "env.tape.sources[s].domain_competence[d]".to_string(),
                behavioral_target: "Trust(S0 | Domain 0) - Trust(S0 | Domain 1)".to_string(),
                conditioned_on: vec!["domain_context in {0, 1}".to_string()],
                predictor_state: "h_decision".to_string(),
                intervention_type: "domain_context_swap".to_string(),
                controls_applied: vec!["source_identity_held_constant".to_string()],
            },
            "scout_4_convergent_content" => Self {
                scout_id: scout_id.to_string(),
                construct_name: "Causal Provenance Effect (CPE)".to_string(),
                ground_truth_origin: "env.provenance_events[acq].immediate_source_id".to_string(),
                behavioral_target: "P(a | X, DirectObs) - P(a | X, S1)".to_string(),
                conditioned_on: vec!["decision_content_symbol = 2 (X=1)".to_string()],
                predictor_state: "h_decision".to_string(),
                intervention_type: "raw_provenance_subspace_lesion".to_string(),
                controls_applied: vec!["exact_identical_current_content_symbol".to_string()],
            },
            "scout_5_dependency_duplicate" => Self {
                scout_id: scout_id.to_string(),
                construct_name: "Dependency Discounting Index (DDI)".to_string(),
                ground_truth_origin: "env.provenance_events.dependency_group_id".to_string(),
                behavioral_target: "Weight(Independent Corroboration) - Weight(Copied Evidence)".to_string(),
                conditioned_on: vec!["total_reports_received = 2".to_string(), "identical_surface_claims".to_string()],
                predictor_state: "h_decision".to_string(),
                intervention_type: "dependency_root_lesion".to_string(),
                controls_applied: vec!["message_count_matched".to_string(), "surface_agreement_matched".to_string()],
            },
            "scout_6_laundering_depth" => Self {
                scout_id: scout_id.to_string(),
                construct_name: "Provenance Half-Life / Ancestry Depth".to_string(),
                ground_truth_origin: "env.provenance_events.transmission_depth".to_string(),
                behavioral_target: "d(Action_Weight) / d(Transmission_Depth)".to_string(),
                conditioned_on: vec!["reported_claim_constant".to_string()],
                predictor_state: "h_decision".to_string(),
                intervention_type: "depth_coordinate_clamp".to_string(),
                controls_applied: vec!["per_hop_copy_noise_fixed = 0.10".to_string()],
            },
            "scout_7_source_content_conflict" => Self {
                scout_id: scout_id.to_string(),
                construct_name: "Source Vigilance vs Content Scrutiny".to_string(),
                ground_truth_origin: "Source_Reliability (0.85/0.55) x Content_Plausibility (0.80/0.20)".to_string(),
                behavioral_target: "2x2 Factorial Action Differential".to_string(),
                conditioned_on: vec!["factorial_2x2_design".to_string()],
                predictor_state: "h_decision".to_string(),
                intervention_type: "orthogonal_vigilance_dissociation".to_string(),
                controls_applied: vec!["marginal_content_and_source_matching".to_string()],
            },
            _ => Self {
                scout_id: "scout_8_self_other_source".to_string(),
                construct_name: "Self-Generated vs Other-Generated Reality Monitoring".to_string(),
                ground_truth_origin: "SourceType::SelfGenerated vs SourceType::Helpful".to_string(),
                behavioral_target: "P(action | Self_Prediction) - P(action | Peer_Report)".to_string(),
                conditioned_on: vec!["reliability_matched = 0.85".to_string()],
                predictor_state: "h_decision".to_string(),
                intervention_type: "agency_provenance_swap".to_string(),
                controls_applied: vec!["objective_accuracy_matched_at_85_percent".to_string()],
            },
        }
    }
}
