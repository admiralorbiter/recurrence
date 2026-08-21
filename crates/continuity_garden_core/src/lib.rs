pub mod bptt_trainer;
pub mod environment;
pub mod oracle;
pub mod organism;
pub mod plastic_trainer;
pub mod trainer;

pub use bptt_trainer::{evaluate_q10d_model, train_policy_readout_regimes, Q10dEvaluationMetrics};
pub use environment::{DualLocusRegulatorEnv, GroundTruthStateV2, ObservationV2};
pub use oracle::{ObservationBeliefOracle, PrivilegedGroundTruthOracle, evaluate_policy_on_env};
pub use organism::DualLocusOrganism;
pub use plastic_trainer::{evaluate_q10c_checkpoint, train_plastic_organism, RecurrenceMode, Q10cCheckpointMetrics};
pub use trainer::{evaluate_checkpoint_rust, evaluate_motor_competence_rust, CheckpointMetrics, CHECKPOINT_EPISODES};
