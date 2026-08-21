use std::time::Instant;
use continuity_garden_core::environment::DualLocusRegulatorEnv;
use continuity_garden_core::oracle::{
    evaluate_policy_on_env, AlwaysMaintainPolicy, NeverMaintainPolicy, ObservationBeliefOracle,
    PrivilegedGroundTruthOracle, ReactiveSensorDropPolicy, ShortHistoryWindowPolicy, WarningReflexPolicy,
};

fn main() {
    println!("=======================================================");
    println!("Executing Gate D0a Observability Calibration (Native Rust)");
    println!("=======================================================");

    let start = Instant::now();
    let num_episodes = 200;
    let seed = 42;

    let mut env = DualLocusRegulatorEnv::new(seed, false);

    let never = evaluate_policy_on_env(NeverMaintainPolicy, &mut env, num_episodes, seed);
    let always = evaluate_policy_on_env(AlwaysMaintainPolicy, &mut env, num_episodes, seed);
    let reactive = evaluate_policy_on_env(ReactiveSensorDropPolicy { threshold: 0.60 }, &mut env, num_episodes, seed);
    let warn = evaluate_policy_on_env(WarningReflexPolicy { already_maintained: false }, &mut env, num_episodes, seed);
    let short = evaluate_policy_on_env(ShortHistoryWindowPolicy::new(2), &mut env, num_episodes, seed);
    let belief = evaluate_policy_on_env(
        ObservationBeliefOracle { threshold: 0.60, precursor_noise_std: 0.35, precursor_history: Vec::new() },
        &mut env,
        num_episodes,
        seed,
    );
    let priv_oracle = evaluate_policy_on_env(PrivilegedGroundTruthOracle, &mut env, num_episodes, seed);

    let elapsed = start.elapsed();

    println!("  never_maintain                 : Return = {:+.2} (+/- {:.2}) | Maint = {:.1} | Hits = {:.1}", never.mean_return, never.std_return, never.mean_maintenance_count, never.mean_target_hits);
    println!("  always_maintain                : Return = {:+.2} (+/- {:.2}) | Maint = {:.1} | Hits = {:.1}", always.mean_return, always.std_return, always.mean_maintenance_count, always.mean_target_hits);
    println!("  reactive_sensor_drop           : Return = {:+.2} (+/- {:.2}) | Maint = {:.1} | Hits = {:.1}", reactive.mean_return, reactive.std_return, reactive.mean_maintenance_count, reactive.mean_target_hits);
    println!("  warning_reflex                 : Return = {:+.2} (+/- {:.2}) | Maint = {:.1} | Hits = {:.1}", warn.mean_return, warn.std_return, warn.mean_maintenance_count, warn.mean_target_hits);
    println!("  short_history_window           : Return = {:+.2} (+/- {:.2}) | Maint = {:.1} | Hits = {:.1}", short.mean_return, short.std_return, short.mean_maintenance_count, short.mean_target_hits);
    println!("  observation_belief_oracle      : Return = {:+.2} (+/- {:.2}) | Maint = {:.1} | Hits = {:.1}", belief.mean_return, belief.std_return, belief.mean_maintenance_count, belief.mean_target_hits);
    println!("  privileged_ground_truth_oracle : Return = {:+.2} (+/- {:.2}) | Maint = {:.1} | Hits = {:.1}", priv_oracle.mean_return, priv_oracle.std_return, priv_oracle.mean_maintenance_count, priv_oracle.mean_target_hits);

    let max_heuristic = reactive.mean_return.max(warn.mean_return).max(short.mean_return).max(never.mean_return);
    let advantage = belief.mean_return - max_heuristic;

    println!("\n=======================================================");
    println!("Gate D0a Rust Calibration Results:");
    println!("  E[R_Privileged]         = {:+.2}", priv_oracle.mean_return);
    println!("  E[R_Observation_Belief] = {:+.2}", belief.mean_return);
    println!("  Max Heuristic Baseline  = {:+.2}", max_heuristic);
    println!("  Belief Oracle Advantage = {:+.2} (Target: >= +0.20)", advantage);
    println!("  Elapsed Execution Time  : {:?}", elapsed);
    println!("=======================================================");
}
