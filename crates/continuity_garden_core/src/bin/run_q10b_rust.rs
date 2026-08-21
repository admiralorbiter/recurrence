//! Full 8-Seed Parallel Q10b Developmental Battery Runner in Native Rust.

use continuity_garden_core::organism::DualLocusOrganism;
use continuity_garden_core::trainer::{
    evaluate_checkpoint_rust, train_duallocus_organism_rust, CheckpointMetrics, CHECKPOINT_EPISODES,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn compute_two_consecutive_onset(passes: &[bool]) -> Option<usize> {
    for i in 0..passes.len().saturating_sub(1) {
        if passes[i] && passes[i + 1] {
            return Some(CHECKPOINT_EPISODES[i]);
        }
    }
    None
}

fn main() {
    let seeds: Vec<u64> = vec![42, 43, 44, 45, 46, 47, 48, 49];

    println!("=======================================================");
    println!("Executing Q10b: Dual-Locus Anticipatory Regulation (Rayon Parallel Rust)");
    println!("Paired Seeds: {:?}", seeds);
    println!("Checkpoints : {:?}", CHECKPOINT_EPISODES);
    println!("=======================================================");

    let start = Instant::now();

    // Execute all 8 paired seeds in parallel across all CPU cores!
    let seed_results: Vec<(u64, Option<usize>, Option<usize>, HashMap<String, CheckpointMetrics>)> = seeds
        .par_iter()
        .map(|&seed| {
            // Lineage A
            let mut model_a = DualLocusOrganism::new(seed);
            let ckpts_a = train_duallocus_organism_rust(&mut model_a, 3200, 50, 0.003, 0.95, false, seed);

            let mut dev_battery_a = HashMap::new();
            let mut rep_passes = Vec::new();
            let mut recruit_passes = Vec::new();

            for &ckpt_t in &CHECKPOINT_EPISODES {
                let model_eval = ckpts_a.get(&ckpt_t).unwrap();
                let metrics = evaluate_checkpoint_rust(model_eval, seed, 100);

                let pass_rep = metrics.delta_r2_vs_current >= 0.20 && metrics.ladder_r2_h_bayesian_q >= 0.30;
                let pass_recruit = metrics.maint_specificity >= 0.25 && metrics.mean_return >= 25.0 && metrics.motor_competence_pass;

                rep_passes.push(pass_rep);
                recruit_passes.push(pass_recruit);

                dev_battery_a.insert(ckpt_t.to_string(), metrics);
            }

            let t_rep = compute_two_consecutive_onset(&rep_passes);
            let t_recruit = compute_two_consecutive_onset(&recruit_passes);

            (seed, t_rep, t_recruit, dev_battery_a)
        })
        .collect();

    let elapsed = start.elapsed();

    println!("\n=======================================================");
    println!("Q10b PARALLEL EXECUTION FINISHED IN {:?}", elapsed);
    println!("=======================================================");

    let mut rep_count = 0;
    let mut recruit_count = 0;
    let mut precedence_count = 0;

    for (s, t_rep, t_recruit, _) in &seed_results {
        println!("  Seed {:<4}: t_rep = {:?} | t_recruit = {:?}", s, t_rep, t_recruit);
        if t_rep.is_some() {
            rep_count += 1;
        }
        if t_recruit.is_some() {
            recruit_count += 1;
        }
        if let (Some(r), Some(c)) = (t_rep, t_recruit) {
            if r < c {
                precedence_count += 1;
            }
        }
    }

    println!("\n=======================================================");
    println!("Q10b Rust Aggregate Results:");
    println!("  Representation Onset (t_rep)   : {}/8 seeds", rep_count);
    println!("  Recruitment Onset (t_recruit)  : {}/8 seeds", recruit_count);
    println!("  T_rep < T_recruit Precedence   : {}/8 seeds", precedence_count);
    println!("  Total Execution Time           : {:?}", elapsed);
    println!("=======================================================\n");

    // Save summary JSON
    let output_dir = Path::new("../../results/e22_garden_q10_endogenous_regulation/run_q10b_rust_parallel");
    std::fs::create_dir_all(output_dir).ok();

    let json_data = serde_json::to_string_pretty(&seed_results).unwrap();
    let mut f = File::create(output_dir.join("q10b_rust_summary.json")).unwrap();
    f.write_all(json_data.as_bytes()).unwrap();

    println!("Saved Rust developmental summary to {:?}", output_dir.join("q10b_rust_summary.json"));
}
