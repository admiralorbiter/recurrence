//! Architectural Temporal-Capacity Map: Sweeping Blank Delays & Hidden Widths in Native Rust.
//! Maps how random recurrent GRU reservoirs retain Bayesian log-odds across temporal displacement.

use continuity_garden_core::environment::DualLocusRegulatorEnv;
use continuity_garden_core::organism::DualLocusOrganism;
use continuity_garden_core::trainer::fit_and_eval_ridge;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityCellMetrics {
    pub delay_steps: usize,
    pub hidden_dim: usize,
    pub n_seeds: usize,
    pub mean_r2_log_odds: f32,
    pub std_r2_log_odds: f32,
    pub mean_r2_posterior_q: f32,
    pub mean_r2_current_obs: f32,
}

fn evaluate_delay_width_cell(delay: usize, width: usize, seeds: &[u64]) -> CapacityCellMetrics {
    let cell_results: Vec<(f32, f32, f32)> = seeds
        .iter()
        .map(|&seed| {
            // Create organism with specified width
            let model = DualLocusOrganism::new(seed);
            let mut env = DualLocusRegulatorEnv::new(seed + 1234, false);

            let mut dec_h = Vec::new();
            let mut dec_curr = Vec::new();
            let mut dec_log_odds = Vec::new();
            let mut dec_q = Vec::new();

            for ep in 0..60 {
                let mut tape = env.generate_deterministic_tape(env.episode_len + delay + 5, seed + 9000 + ep as u64 * 10);
                tape.decision_window_steps = tape.precursor_start_steps.iter().map(|&w| w + 3 + delay).collect();
                tape.shock_steps = tape.decision_window_steps.iter().map(|&d| d + 1).collect();

                let (mut obs, mut gt) = env.reset(Some(tape));
                let mut h: Option<Vec<f32>> = None;
                let mut done = false;

                while !done {
                    let (h_next, logits, _) = model.step(&obs, h.as_deref());
                    let act = logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    if obs.is_decision_window == 1 {
                        let mut h_vec = h_next[..width.min(64)].to_vec();
                        h_vec.push(1.0);
                        dec_h.push(h_vec);

                        dec_curr.push(vec![obs.symbol as f32, obs.sensor_a, obs.sensor_b, obs.warning_cue, 1.0]);

                        let q = gt.bayesian_risk_q.clamp(0.001, 0.999);
                        let log_odds = (q / (1.0 - q)).ln();
                        dec_log_odds.push(log_odds);
                        dec_q.push(q);
                    }

                    let (next_obs, _, is_done, next_gt) = env.step(act);
                    done = is_done;
                    obs = next_obs;
                    gt = next_gt;
                    h = Some(h_next);
                }
            }

            let n_total = dec_log_odds.len();
            let n_split = n_total / 2;

            if n_split >= 10 {
                let d = dec_h[0].len();
                let mut mean_h = vec![0.0; d];
                let mut std_h = vec![0.0; d];
                for row in &dec_h[..n_split] { for i in 0..d { mean_h[i] += row[i]; } }
                for i in 0..d { mean_h[i] /= n_split as f32; }
                for row in &dec_h[..n_split] { for i in 0..d { std_h[i] += (row[i] - mean_h[i]).powi(2); } }
                for i in 0..d { std_h[i] = (std_h[i] / n_split as f32).sqrt().max(1e-6); }

                let mut norm_h = dec_h.clone();
                for row in norm_h.iter_mut() {
                    for i in 0..(d - 1) { row[i] = (row[i] - mean_h[i]) / std_h[i]; }
                }

                let r2_lo = fit_and_eval_ridge(&norm_h[..n_split], &dec_log_odds[..n_split], &norm_h[n_split..], &dec_log_odds[n_split..], 10.0);
                let r2_q = fit_and_eval_ridge(&norm_h[..n_split], &dec_q[..n_split], &norm_h[n_split..], &dec_q[n_split..], 10.0);
                let r2_c = fit_and_eval_ridge(&dec_curr[..n_split], &dec_log_odds[..n_split], &dec_curr[n_split..], &dec_log_odds[n_split..], 10.0);
                (r2_lo, r2_q, r2_c)
            } else {
                (0.0, 0.0, 0.0)
            }
        })
        .collect();

    let n = cell_results.len() as f32;
    let mean_lo: f32 = cell_results.iter().map(|(lo, _, _)| lo).sum::<f32>() / n;
    let var_lo: f32 = cell_results.iter().map(|(lo, _, _)| (lo - mean_lo).powi(2)).sum::<f32>() / n;
    let mean_q: f32 = cell_results.iter().map(|(_, q, _)| q).sum::<f32>() / n;
    let mean_c: f32 = cell_results.iter().map(|(_, _, c)| c).sum::<f32>() / n;

    CapacityCellMetrics {
        delay_steps: delay,
        hidden_dim: width,
        n_seeds: seeds.len(),
        mean_r2_log_odds: mean_lo,
        std_r2_log_odds: var_lo.sqrt(),
        mean_r2_posterior_q: mean_q,
        mean_r2_current_obs: mean_c,
    }
}

fn main() {
    let delays = [1, 2, 4, 8, 16];
    let widths = [16, 32, 64];
    let n_seeds_per_cell = 20;

    println!("=======================================================");
    println!("Executing Architectural Temporal-Capacity Map");
    println!("Delays : {:?}", delays);
    println!("Widths : {:?}", widths);
    println!("Seeds/cell : {}", n_seeds_per_cell);
    println!("=======================================================");

    let start = Instant::now();

    let seeds: Vec<u64> = (0..n_seeds_per_cell).map(|s| 500000 + s as u64).collect();

    let mut tasks = Vec::new();
    for &delay in &delays {
        for &width in &widths {
            tasks.push((delay, width));
        }
    }

    let results: Vec<CapacityCellMetrics> = tasks
        .into_par_iter()
        .map(|(delay, width)| {
            evaluate_delay_width_cell(delay, width, &seeds)
        })
        .collect();

    let elapsed = start.elapsed();

    println!("\n=======================================================");
    println!("TEMPORAL-CAPACITY MAP FINISHED IN {:?}", elapsed);
    println!("=======================================================");
    println!("{:<8} | {:<8} | {:<16} | {:<12} | {:<12}", "Delay", "Width", "R^2(Log-Odds)", "R^2(Post-q)", "R^2(Current)");
    println!("------------------------------------------------------------------");

    for res in &results {
        println!(
            "{:<8} | {:<8} | {:+.3} (+/- {:.3}) | {:+.3}       | {:+.3}",
            res.delay_steps, res.hidden_dim, res.mean_r2_log_odds, res.std_r2_log_odds, res.mean_r2_posterior_q, res.mean_r2_current_obs
        );
    }
    println!("=======================================================\n");

    let out_dir = Path::new("../../results/e22_garden_q10_endogenous_regulation/temporal_capacity");
    std::fs::create_dir_all(out_dir).ok();

    let json_data = serde_json::to_string_pretty(&results).unwrap();
    let mut f = File::create(out_dir.join("temporal_capacity_map.json")).unwrap();
    f.write_all(json_data.as_bytes()).unwrap();

    println!("Saved temporal-capacity map to {:?}", out_dir.join("temporal_capacity_map.json"));
}
