//! Reservoir Census: Probing 1,024 Random Recurrent GRU Reservoirs at T=0.
//! Measures the distribution of architectural temporal information availability
//! using standardized Ridge regression on Bayesian Log-Odds L_t = ln(q / (1 - q)).

use continuity_garden_core::environment::DualLocusRegulatorEnv;
use continuity_garden_core::organism::DualLocusOrganism;
use continuity_garden_core::trainer::{fit_and_eval_ridge, solve_linear_system};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirCensusStats {
    pub n_reservoirs: usize,
    pub mean_r2_log_odds: f32,
    pub std_r2_log_odds: f32,
    pub min_r2: f32,
    pub p10_r2: f32,
    pub p25_r2: f32,
    pub median_r2: f32,
    pub p75_r2: f32,
    pub p90_r2: f32,
    pub max_r2: f32,
    pub mean_r2_posterior_q: f32,
    pub mean_r2_current_obs: f32,
    pub mean_delta_r2: f32,
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    let idx = ((sorted.len() as f32 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let n_reservoirs = 1024;
    println!("=======================================================");
    println!("Executing Reservoir Census ({} Random Untrained Reservoirs @ T=0)", n_reservoirs);
    println!("Target: Bayesian Log-Odds L_t = ln(q / (1 - q))");
    println!("=======================================================");

    let start = Instant::now();

    let results: Vec<(f32, f32, f32)> = (0..n_reservoirs)
        .into_par_iter()
        .map(|seed_idx| {
            let seed = 100000 + seed_idx as u64;
            let model = DualLocusOrganism::new(seed);
            let mut env = DualLocusRegulatorEnv::new(seed + 999, false);

            let mut dec_h = Vec::new();
            let mut dec_curr = Vec::new();
            let mut dec_log_odds = Vec::new();
            let mut dec_q = Vec::new();

            for ep in 0..60 {
                let tape = env.generate_deterministic_tape(env.episode_len, seed + 5000 + ep as u64 * 10);
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
                        let mut h_vec = h_next.clone();
                        h_vec.push(1.0); // Fitted intercept
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

            if n_split >= 15 {
                // Standardize features on train split
                let d = dec_h[0].len();
                let mut mean_h = vec![0.0; d];
                let mut std_h = vec![0.0; d];
                for row in &dec_h[..n_split] {
                    for i in 0..d {
                        mean_h[i] += row[i];
                    }
                }
                for i in 0..d {
                    mean_h[i] /= n_split as f32;
                }
                for row in &dec_h[..n_split] {
                    for i in 0..d {
                        std_h[i] += (row[i] - mean_h[i]).powi(2);
                    }
                }
                for i in 0..d {
                    std_h[i] = (std_h[i] / n_split as f32).sqrt().max(1e-6);
                }

                let mut norm_h = dec_h.clone();
                for row in norm_h.iter_mut() {
                    for i in 0..(d - 1) { // Leave bias term untouched
                        row[i] = (row[i] - mean_h[i]) / std_h[i];
                    }
                }

                let r2_log_odds = fit_and_eval_ridge(
                    &norm_h[..n_split],
                    &dec_log_odds[..n_split],
                    &norm_h[n_split..],
                    &dec_log_odds[n_split..],
                    10.0,
                );

                let r2_q = fit_and_eval_ridge(
                    &norm_h[..n_split],
                    &dec_q[..n_split],
                    &norm_h[n_split..],
                    &dec_q[n_split..],
                    10.0,
                );

                let r2_curr = fit_and_eval_ridge(
                    &dec_curr[..n_split],
                    &dec_log_odds[..n_split],
                    &dec_curr[n_split..],
                    &dec_log_odds[n_split..],
                    10.0,
                );

                (r2_log_odds, r2_q, r2_curr)
            } else {
                (0.0, 0.0, 0.0)
            }
        })
        .collect();

    let elapsed = start.elapsed();

    let mut log_odds_r2s: Vec<f32> = results.iter().map(|(lo, _, _)| *lo).collect();
    log_odds_r2s.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut q_r2s: Vec<f32> = results.iter().map(|(_, q, _)| *q).collect();
    q_r2s.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let curr_r2s: Vec<f32> = results.iter().map(|(_, _, c)| *c).collect();

    let mean_lo: f32 = log_odds_r2s.iter().sum::<f32>() / n_reservoirs as f32;
    let var_lo: f32 = log_odds_r2s.iter().map(|&r| (r - mean_lo).powi(2)).sum::<f32>() / n_reservoirs as f32;

    let mean_q: f32 = q_r2s.iter().sum::<f32>() / n_reservoirs as f32;
    let mean_curr: f32 = curr_r2s.iter().sum::<f32>() / n_reservoirs as f32;

    let stats = ReservoirCensusStats {
        n_reservoirs,
        mean_r2_log_odds: mean_lo,
        std_r2_log_odds: var_lo.sqrt(),
        min_r2: log_odds_r2s[0],
        p10_r2: percentile(&log_odds_r2s, 0.10),
        p25_r2: percentile(&log_odds_r2s, 0.25),
        median_r2: percentile(&log_odds_r2s, 0.50),
        p75_r2: percentile(&log_odds_r2s, 0.75),
        p90_r2: percentile(&log_odds_r2s, 0.90),
        max_r2: log_odds_r2s[log_odds_r2s.len() - 1],
        mean_r2_posterior_q: mean_q,
        mean_r2_current_obs: mean_curr,
        mean_delta_r2: mean_lo - mean_curr,
    };

    println!("\n=======================================================");
    println!("RESERVOIR CENSUS RESULTS (1,024 Random Untrained GRUs):");
    println!("  Execution Time           : {:?}", elapsed);
    println!("  Mean R^2 (Log-Odds)      : {:+.3} (+/- {:.3})", stats.mean_r2_log_odds, stats.std_r2_log_odds);
    println!("  Percentiles (Log-Odds)   : p10={:+.3}, p25={:+.3}, Median={:+.3}, p75={:+.3}, p90={:+.3}", stats.p10_r2, stats.p25_r2, stats.median_r2, stats.p75_r2, stats.p90_r2);
    println!("  Min / Max R^2            : {:+.3} / {:+.3}", stats.min_r2, stats.max_r2);
    println!("  Mean R^2 (Posterior q)   : {:+.3}", stats.mean_r2_posterior_q);
    println!("  Current Obs Shortcut R^2 : {:+.3}", stats.mean_r2_current_obs);
    println!("  Delta R^2 (Reservoir-Obs): {:+.3}", stats.mean_delta_r2);
    println!("=======================================================\n");

    let out_dir = Path::new("../../results/e22_garden_q10_endogenous_regulation/census");
    std::fs::create_dir_all(out_dir).ok();
    let json_data = serde_json::to_string_pretty(&stats).unwrap();
    let mut f = File::create(out_dir.join("reservoir_census_1024.json")).unwrap();
    f.write_all(json_data.as_bytes()).unwrap();

    println!("Saved 1,024-reservoir census stats to {:?}", out_dir.join("reservoir_census_1024.json"));
}
