//! SCOUT-E-Q17E-H-R1: Mechanical Exact Replay on Exact Serialized Scout-G Organisms
//!
//! Loads the 16 winning Condition 5 organisms produced by Scout G, asserts cryptographic
//! SHA-256 parameter digest equality, verifies exact k2/k3 margin reproducibility, and executes
//! the full 6-probe compositional binding battery.

use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;

use continuity_garden_core::typed_model::{
    SerializedScoutGOrganism, TransitionObservation, EDGE_DIM, REL_DIM,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactReplaySeedAssay {
    pub seed_index: usize,
    pub seed: u64,
    pub aux_train_seed: u64,
    pub parameter_sha256: String,
    pub sha_verified: bool,
    pub original_k2_margin: f32,
    pub recomputed_k2_margin: f32,
    pub original_k3_margin: f32,
    pub recomputed_k3_margin: f32,
    pub margin_pre_edge: f32,       // Q(1, 4) on m2 (before step 3)
    pub margin_intact: f32,         // C(m2, e_34)
    pub margin_zero_edge: f32,      // C(m2, 0)
    pub margin_wrong_dst: f32,      // C(m2, e_35)
    pub margin_wrong_src: f32,      // C(m2, e_54)
    pub margin_donor_hist: f32,     // C(m2_donor, e_34)
    pub zero_edge_drop: f32,        // intact - zero
    pub wrong_dst_drop: f32,        // intact - wrong_dst
    pub wrong_src_drop: f32,        // intact - wrong_src
    pub donor_hist_drop: f32,       // intact - donor_hist
    pub unbound_source_anomaly: bool, // wrong_src_drop <= 0
}

fn main() {
    println!("================================================================================");
    println!("SCOUT-E-Q17E-H-R1: Exact Replay on Serialized Winning Scout-G Organisms");
    println!("Loading serialized models from crates/continuity_garden_core/data/q17e_g_serialized_models.json");
    println!("================================================================================");

    let models_path = Path::new("crates/continuity_garden_core/data/q17e_g_serialized_models.json");
    if !models_path.exists() {
        eprintln!("ERROR: Serialized models file not found at {}", models_path.display());
        eprintln!("Please run `cargo run --release --bin scout_q17e_g` first.");
        std::process::exit(1);
    }

    let file = File::open(models_path).expect("Failed to open serialized models file");
    let reader = BufReader::new(file);
    let serialized_organisms: Vec<SerializedScoutGOrganism> =
        serde_json::from_reader(reader).expect("Failed to deserialize Scout-G organisms");

    assert_eq!(serialized_organisms.len(), 16, "Expected exactly 16 serialized organisms");

    let mut assays = Vec::new();

    for organism in &serialized_organisms {
        let model = &organism.model;
        let computed_sha = model.compute_parameter_sha256();
        let sha_verified = computed_sha == organism.parameter_sha256;

        if !sha_verified {
            eprintln!(
                "FATAL: SHA-256 mismatch for seed index {}: expected {}, computed {}",
                organism.seed_index, organism.parameter_sha256, computed_sha
            );
            std::process::exit(1);
        }

        let a = 1;
        let b = 2;
        let c = 3;
        let d = 4;

        // Step 1: 1 -> 2
        let m0 = vec![0.0f32; REL_DIM];
        let (e1, _) = model.encode_edge(&TransitionObservation::new(a, 1, b));
        let (m1, _, _) = model.compose_relation(&m0, &e1);

        // Step 2: 2 -> 3
        let (e2, _) = model.encode_edge(&TransitionObservation::new(b, 2, c));
        let (m2, _, _) = model.compose_relation(&m1, &e2);

        // Recompute k=2
        let recomputed_k2 = model.query_composition(&m2, (a, c)) - model.query_composition(&m2, (c, a));
        assert!(
            (recomputed_k2 - organism.k2_margin).abs() < 1e-5,
            "k2 margin mismatch: expected {}, got {}",
            organism.k2_margin,
            recomputed_k2
        );

        // Probe 1: Pre-edge probe on m2 for (1, 4)
        let margin_pre_edge = model.query_composition(&m2, (a, d)) - model.query_composition(&m2, (d, a));

        // Step 3 Intact: 3 -> 4
        let (e3_intact, _) = model.encode_edge(&TransitionObservation::with_noise(c, 1, d, 0.0));
        let (m3_intact, _, _) = model.compose_relation(&m2, &e3_intact);
        let margin_intact = model.query_composition(&m3_intact, (a, d)) - model.query_composition(&m3_intact, (d, a));

        // Recompute k=3
        assert!(
            (margin_intact - organism.k3_margin).abs() < 1e-5,
            "k3 margin mismatch: expected {}, got {}",
            organism.k3_margin,
            margin_intact
        );

        // Probe 3: Zero final edge
        let (m3_zero, _, _) = model.compose_relation(&m2, &vec![0.0f32; EDGE_DIM]);
        let margin_zero_edge = model.query_composition(&m3_zero, (a, d)) - model.query_composition(&m3_zero, (d, a));

        // Probe 4: Wrong destination (3 -> 5)
        let (e3_wrong_dst, _) = model.encode_edge(&TransitionObservation::with_noise(c, 1, 5, 0.0));
        let (m3_wrong_dst, _, _) = model.compose_relation(&m2, &e3_wrong_dst);
        let margin_wrong_dst = model.query_composition(&m3_wrong_dst, (a, d)) - model.query_composition(&m3_wrong_dst, (d, a));

        // Probe 5: Wrong source (5 -> 4)
        let (e3_wrong_src, _) = model.encode_edge(&TransitionObservation::with_noise(5, 1, d, 0.0));
        let (m3_wrong_src, _, _) = model.compose_relation(&m2, &e3_wrong_src);
        let margin_wrong_src = model.query_composition(&m3_wrong_src, (a, d)) - model.query_composition(&m3_wrong_src, (d, a));

        // Probe 6: Donor history transplant (Donor: 4 -> 3 -> 2 with intact 3 -> 4)
        let (e1_d, _) = model.encode_edge(&TransitionObservation::with_noise(d, 1, c, 0.01));
        let (m1_d, _, _) = model.compose_relation(&m0, &e1_d);
        let (e2_d, _) = model.encode_edge(&TransitionObservation::with_noise(c, 2, b, 0.01));
        let (m2_donor, _, _) = model.compose_relation(&m1_d, &e2_d);
        let (m3_donor_hist, _, _) = model.compose_relation(&m2_donor, &e3_intact);
        let margin_donor_hist = model.query_composition(&m3_donor_hist, (a, d)) - model.query_composition(&m3_donor_hist, (d, a));

        let zero_drop = margin_intact - margin_zero_edge;
        let wrong_dst_drop = margin_intact - margin_wrong_dst;
        let wrong_src_drop = margin_intact - margin_wrong_src;
        let donor_hist_drop = margin_intact - margin_donor_hist;
        let unbound_source_anomaly = wrong_src_drop <= 0.0;

        println!(
            "Seed [{:>2}] SHA:{:.8} | Pre: {:>+6.2} | Intact: {:>+6.2} | Zero: {:>+6.2} | W_Dst(3->5): {:>+6.2} | W_Src(5->4): {:>+6.2} | Donor: {:>+6.2} | Unbound: {}",
            organism.seed_index, organism.parameter_sha256, margin_pre_edge, margin_intact, margin_zero_edge, margin_wrong_dst, margin_wrong_src, margin_donor_hist,
            if unbound_source_anomaly { "YES (Anomaly)" } else { "NO (Src Drop)" }
        );

        assays.push(ExactReplaySeedAssay {
            seed_index: organism.seed_index,
            seed: organism.seed,
            aux_train_seed: organism.aux_train_seed,
            parameter_sha256: organism.parameter_sha256.clone(),
            sha_verified,
            original_k2_margin: organism.k2_margin,
            recomputed_k2_margin: recomputed_k2,
            original_k3_margin: organism.k3_margin,
            recomputed_k3_margin: margin_intact,
            margin_pre_edge,
            margin_intact,
            margin_zero_edge,
            margin_wrong_dst,
            margin_wrong_src,
            margin_donor_hist,
            zero_edge_drop: zero_drop,
            wrong_dst_drop,
            wrong_src_drop,
            donor_hist_drop,
            unbound_source_anomaly,
        });
    }

    let n = assays.len() as f32;
    let avg_pre = assays.iter().map(|a| a.margin_pre_edge).sum::<f32>() / n;
    let avg_intact = assays.iter().map(|a| a.margin_intact).sum::<f32>() / n;
    let avg_zero = assays.iter().map(|a| a.margin_zero_edge).sum::<f32>() / n;
    let avg_wrong_dst = assays.iter().map(|a| a.margin_wrong_dst).sum::<f32>() / n;
    let avg_wrong_src = assays.iter().map(|a| a.margin_wrong_src).sum::<f32>() / n;
    let avg_donor_hist = assays.iter().map(|a| a.margin_donor_hist).sum::<f32>() / n;
    let unbound_count = assays.iter().filter(|a| a.unbound_source_anomaly).count();

    println!("--------------------------------------------------------------------------------");
    println!("MECHANICAL EXACT REPLAY SUMMARY ON 16 VERIFIED SCOUT-G WINNING ORGANISMS:");
    println!("  1. All 16 SHA-256 digests matched recorded hashes: 16/16 (100.0%)");
    println!("  2. All 16 k2/k3 margins recomputed identically:    16/16 (100.0%)");
    println!("  3. Pre-Edge Probe on m2 for (1, 4):                {:>+6.2}", avg_pre);
    println!("  4. Intact 3-Step Sequence (1->2->3->4):            {:>+6.2}", avg_intact);
    println!("  5. Zero Final Edge Ablation (e3 = 0):              {:>+6.2} (Drop: {:>+6.2})", avg_zero, avg_intact - avg_zero);
    println!("  6. Wrong Destination (3 -> 5):                     {:>+6.2} (Drop: {:>+6.2})", avg_wrong_dst, avg_intact - avg_wrong_dst);
    println!("  7. Wrong Source (5 -> 4):                          {:>+6.2} (Drop: {:>+6.2})", avg_wrong_src, avg_intact - avg_wrong_src);
    println!("  8. Donor History Transplant:                       {:>+6.2} (Drop: {:>+6.2})", avg_donor_hist, avg_intact - avg_donor_hist);
    println!("  9. Unbound Source Anomaly Rate (Wrong Src Drop<=0): {}/16 ({:.1}%)", unbound_count, unbound_count as f32 / n * 100.0);
    println!("================================================================================");

    let out_dir = Path::new("crates/continuity_garden_core/data");
    fs::create_dir_all(out_dir).expect("Failed to create data directory");
    let out_path = out_dir.join("q17e_h_r1_exact_replay_results.json");
    let json_bytes = serde_json::to_string_pretty(&assays).expect("Failed to serialize results");
    fs::write(&out_path, json_bytes).expect("Failed to write results file");
    println!("Mechanical Exact Replay telemetry saved to: {}", out_path.display());
}
