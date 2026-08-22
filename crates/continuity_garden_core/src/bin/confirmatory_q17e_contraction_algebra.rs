//! CONFIRMATORY BENCHMARK: CONTRACT-E-Q17E
//! Autonomous Selection of Relational Composition Algebra & Multi-Hop Closure
//!
//! Preregistered Confirmatory Evaluation across 16 completely fresh independent seeds:
//! Master Seed Formula: 99000 + 777 * i (i = 1..=16), disjoint from Scout K/K-R1/K-R2.
//!
//! Confirmatory Conditions:
//! 1. Primary Assay: 6-Way General Composition Mixture (RE, RtE, REt, RtEt, R.E, R+E)
//!    - Untrained Uniform Baseline [16.7% each] vs 2-Step Trained Mixture
//! 2. Topology Replication: 4-Way Contraction Topology Mixture (RE, RtE, REt, RtEt)
//!    - Untrained Uniform Baseline [25.0% each] vs 2-Step Trained Mixture
//! 3. Negative Control: Wrong-Index Contraction (Rt * E)
//!
//! Training: Exclusively on 1- and 2-step experience with counterfactuals and broken joins.
//! Zero 3-hop training labels. 120 epochs x 100 batches, lr=0.08. 200 eval trajectories/seed.

use std::fs;
use std::path::Path;

use continuity_garden_core::typed_model::sigmoid;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const NUM_NODES: usize = 6;
pub const TENSOR_P: usize = 11;
pub const TENSOR_DIM: usize = TENSOR_P * TENSOR_P; // 121

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfirmatoryBatteryResult {
    pub name: String,
    pub k2_accuracy: f32,
    pub k2_pass: bool,
    pub k3_accuracy: f32,
    pub k3_pass: bool,
    pub k3_target_score: f32,
    pub k3_reverse_score: f32,
    pub k3_distractor_score: f32,
    pub k3_selectivity_margin: f32,
    pub source_grounding_drop: f32,
    pub destination_grounding_gap: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfirmatorySeedResult {
    pub seed_index: usize,
    pub seed: u64,
    // 6-Way Primary Assay
    pub primary_6way_initial_probs: Vec<f32>,
    pub primary_6way_final_probs: Vec<f32>,
    pub primary_6way_canonical_prob: f32,
    pub primary_6way_dominant: bool,
    pub primary_6way_untrained: ConfirmatoryBatteryResult,
    pub primary_6way_trained: ConfirmatoryBatteryResult,
    pub primary_6way_margin_delta: f32,
    // 4-Way Topology Replication
    pub topo_4way_initial_probs: Vec<f32>,
    pub topo_4way_final_probs: Vec<f32>,
    pub topo_4way_canonical_prob: f32,
    pub topo_4way_dominant: bool,
    pub topo_4way_untrained: ConfirmatoryBatteryResult,
    pub topo_4way_trained: ConfirmatoryBatteryResult,
    pub topo_4way_margin_delta: f32,
    // Negative Control
    pub negative_control_wrong_index: ConfirmatoryBatteryResult,
}

// -----------------------------------------------------------------------------
// Matrix Operation Primitives
// -----------------------------------------------------------------------------
fn op_re(r: &[f32], e: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; TENSOR_DIM];
    for i in 0..TENSOR_P {
        for j in 0..TENSOR_P {
            let mut sum = 0.0f32;
            for k in 0..TENSOR_P {
                sum += r[i * TENSOR_P + k] * e[k * TENSOR_P + j];
            }
            out[i * TENSOR_P + j] = sum;
        }
    }
    out
}

fn op_rt_e(r: &[f32], e: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; TENSOR_DIM];
    for i in 0..TENSOR_P {
        for j in 0..TENSOR_P {
            let mut sum = 0.0f32;
            for k in 0..TENSOR_P {
                sum += r[k * TENSOR_P + i] * e[k * TENSOR_P + j];
            }
            out[i * TENSOR_P + j] = sum;
        }
    }
    out
}

fn op_r_et(r: &[f32], e: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; TENSOR_DIM];
    for i in 0..TENSOR_P {
        for j in 0..TENSOR_P {
            let mut sum = 0.0f32;
            for k in 0..TENSOR_P {
                sum += r[i * TENSOR_P + k] * e[j * TENSOR_P + k];
            }
            out[i * TENSOR_P + j] = sum;
        }
    }
    out
}

fn op_rt_et(r: &[f32], e: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; TENSOR_DIM];
    for i in 0..TENSOR_P {
        for j in 0..TENSOR_P {
            let mut sum = 0.0f32;
            for k in 0..TENSOR_P {
                sum += r[k * TENSOR_P + i] * e[j * TENSOR_P + k];
            }
            out[i * TENSOR_P + j] = sum;
        }
    }
    out
}

fn op_hadamard(r: &[f32], e: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; TENSOR_DIM];
    for i in 0..TENSOR_DIM {
        out[i] = r[i] * e[i];
    }
    out
}

fn op_additive(r: &[f32], e: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; TENSOR_DIM];
    for i in 0..TENSOR_DIM {
        out[i] = (r[i] + e[i]) * 0.5;
    }
    out
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    logits.iter().map(|&l| (l - max_l).exp() / exp_sum).collect()
}

// -----------------------------------------------------------------------------
// Organism Implementation
// -----------------------------------------------------------------------------
pub struct ConfirmatoryOrganism {
    pub embeddings: Vec<f32>,
    pub op_logits_6way: Vec<f32>,
    pub op_logits_4way: Vec<f32>,
}

impl ConfirmatoryOrganism {
    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let norm_dist = Normal::new(0.0f32, 1.0f32).unwrap();
        let mut embeddings = vec![0.0f32; NUM_NODES * TENSOR_P];
        for i in 0..NUM_NODES {
            let mut norm_sq = 0.0f32;
            for j in 0..TENSOR_P {
                let val = norm_dist.sample(&mut rng);
                embeddings[i * TENSOR_P + j] = val;
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt().max(1e-6);
            for j in 0..TENSOR_P {
                embeddings[i * TENSOR_P + j] /= norm;
            }
        }
        let op_logits_6way = vec![0.0f32; 6];
        let op_logits_4way = vec![0.0f32; 4];
        Self {
            embeddings,
            op_logits_6way,
            op_logits_4way,
        }
    }

    pub fn get_node_emb(&self, node_1idx: usize) -> Vec<f32> {
        let idx = (node_1idx - 1) % NUM_NODES;
        self.embeddings[idx * TENSOR_P..(idx + 1) * TENSOR_P].to_vec()
    }

    pub fn encode_edge(&self, s: usize, d: usize) -> Vec<f32> {
        let h_s = self.get_node_emb(s);
        let h_d = self.get_node_emb(d);
        let mut e = vec![0.0f32; TENSOR_DIM];
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                e[i * TENSOR_P + j] = h_s[i] * h_d[j];
            }
        }
        e
    }

    pub fn get_ops_6way(&self, r: &[f32], e: &[f32]) -> [Vec<f32>; 6] {
        [
            op_re(r, e),
            op_rt_e(r, e),
            op_r_et(r, e),
            op_rt_et(r, e),
            op_hadamard(r, e),
            op_additive(r, e),
        ]
    }

    pub fn get_ops_4way(&self, r: &[f32], e: &[f32]) -> [Vec<f32>; 4] {
        [op_re(r, e), op_rt_e(r, e), op_r_et(r, e), op_rt_et(r, e)]
    }

    pub fn compose_6way(&self, r_prev: &[f32], e_next: &[f32]) -> Vec<f32> {
        if r_prev.iter().all(|&v| v == 0.0) {
            return e_next.to_vec();
        }
        let probs = softmax(&self.op_logits_6way);
        let ops = self.get_ops_6way(r_prev, e_next);
        let mut out = vec![0.0f32; TENSOR_DIM];
        for (j, op) in ops.iter().enumerate() {
            let p = probs[j];
            for i in 0..TENSOR_DIM {
                out[i] += p * op[i];
            }
        }
        out
    }

    pub fn compose_4way(&self, r_prev: &[f32], e_next: &[f32]) -> Vec<f32> {
        if r_prev.iter().all(|&v| v == 0.0) {
            return e_next.to_vec();
        }
        let probs = softmax(&self.op_logits_4way);
        let ops = self.get_ops_4way(r_prev, e_next);
        let mut out = vec![0.0f32; TENSOR_DIM];
        for (j, op) in ops.iter().enumerate() {
            let p = probs[j];
            for i in 0..TENSOR_DIM {
                out[i] += p * op[i];
            }
        }
        out
    }

    pub fn compose_wrong_index(&self, r_prev: &[f32], e_next: &[f32]) -> Vec<f32> {
        if r_prev.iter().all(|&v| v == 0.0) {
            return e_next.to_vec();
        }
        op_rt_e(r_prev, e_next)
    }

    pub fn query(&self, r: &[f32], s: usize, d: usize) -> f32 {
        let h_s = self.get_node_emb(s);
        let h_d = self.get_node_emb(d);
        let mut score = 0.0f32;
        for i in 0..TENSOR_P {
            for j in 0..TENSOR_P {
                score += h_s[i] * r[i * TENSOR_P + j] * h_d[j];
            }
        }
        score
    }

    pub fn run_battery<F>(&self, eval_seed: u64, name: &str, compose_fn: F) -> ConfirmatoryBatteryResult
    where
        F: Fn(&[f32], &[f32]) -> Vec<f32>,
    {
        let mut rng = ChaCha8Rng::seed_from_u64(eval_seed);
        let nodes = [1, 2, 3, 4, 5, 6];

        let mut k2_correct = 0;
        let mut k3_correct = 0;
        let mut total_k3_tgt = 0.0f32;
        let mut total_k3_rev = 0.0f32;
        let mut total_k3_dist = 0.0f32;
        let mut total_k3_margin = 0.0f32;
        let mut total_source_drop = 0.0f32;
        let mut total_dest_gap = 0.0f32;

        let n_eval = 200;
        for _ in 0..n_eval {
            let mut perm = nodes;
            perm.shuffle(&mut rng);
            let u = perm[0];
            let v = perm[1];
            let w = perm[2];
            let z = perm[3];
            let x = perm[4];
            let y = perm[5];

            let e1 = self.encode_edge(u, v);
            let e2 = self.encode_edge(v, w);
            let r2 = compose_fn(&e1, &e2);

            let k2_tgt = self.query(&r2, u, w);
            let k2_rev = self.query(&r2, w, u);
            let k2_dist = self.query(&r2, u, y);
            if k2_tgt > k2_rev && k2_tgt > k2_dist {
                k2_correct += 1;
            }

            // Zero-shot step 3
            let e3 = self.encode_edge(w, z);
            let r3 = compose_fn(&r2, &e3);

            let k3_tgt = self.query(&r3, u, z);
            let k3_rev = self.query(&r3, z, u);
            let k3_dist = self.query(&r3, u, y);
            let margin = k3_tgt - k3_dist;

            total_k3_tgt += k3_tgt;
            total_k3_rev += k3_rev;
            total_k3_dist += k3_dist;
            total_k3_margin += margin;

            if k3_tgt > k3_rev && k3_tgt > k3_dist {
                k3_correct += 1;
            }

            let e3_brk = self.encode_edge(x, z);
            let r3_brk = compose_fn(&r2, &e3_brk);
            let k3_brk_tgt = self.query(&r3_brk, u, z);
            total_source_drop += k3_tgt - k3_brk_tgt;

            let e3_alt = self.encode_edge(w, y);
            let r3_alt = compose_fn(&r2, &e3_alt);
            let k3_alt_tgt = self.query(&r3_alt, u, y);
            let k3_alt_old = self.query(&r3_alt, u, z);
            total_dest_gap += k3_alt_tgt - k3_alt_old;
        }

        let n = n_eval as f32;
        ConfirmatoryBatteryResult {
            name: name.to_string(),
            k2_accuracy: k2_correct as f32 / n,
            k2_pass: (k2_correct as f32 / n) >= 0.85,
            k3_accuracy: k3_correct as f32 / n,
            k3_pass: (k3_correct as f32 / n) >= 0.80,
            k3_target_score: total_k3_tgt / n,
            k3_reverse_score: total_k3_rev / n,
            k3_distractor_score: total_k3_dist / n,
            k3_selectivity_margin: total_k3_margin / n,
            source_grounding_drop: total_source_drop / n,
            destination_grounding_gap: total_dest_gap / n,
        }
    }

    pub fn train_6way(&mut self, train_seed: u64, epochs: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(train_seed);
        let nodes = [1, 2, 3, 4, 5, 6];
        let lr_logits = 0.08f32;

        for _epoch in 0..epochs {
            for _batch in 0..100 {
                let mut perm = nodes;
                perm.shuffle(&mut rng);
                let u = perm[0];
                let v = perm[1];
                let w = perm[2];
                let x = perm[3];
                let y = perm[4];

                let e1 = self.encode_edge(u, v);
                let e2 = self.encode_edge(v, w);
                let probs = softmax(&self.op_logits_6way);
                let ops_2 = self.get_ops_6way(&e1, &e2);

                let mut r2 = vec![0.0f32; TENSOR_DIM];
                for (j, op) in ops_2.iter().enumerate() {
                    for i in 0..TENSOR_DIM {
                        r2[i] += probs[j] * op[i];
                    }
                }

                let queries = [
                    (u, w, 1.0f32, 1.0f32),
                    (w, u, 0.0f32, 0.5f32),
                    (u, y, 0.0f32, 0.5f32),
                ];

                for &(qs, qd, target, weight) in &queries {
                    let score = self.query(&r2, qs, qd);
                    let err = (sigmoid(score) - target) * weight;

                    let h_qs = self.get_node_emb(qs);
                    let h_qd = self.get_node_emb(qd);

                    let mut grad_op = vec![0.0f32; 6];
                    for (j, op) in ops_2.iter().enumerate() {
                        let mut op_score = 0.0f32;
                        for i in 0..TENSOR_P {
                            for k in 0..TENSOR_P {
                                op_score += h_qs[i] * op[i * TENSOR_P + k] * h_qd[k];
                            }
                        }
                        grad_op[j] = err * op_score;
                    }

                    let dot_p_g: f32 = probs.iter().zip(grad_op.iter()).map(|(&p, &g)| p * g).sum();
                    for j in 0..6 {
                        let grad_logit = probs[j] * (grad_op[j] - dot_p_g);
                        self.op_logits_6way[j] -= lr_logits * grad_logit;
                    }
                }

                let e2_brk = self.encode_edge(x, w);
                let ops_brk = self.get_ops_6way(&e1, &e2_brk);
                let mut r2_brk = vec![0.0f32; TENSOR_DIM];
                for (j, op) in ops_brk.iter().enumerate() {
                    for i in 0..TENSOR_DIM {
                        r2_brk[i] += probs[j] * op[i];
                    }
                }
                let score_brk = self.query(&r2_brk, u, w);
                let err_brk = sigmoid(score_brk) - 0.0f32;

                let h_u = self.get_node_emb(u);
                let h_w = self.get_node_emb(w);
                let mut grad_op_brk = vec![0.0f32; 6];
                for (j, op) in ops_brk.iter().enumerate() {
                    let mut op_score = 0.0f32;
                    for i in 0..TENSOR_P {
                        for k in 0..TENSOR_P {
                            op_score += h_u[i] * op[i * TENSOR_P + k] * h_w[k];
                        }
                    }
                    grad_op_brk[j] = err_brk * op_score;
                }
                let dot_p_g_brk: f32 = probs.iter().zip(grad_op_brk.iter()).map(|(&p, &g)| p * g).sum();
                for j in 0..6 {
                    let grad_logit = probs[j] * (grad_op_brk[j] - dot_p_g_brk);
                    self.op_logits_6way[j] -= lr_logits * grad_logit;
                }
            }
        }
    }

    pub fn train_4way(&mut self, train_seed: u64, epochs: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(train_seed);
        let nodes = [1, 2, 3, 4, 5, 6];
        let lr_logits = 0.08f32;

        for _epoch in 0..epochs {
            for _batch in 0..100 {
                let mut perm = nodes;
                perm.shuffle(&mut rng);
                let u = perm[0];
                let v = perm[1];
                let w = perm[2];
                let x = perm[3];
                let y = perm[4];

                let e1 = self.encode_edge(u, v);
                let e2 = self.encode_edge(v, w);
                let probs = softmax(&self.op_logits_4way);
                let ops_2 = self.get_ops_4way(&e1, &e2);

                let mut r2 = vec![0.0f32; TENSOR_DIM];
                for (j, op) in ops_2.iter().enumerate() {
                    for i in 0..TENSOR_DIM {
                        r2[i] += probs[j] * op[i];
                    }
                }

                let queries = [
                    (u, w, 1.0f32, 1.0f32),
                    (w, u, 0.0f32, 0.5f32),
                    (u, y, 0.0f32, 0.5f32),
                ];

                for &(qs, qd, target, weight) in &queries {
                    let score = self.query(&r2, qs, qd);
                    let err = (sigmoid(score) - target) * weight;

                    let h_qs = self.get_node_emb(qs);
                    let h_qd = self.get_node_emb(qd);

                    let mut grad_op = vec![0.0f32; 4];
                    for (j, op) in ops_2.iter().enumerate() {
                        let mut op_score = 0.0f32;
                        for i in 0..TENSOR_P {
                            for k in 0..TENSOR_P {
                                op_score += h_qs[i] * op[i * TENSOR_P + k] * h_qd[k];
                            }
                        }
                        grad_op[j] = err * op_score;
                    }

                    let dot_p_g: f32 = probs.iter().zip(grad_op.iter()).map(|(&p, &g)| p * g).sum();
                    for j in 0..4 {
                        let grad_logit = probs[j] * (grad_op[j] - dot_p_g);
                        self.op_logits_4way[j] -= lr_logits * grad_logit;
                    }
                }

                let e2_brk = self.encode_edge(x, w);
                let ops_brk = self.get_ops_4way(&e1, &e2_brk);
                let mut r2_brk = vec![0.0f32; TENSOR_DIM];
                for (j, op) in ops_brk.iter().enumerate() {
                    for i in 0..TENSOR_DIM {
                        r2_brk[i] += probs[j] * op[i];
                    }
                }
                let score_brk = self.query(&r2_brk, u, w);
                let err_brk = sigmoid(score_brk) - 0.0f32;

                let h_u = self.get_node_emb(u);
                let h_w = self.get_node_emb(w);
                let mut grad_op_brk = vec![0.0f32; 4];
                for (j, op) in ops_brk.iter().enumerate() {
                    let mut op_score = 0.0f32;
                    for i in 0..TENSOR_P {
                        for k in 0..TENSOR_P {
                            op_score += h_u[i] * op[i * TENSOR_P + k] * h_w[k];
                        }
                    }
                    grad_op_brk[j] = err_brk * op_score;
                }
                let dot_p_g_brk: f32 = probs.iter().zip(grad_op_brk.iter()).map(|(&p, &g)| p * g).sum();
                for j in 0..4 {
                    let grad_logit = probs[j] * (grad_op_brk[j] - dot_p_g_brk);
                    self.op_logits_4way[j] -= lr_logits * grad_logit;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Seed Execution
// -----------------------------------------------------------------------------
fn run_confirmatory_q17e_seed(seed_index: usize) -> ConfirmatorySeedResult {
    // Master Seed Formula: 99000 + 777 * i (Disjoint from all previous scouts)
    let seed = 99000 + (seed_index as u64) * 777;
    let train_seed = seed + 999;
    let eval_seed = seed ^ 0xDEADBEEFCAFE;

    let mut organism = ConfirmatoryOrganism::new(seed);

    // Negative Control
    let negative_control_wrong_index = organism.run_battery(
        eval_seed,
        "wrong_index_contraction",
        |r, e| organism.compose_wrong_index(r, e),
    );

    // 1. Primary 6-Way Assay
    let primary_6way_initial_probs = softmax(&organism.op_logits_6way);
    let primary_6way_untrained = organism.run_battery(
        eval_seed,
        "primary_6way_untrained_uniform",
        |r, e| organism.compose_6way(r, e),
    );

    organism.train_6way(train_seed, 120);
    let primary_6way_final_probs = softmax(&organism.op_logits_6way);
    let primary_6way_canonical_prob = primary_6way_final_probs[0];
    let primary_6way_dominant = primary_6way_canonical_prob >= 0.50;
    let primary_6way_trained = organism.run_battery(
        eval_seed,
        "primary_6way_trained",
        |r, e| organism.compose_6way(r, e),
    );
    let primary_6way_margin_delta = primary_6way_trained.k3_selectivity_margin - primary_6way_untrained.k3_selectivity_margin;

    // 2. Topology 4-Way Replication
    let topo_4way_initial_probs = softmax(&organism.op_logits_4way);
    let topo_4way_untrained = organism.run_battery(
        eval_seed,
        "topo_4way_untrained_uniform",
        |r, e| organism.compose_4way(r, e),
    );

    organism.train_4way(train_seed, 120);
    let topo_4way_final_probs = softmax(&organism.op_logits_4way);
    let topo_4way_canonical_prob = topo_4way_final_probs[0];
    let topo_4way_dominant = topo_4way_canonical_prob >= 0.70;
    let topo_4way_trained = organism.run_battery(
        eval_seed,
        "topo_4way_trained",
        |r, e| organism.compose_4way(r, e),
    );
    let topo_4way_margin_delta = topo_4way_trained.k3_selectivity_margin - topo_4way_untrained.k3_selectivity_margin;

    ConfirmatorySeedResult {
        seed_index,
        seed,
        primary_6way_initial_probs,
        primary_6way_final_probs,
        primary_6way_canonical_prob,
        primary_6way_dominant,
        primary_6way_untrained,
        primary_6way_trained,
        primary_6way_margin_delta,
        topo_4way_initial_probs,
        topo_4way_final_probs,
        topo_4way_canonical_prob,
        topo_4way_dominant,
        topo_4way_untrained,
        topo_4way_trained,
        topo_4way_margin_delta,
        negative_control_wrong_index,
    }
}

fn main() {
    println!("================================================================================");
    println!("CONFIRMATORY BENCHMARK: CONTRACT-E-Q17E");
    println!("Autonomous Contraction Algebra Selection & Zero-Shot Multi-Hop Closure");
    println!("Preregistered Master Seeds (99000 + 777*i, N=16) with Paired Controls");
    println!("================================================================================\n");

    let results: Vec<ConfirmatorySeedResult> = (1..=16)
        .into_par_iter()
        .map(|i| run_confirmatory_q17e_seed(i))
        .collect();

    let n = results.len() as f32;

    // Gate 1: k=2 validity
    let g1_pass_count = results.iter().filter(|r| r.primary_6way_trained.k2_pass).count();
    let g1_pass = g1_pass_count >= 15;

    // Gate 2: Canonical operator dominance
    let g2_pass_count = results.iter().filter(|r| r.primary_6way_dominant).count();
    let g2_pass = g2_pass_count >= 15;

    // Gate 3: Causal Source drop >= +0.50
    let g3_pass_count = results.iter().filter(|r| r.primary_6way_trained.source_grounding_drop >= 0.50).count();
    let g3_pass = g3_pass_count >= 14;

    // Gate 4: Causal Destination gap >= +0.50
    let g4_pass_count = results.iter().filter(|r| r.primary_6way_trained.destination_grounding_gap >= 0.50).count();
    let g4_pass = g4_pass_count >= 14;

    // Gate 5: Paired Selectivity Gain (Trained > Untrained)
    let g5_pass_count = results.iter().filter(|r| r.primary_6way_margin_delta > 0.0).count();
    let g5_pass = g5_pass_count >= 14;

    // Gate 6: Aggregate Absolute Margin Gain >= +0.40
    let avg_un_margin = results.iter().map(|r| r.primary_6way_untrained.k3_selectivity_margin).sum::<f32>() / n;
    let avg_tr_margin = results.iter().map(|r| r.primary_6way_trained.k3_selectivity_margin).sum::<f32>() / n;
    let agg_margin_gain = avg_tr_margin - avg_un_margin;
    let g6_pass = agg_margin_gain >= 0.40;

    // Gate 7: Descriptive Amplification
    let rel_amp = if avg_un_margin > 0.0 { avg_tr_margin / avg_un_margin } else { 0.0 };
    let g7_pass = rel_amp >= 5.0;

    // Gate 8: Negative Control (Wrong Contraction retains k=2 competence >=12/16 while failing k=3 closure <= 2/16)
    let neg_k2_pass_count = results.iter().filter(|r| r.negative_control_wrong_index.k2_pass).count();
    let neg_k3_pass_count = results.iter().filter(|r| r.negative_control_wrong_index.k3_pass).count();
    let g8_pass = (neg_k2_pass_count >= 12) && (neg_k3_pass_count <= 2);

    let all_gates_passed = g1_pass && g2_pass && g3_pass && g4_pass && g5_pass && g6_pass && g8_pass;

    println!("--------------------------------------------------------------------------------");
    println!("PRIMARY 6-WAY ASSAY (UNTRAINED UNIFORM vs 2-STEP TRAINED):");
    println!("  Untrained Uniform Baseline Margin:               {:<+5.2}", avg_un_margin);
    println!("  2-Step Trained Mixture Margin:                   {:<+5.2} (Absolute Gain: {:<+5.2}, Rel: {:.1}x)", avg_tr_margin, agg_margin_gain, rel_amp);
    println!("  Mean Canonical O1 (R * E) Probability:           {:.1}%", results.iter().map(|r| r.primary_6way_canonical_prob).sum::<f32>() / n * 100.0);
    println!("--------------------------------------------------------------------------------");
    println!("CONFIRMATORY ACCEPTANCE GATES (CONTRACT-E-Q17E):");
    println!("  Gate 1 (k=2 Developmental Validity >= 15/16):   {}/16 [{}]", g1_pass_count, if g1_pass { "PASS" } else { "FAIL" });
    println!("  Gate 2 (Canonical Operator Dominance >= 15/16):  {}/16 [{}]", g2_pass_count, if g2_pass { "PASS" } else { "FAIL" });
    println!("  Gate 3 (Causal Source Drop >= +0.50 in >=14/16): {}/16 [{}]", g3_pass_count, if g3_pass { "PASS" } else { "FAIL" });
    println!("  Gate 4 (Causal Dest Gap >= +0.50 in >=14/16):    {}/16 [{}]", g4_pass_count, if g4_pass { "PASS" } else { "FAIL" });
    println!("  Gate 5 (Paired Selectivity Margin Gain >=14/16): {}/16 [{}]", g5_pass_count, if g5_pass { "PASS" } else { "FAIL" });
    println!("  Gate 6 (Aggregate Absolute Margin Gain >= +0.40): {:<+5.2} [{}]", agg_margin_gain, if g6_pass { "PASS" } else { "FAIL" });
    println!("  Gate 7 (Descriptive Amplification >= 5.0x):      {:.1}x [{}]", rel_amp, if g7_pass { "PASS" } else { "FAIL" });
    println!("  Gate 8 (Negative Control Specificity):           k2:{}/16 k3:{}/16 [{}]", neg_k2_pass_count, neg_k3_pass_count, if g8_pass { "PASS" } else { "FAIL" });
    println!("--------------------------------------------------------------------------------");
    println!("ALL CONFIRMATORY GATES: {}", if all_gates_passed { "PASS (VERIFIED)" } else { "FAIL" });
    println!("================================================================================\n");

    let out_dir = Path::new("crates/continuity_garden_core/data");
    fs::create_dir_all(out_dir).expect("Failed to create data directory");
    let out_path = out_dir.join("confirmatory_q17e_results.json");
    let json_bytes = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    fs::write(&out_path, json_bytes).expect("Failed to write results file");
    println!("Persisted Confirmatory Q17E telemetry to: {}", out_path.display());
}
