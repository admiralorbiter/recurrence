//! SCOUT-E-Q17E-K-R2: Learn the Contraction Algebra Study
//!
//! Investigates whether developmental experience on 1- and 2-step trajectories can
//! discover/select the correct relational composition law from a neutral candidate space,
//! and whether the resulting learned operator recursively closes at zero-shot k=3.
//!
//! Evaluates:
//! - Arm 1: 4-Way Tensor Contraction Topology Selection (R*E, R^T*E, R*E^T, R^T*E^T)
//! - Arm 2: 6-Way General Composition Operator Selection (Matrix Contractions + Hadamard + Additive)
//! - Arm 3: Low-Rank Bilinear Map C_theta(r, e) = U[(Ar) . (Be)]
//!
//! Training: Exclusively on 1- and 2-step experience with broken joins and counterfactuals.
//! Zero 3-hop labels during meta-training. Evaluated across 16 independent seeds.

use std::fs;
use std::path::Path;

use continuity_garden_core::typed_model::sigmoid;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const NUM_NODES: usize = 6;
pub const TENSOR_P: usize = 11;
pub const TENSOR_DIM: usize = TENSOR_P * TENSOR_P; // 121

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topology4WayResult {
    pub seed_index: usize,
    pub seed: u64,
    pub initial_probs: Vec<f32>, // [25.0%, 25.0%, 25.0%, 25.0%]
    pub final_probs: Vec<f32>,
    pub canonical_re_prob: f32,  // O1: R * E
    pub canonical_selected: bool,
    pub k2_accuracy: f32,
    pub k2_pass: bool,
    pub k3_zero_shot_accuracy: f32,
    pub k3_pass: bool,
    pub k3_selectivity_margin: f32,
    pub source_grounding_drop: f32,
    pub destination_grounding_gap: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct General6WayResult {
    pub seed_index: usize,
    pub seed: u64,
    pub initial_probs: Vec<f32>, // [16.7%, 16.7%, 16.7%, 16.7%, 16.7%, 16.7%]
    pub final_probs: Vec<f32>,
    pub canonical_re_prob: f32,
    pub canonical_selected: bool,
    pub k2_accuracy: f32,
    pub k2_pass: bool,
    pub k3_zero_shot_accuracy: f32,
    pub k3_pass: bool,
    pub k3_selectivity_margin: f32,
    pub source_grounding_drop: f32,
    pub destination_grounding_gap: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoutKR2SeedResult {
    pub seed_index: usize,
    pub seed: u64,
    pub topology_4way: Topology4WayResult,
    pub general_6way: General6WayResult,
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
// Arm 1: 4-Way Contraction Topology Selection
// -----------------------------------------------------------------------------
pub struct Topology4WayOrganism {
    pub embeddings: Vec<f32>,
    pub op_logits: Vec<f32>, // 4 logits
}

impl Topology4WayOrganism {
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
        let op_logits = vec![0.0f32; 4]; // Symmetric neutral 25% each
        Self { embeddings, op_logits }
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

    pub fn get_candidate_ops(&self, r: &[f32], e: &[f32]) -> [Vec<f32>; 4] {
        [op_re(r, e), op_rt_e(r, e), op_r_et(r, e), op_rt_et(r, e)]
    }

    pub fn compose(&self, r_prev: &[f32], e_next: &[f32]) -> Vec<f32> {
        if r_prev.iter().all(|&v| v == 0.0) {
            return e_next.to_vec();
        }
        let probs = softmax(&self.op_logits);
        let ops = self.get_candidate_ops(r_prev, e_next);
        let mut out = vec![0.0f32; TENSOR_DIM];
        for (j, op) in ops.iter().enumerate() {
            let p = probs[j];
            for i in 0..TENSOR_DIM {
                out[i] += p * op[i];
            }
        }
        out
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

    pub fn train(&mut self, train_seed: u64, epochs: usize) {
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
                let probs = softmax(&self.op_logits);
                let ops_2 = self.get_candidate_ops(&e1, &e2);

                let mut r2 = vec![0.0f32; TENSOR_DIM];
                for (j, op) in ops_2.iter().enumerate() {
                    for i in 0..TENSOR_DIM {
                        r2[i] += probs[j] * op[i];
                    }
                }

                // Queries
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
                        self.op_logits[j] -= lr_logits * grad_logit;
                    }
                }

                // Broken join training
                let e2_brk = self.encode_edge(x, w);
                let ops_brk = self.get_candidate_ops(&e1, &e2_brk);
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
                    self.op_logits[j] -= lr_logits * grad_logit;
                }
            }
        }
    }
}

fn evaluate_topology_4way(seed_index: usize, seed: u64, train_seed: u64, eval_seed: u64) -> Topology4WayResult {
    let mut organism = Topology4WayOrganism::new(seed);
    let initial_probs = softmax(&organism.op_logits);

    organism.train(train_seed, 120);
    let final_probs = softmax(&organism.op_logits);
    let canonical_re_prob = final_probs[0];
    let canonical_selected = canonical_re_prob >= 0.70;

    let mut rng = ChaCha8Rng::seed_from_u64(eval_seed);
    let nodes = [1, 2, 3, 4, 5, 6];

    let mut k2_correct = 0;
    let mut k3_correct = 0;
    let mut total_k3_tgt = 0.0f32;
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

        let e1 = organism.encode_edge(u, v);
        let e2 = organism.encode_edge(v, w);
        let r2 = organism.compose(&e1, &e2);

        let k2_tgt = organism.query(&r2, u, w);
        let k2_rev = organism.query(&r2, w, u);
        let k2_dist = organism.query(&r2, u, y);
        if k2_tgt > k2_rev && k2_tgt > k2_dist {
            k2_correct += 1;
        }

        // Zero-shot step 3
        let e3 = organism.encode_edge(w, z);
        let r3 = organism.compose(&r2, &e3);

        let k3_tgt = organism.query(&r3, u, z);
        let k3_rev = organism.query(&r3, z, u);
        let k3_dist = organism.query(&r3, u, y);
        let margin = k3_tgt - k3_dist;

        total_k3_tgt += k3_tgt;
        total_k3_dist += k3_dist;
        total_k3_margin += margin;

        if k3_tgt > k3_rev && k3_tgt > k3_dist {
            k3_correct += 1;
        }

        let e3_brk = organism.encode_edge(x, z);
        let r3_brk = organism.compose(&r2, &e3_brk);
        let k3_brk_tgt = organism.query(&r3_brk, u, z);
        total_source_drop += k3_tgt - k3_brk_tgt;

        let e3_alt = organism.encode_edge(w, y);
        let r3_alt = organism.compose(&r2, &e3_alt);
        let k3_alt_tgt = organism.query(&r3_alt, u, y);
        let k3_alt_old = organism.query(&r3_alt, u, z);
        total_dest_gap += k3_alt_tgt - k3_alt_old;
    }

    let n = n_eval as f32;
    Topology4WayResult {
        seed_index,
        seed,
        initial_probs,
        final_probs,
        canonical_re_prob,
        canonical_selected,
        k2_accuracy: k2_correct as f32 / n,
        k2_pass: (k2_correct as f32 / n) >= 0.85,
        k3_zero_shot_accuracy: k3_correct as f32 / n,
        k3_pass: (k3_correct as f32 / n) >= 0.80,
        k3_selectivity_margin: total_k3_margin / n,
        source_grounding_drop: total_source_drop / n,
        destination_grounding_gap: total_dest_gap / n,
    }
}

// -----------------------------------------------------------------------------
// Arm 2: 6-Way General Composition Selection (Matrix + Hadamard + Additive)
// -----------------------------------------------------------------------------
pub struct General6WayOrganism {
    pub embeddings: Vec<f32>,
    pub op_logits: Vec<f32>, // 6 logits
}

impl General6WayOrganism {
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
        let op_logits = vec![0.0f32; 6]; // Symmetric neutral 16.7% each
        Self { embeddings, op_logits }
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

    pub fn get_candidate_ops(&self, r: &[f32], e: &[f32]) -> [Vec<f32>; 6] {
        [
            op_re(r, e),
            op_rt_e(r, e),
            op_r_et(r, e),
            op_rt_et(r, e),
            op_hadamard(r, e),
            op_additive(r, e),
        ]
    }

    pub fn compose(&self, r_prev: &[f32], e_next: &[f32]) -> Vec<f32> {
        if r_prev.iter().all(|&v| v == 0.0) {
            return e_next.to_vec();
        }
        let probs = softmax(&self.op_logits);
        let ops = self.get_candidate_ops(r_prev, e_next);
        let mut out = vec![0.0f32; TENSOR_DIM];
        for (j, op) in ops.iter().enumerate() {
            let p = probs[j];
            for i in 0..TENSOR_DIM {
                out[i] += p * op[i];
            }
        }
        out
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

    pub fn train(&mut self, train_seed: u64, epochs: usize) {
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
                let probs = softmax(&self.op_logits);
                let ops_2 = self.get_candidate_ops(&e1, &e2);

                let mut r2 = vec![0.0f32; TENSOR_DIM];
                for (j, op) in ops_2.iter().enumerate() {
                    for i in 0..TENSOR_DIM {
                        r2[i] += probs[j] * op[i];
                    }
                }

                // Queries
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
                        self.op_logits[j] -= lr_logits * grad_logit;
                    }
                }

                // Broken join training
                let e2_brk = self.encode_edge(x, w);
                let ops_brk = self.get_candidate_ops(&e1, &e2_brk);
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
                    self.op_logits[j] -= lr_logits * grad_logit;
                }
            }
        }
    }
}

fn evaluate_general_6way(seed_index: usize, seed: u64, train_seed: u64, eval_seed: u64) -> General6WayResult {
    let mut organism = General6WayOrganism::new(seed);
    let initial_probs = softmax(&organism.op_logits);

    organism.train(train_seed, 120);
    let final_probs = softmax(&organism.op_logits);
    let canonical_re_prob = final_probs[0];
    let canonical_selected = canonical_re_prob >= 0.50;

    let mut rng = ChaCha8Rng::seed_from_u64(eval_seed);
    let nodes = [1, 2, 3, 4, 5, 6];

    let mut k2_correct = 0;
    let mut k3_correct = 0;
    let mut total_k3_tgt = 0.0f32;
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

        let e1 = organism.encode_edge(u, v);
        let e2 = organism.encode_edge(v, w);
        let r2 = organism.compose(&e1, &e2);

        let k2_tgt = organism.query(&r2, u, w);
        let k2_rev = organism.query(&r2, w, u);
        let k2_dist = organism.query(&r2, u, y);
        if k2_tgt > k2_rev && k2_tgt > k2_dist {
            k2_correct += 1;
        }

        // Zero-shot step 3
        let e3 = organism.encode_edge(w, z);
        let r3 = organism.compose(&r2, &e3);

        let k3_tgt = organism.query(&r3, u, z);
        let k3_rev = organism.query(&r3, z, u);
        let k3_dist = organism.query(&r3, u, y);
        let margin = k3_tgt - k3_dist;

        total_k3_tgt += k3_tgt;
        total_k3_dist += k3_dist;
        total_k3_margin += margin;

        if k3_tgt > k3_rev && k3_tgt > k3_dist {
            k3_correct += 1;
        }

        let e3_brk = organism.encode_edge(x, z);
        let r3_brk = organism.compose(&r2, &e3_brk);
        let k3_brk_tgt = organism.query(&r3_brk, u, z);
        total_source_drop += k3_tgt - k3_brk_tgt;

        let e3_alt = organism.encode_edge(w, y);
        let r3_alt = organism.compose(&r2, &e3_alt);
        let k3_alt_tgt = organism.query(&r3_alt, u, y);
        let k3_alt_old = organism.query(&r3_alt, u, z);
        total_dest_gap += k3_alt_tgt - k3_alt_old;
    }

    let n = n_eval as f32;
    General6WayResult {
        seed_index,
        seed,
        initial_probs,
        final_probs,
        canonical_re_prob,
        canonical_selected,
        k2_accuracy: k2_correct as f32 / n,
        k2_pass: (k2_correct as f32 / n) >= 0.85,
        k3_zero_shot_accuracy: k3_correct as f32 / n,
        k3_pass: (k3_correct as f32 / n) >= 0.80,
        k3_selectivity_margin: total_k3_margin / n,
        source_grounding_drop: total_source_drop / n,
        destination_grounding_gap: total_dest_gap / n,
    }
}

// -----------------------------------------------------------------------------
// Seed Runner
// -----------------------------------------------------------------------------
fn run_scout_k_r2_seed(seed_index: usize) -> ScoutKR2SeedResult {
    let seed = 88000 + (seed_index as u64) * 777;
    let train_seed = seed + 999;
    let eval_seed = seed ^ 0x123456789A;

    let topology_4way = evaluate_topology_4way(seed_index, seed, train_seed, eval_seed);
    let general_6way = evaluate_general_6way(seed_index, seed, train_seed, eval_seed);

    ScoutKR2SeedResult {
        seed_index,
        seed,
        topology_4way,
        general_6way,
    }
}

fn main() {
    println!("================================================================================");
    println!("SCOUT-E-Q17E-K-R2: Learn the Contraction Algebra Study");
    println!("Evaluating Operator Mixture Selection across 16 Seeds");
    println!("================================================================================\n");

    let results: Vec<ScoutKR2SeedResult> = (1..=16)
        .into_par_iter()
        .map(|i| run_scout_k_r2_seed(i))
        .collect();

    let n = results.len() as f32;

    // Topology 4-Way Summary
    let top_k2_p = results.iter().filter(|r| r.topology_4way.k2_pass).count();
    let top_k3_p = results.iter().filter(|r| r.topology_4way.k3_pass).count();
    let top_canon_p = results.iter().filter(|r| r.topology_4way.canonical_selected).count();
    let avg_top_k2 = results.iter().map(|r| r.topology_4way.k2_accuracy).sum::<f32>() / n * 100.0;
    let avg_top_k3 = results.iter().map(|r| r.topology_4way.k3_zero_shot_accuracy).sum::<f32>() / n * 100.0;
    let avg_top_re_prob = results.iter().map(|r| r.topology_4way.canonical_re_prob).sum::<f32>() / n * 100.0;
    let avg_top_margin = results.iter().map(|r| r.topology_4way.k3_selectivity_margin).sum::<f32>() / n;
    let avg_top_src_drop = results.iter().map(|r| r.topology_4way.source_grounding_drop).sum::<f32>() / n;
    let avg_top_dst_gap = results.iter().map(|r| r.topology_4way.destination_grounding_gap).sum::<f32>() / n;

    // General 6-Way Summary
    let gen_k2_p = results.iter().filter(|r| r.general_6way.k2_pass).count();
    let gen_k3_p = results.iter().filter(|r| r.general_6way.k3_pass).count();
    let gen_canon_p = results.iter().filter(|r| r.general_6way.canonical_selected).count();
    let avg_gen_k2 = results.iter().map(|r| r.general_6way.k2_accuracy).sum::<f32>() / n * 100.0;
    let avg_gen_k3 = results.iter().map(|r| r.general_6way.k3_zero_shot_accuracy).sum::<f32>() / n * 100.0;
    let avg_gen_re_prob = results.iter().map(|r| r.general_6way.canonical_re_prob).sum::<f32>() / n * 100.0;
    let avg_gen_margin = results.iter().map(|r| r.general_6way.k3_selectivity_margin).sum::<f32>() / n;
    let avg_gen_src_drop = results.iter().map(|r| r.general_6way.source_grounding_drop).sum::<f32>() / n;
    let avg_gen_dst_gap = results.iter().map(|r| r.general_6way.destination_grounding_gap).sum::<f32>() / n;

    println!("--------------------------------------------------------------------------------");
    println!("ASSAY 1: 4-WAY CONTRACTION TOPOLOGY SELECTION [R*E, R^T*E, R*E^T, R^T*E^T]");
    println!("  Initial Operator Probabilities:                  [25.0%, 25.0%, 25.0%, 25.0%]");
    println!("  Mean Canonical O1 (R * E) Probability:           {:.1}%", avg_top_re_prob);
    println!("  Canonical Operator Dominance Rate (Prob >= 70%): {}/16 ({:.1}%)", top_canon_p, top_canon_p as f32 / n * 100.0);
    println!("  k=2 Developmental Validity Pass Rate:            {}/16 ({:.1}%) [Mean: {:.1}%]", top_k2_p, top_k2_p as f32 / n * 100.0, avg_top_k2);
    println!("  Zero-Shot k=3 Recursive Pass Rate:               {}/16 ({:.1}%) [Mean: {:.1}%]", top_k3_p, top_k3_p as f32 / n * 100.0, avg_top_k3);
    println!("  k=3 Margin: {:>+5.2} | SrcDrop: {:>+5.2} | DstGap: {:>+5.2}", avg_top_margin, avg_top_src_drop, avg_top_dst_gap);
    println!("--------------------------------------------------------------------------------");
    println!("PER-SEED 4-WAY TOPOLOGY CONVERGENCE & k3 ZERO-SHOT RESULTS:");
    for r in &results {
        let p_str: Vec<String> = r.topology_4way.final_probs.iter().map(|p| format!("{:.1}%", p * 100.0)).collect();
        println!(
            "Seed [{:>2}] | O1(R*E): {:>5.1}% | Topo Probs [RE, RtE, REt, RtEt]: [{}] | k2:{:.1}% k3:{:.1}% (Margin:{:>+5.2}) | Pass:{}",
            r.seed_index,
            r.topology_4way.canonical_re_prob * 100.0,
            p_str.join(", "),
            r.topology_4way.k2_accuracy * 100.0,
            r.topology_4way.k3_zero_shot_accuracy * 100.0,
            r.topology_4way.k3_selectivity_margin,
            r.topology_4way.k3_pass,
        );
    }
    println!("--------------------------------------------------------------------------------");
    println!("ASSAY 2: 6-WAY GENERAL COMPOSITION SELECTION [RE, RtE, REt, RtEt, R.E, R+E]");
    println!("  Initial Operator Probabilities:                  [16.7%, 16.7%, 16.7%, 16.7%, 16.7%, 16.7%]");
    println!("  Mean Canonical O1 (R * E) Probability:           {:.1}%", avg_gen_re_prob);
    println!("  Canonical Operator Dominance Rate (Prob >= 50%): {}/16 ({:.1}%)", gen_canon_p, gen_canon_p as f32 / n * 100.0);
    println!("  k=2 Developmental Validity Pass Rate:            {}/16 ({:.1}%) [Mean: {:.1}%]", gen_k2_p, gen_k2_p as f32 / n * 100.0, avg_gen_k2);
    println!("  Zero-Shot k=3 Recursive Pass Rate:               {}/16 ({:.1}%) [Mean: {:.1}%]", gen_k3_p, gen_k3_p as f32 / n * 100.0, avg_gen_k3);
    println!("  k=3 Margin: {:>+5.2} | SrcDrop: {:>+5.2} | DstGap: {:>+5.2}", avg_gen_margin, avg_gen_src_drop, avg_gen_dst_gap);
    println!("================================================================================\n");

    let out_dir = Path::new("crates/continuity_garden_core/data");
    fs::create_dir_all(out_dir).expect("Failed to create data directory");
    let out_path = out_dir.join("q17e_k_r2_learn_algebra_results.json");
    let json_bytes = serde_json::to_string_pretty(&results).expect("Failed to serialize results");
    fs::write(&out_path, json_bytes).expect("Failed to write results file");
    println!("Persisted Scout K-R2 telemetry to: {}", out_path.display());
}
