//! Recurrent Organism Architecture & Forward Dynamics in Native Rust.

use crate::environment::ObservationV2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub const EMBED_DIM: usize = 16;
pub const HIDDEN_DIM: usize = 64;
pub const TOTAL_INPUT_DIM: usize = 64; // 16 * 4
pub const COMBINED_DIM: usize = 96; // 64 + 32

#[derive(Debug, Clone)]
pub struct DualLocusOrganism {
    // Embedding tables
    pub symbol_embed: Vec<f32>,      // 6 x 16
    pub action_exec_embed: Vec<f32>, // 5 x 16
    pub action_intend_embed: Vec<f32>, // 5 x 16

    // Sensor linear projection: 4 in -> 16 out
    pub sensor_w: Vec<f32>, // 16 x 4
    pub sensor_b: Vec<f32>, // 16

    // GRU parameters (W_ih: 3*64 x 64, W_hh: 3*64 x 64, b_ih: 3*64, b_hh: 3*64)
    pub gru_w_ih: Vec<f32>, // 192 x 64
    pub gru_w_hh: Vec<f32>, // 192 x 64
    pub gru_b: Vec<f32>,    // 192

    // Policy head: 96 in -> 4 out
    pub policy_w: Vec<f32>, // 4 x 96
    pub policy_b: Vec<f32>, // 4

    // Value head: 96 in -> 1 out
    pub value_w: Vec<f32>, // 1 x 96
    pub value_b: f32,
}

impl DualLocusOrganism {
    pub fn new(seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut rand_vec = |len: usize, std: f32| -> Vec<f32> {
            (0..len).map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * std).collect()
        };

        Self {
            symbol_embed: rand_vec(6 * EMBED_DIM, 0.1),
            action_exec_embed: rand_vec(5 * EMBED_DIM, 0.1),
            action_intend_embed: rand_vec(5 * EMBED_DIM, 0.1),

            sensor_w: rand_vec(EMBED_DIM * 4, (2.0 / 4.0f32).sqrt()),
            sensor_b: vec![0.0; EMBED_DIM],

            gru_w_ih: rand_vec(192 * TOTAL_INPUT_DIM, (2.0 / TOTAL_INPUT_DIM as f32).sqrt()),
            gru_w_hh: rand_vec(192 * HIDDEN_DIM, (2.0 / HIDDEN_DIM as f32).sqrt()),
            gru_b: vec![0.0; 192],

            policy_w: rand_vec(4 * COMBINED_DIM, 0.01),
            policy_b: vec![0.0; 4],

            value_w: rand_vec(1 * COMBINED_DIM, 0.1),
            value_b: 0.0,
        }
    }

    pub fn forward_features(&self, obs: &ObservationV2) -> (Vec<f32>, Vec<f32>) {
        let mut input_feats = Vec::with_capacity(TOTAL_INPUT_DIM);
        let mut instant_feats = Vec::with_capacity(32);

        // 1. Symbol embedding (16)
        let s_idx = obs.symbol.min(5);
        let s_slice = &self.symbol_embed[s_idx * EMBED_DIM..(s_idx + 1) * EMBED_DIM];
        input_feats.extend_from_slice(s_slice);
        instant_feats.extend_from_slice(s_slice);

        // 2. Action exec embedding (16)
        let ae_idx = obs.last_action_executed.min(4);
        input_feats.extend_from_slice(&self.action_exec_embed[ae_idx * EMBED_DIM..(ae_idx + 1) * EMBED_DIM]);

        // 3. Action intend embedding (16)
        let ai_idx = obs.last_action_intended.min(4);
        input_feats.extend_from_slice(&self.action_intend_embed[ai_idx * EMBED_DIM..(ai_idx + 1) * EMBED_DIM]);

        // 4. Sensor projection (16): W * [sens_a, sens_b, warn, is_dec] + b
        let sens_in = [obs.sensor_a, obs.sensor_b, obs.warning_cue, obs.is_decision_window as f32];
        let mut sens_out = vec![0.0; EMBED_DIM];
        for i in 0..EMBED_DIM {
            let mut sum = self.sensor_b[i];
            for j in 0..4 {
                sum += self.sensor_w[i * 4 + j] * sens_in[j];
            }
            sens_out[i] = sum.max(0.0); // ReLU
        }
        input_feats.extend_from_slice(&sens_out);
        instant_feats.extend_from_slice(&sens_out);

        (input_feats, instant_feats)
    }

    pub fn step(
        &self,
        obs: &ObservationV2,
        h: Option<&[f32]>,
    ) -> (Vec<f32>, [f32; 4], f32) {
        let (input_feats, instant_feats) = self.forward_features(obs);
        let h_prev = match h {
            Some(slice) => slice,
            None => &[0.0; HIDDEN_DIM],
        };

        // GRU step: gates z, r, n
        let mut gates = self.gru_b.clone();
        for i in 0..192 {
            for j in 0..TOTAL_INPUT_DIM {
                gates[i] += self.gru_w_ih[i * TOTAL_INPUT_DIM + j] * input_feats[j];
            }
            for j in 0..HIDDEN_DIM {
                gates[i] += self.gru_w_hh[i * HIDDEN_DIM + j] * h_prev[j];
            }
        }

        let mut h_next = vec![0.0; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let z = 1.0 / (1.0 + (-gates[i]).exp()); // reset gate
            let r = 1.0 / (1.0 + (-gates[HIDDEN_DIM + i]).exp()); // update gate
            let n = (gates[2 * HIDDEN_DIM + i] * r).tanh(); // candidate
            h_next[i] = (1.0 - z) * n + z * h_prev[i];
        }

        // Combine hidden + instant features (96 dims)
        let mut combined = Vec::with_capacity(COMBINED_DIM);
        combined.extend_from_slice(&h_next);
        combined.extend_from_slice(&instant_feats);

        // Policy logits (4)
        let mut logits = [0.0; 4];
        for i in 0..4 {
            let mut sum = self.policy_b[i];
            for j in 0..COMBINED_DIM {
                sum += self.policy_w[i * COMBINED_DIM + j] * combined[j];
            }
            logits[i] = sum;
        }

        // Value (1)
        let mut val = self.value_b;
        for j in 0..COMBINED_DIM {
            val += self.value_w[j] * combined[j];
        }

        (h_next, logits, val)
    }

    pub fn sample_action(&self, logits: &[f32; 4], rng: &mut impl Rng) -> (usize, f32) {
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_l: [f32; 4] = [
            (logits[0] - max_l).exp(),
            (logits[1] - max_l).exp(),
            (logits[2] - max_l).exp(),
            (logits[3] - max_l).exp(),
        ];
        let sum_exp: f32 = exp_l.iter().sum();
        let probs = [
            exp_l[0] / sum_exp,
            exp_l[1] / sum_exp,
            exp_l[2] / sum_exp,
            exp_l[3] / sum_exp,
        ];

        let r = rng.gen::<f32>();
        let mut accum = 0.0;
        let mut chosen = 0;
        for (i, &p) in probs.iter().enumerate() {
            accum += p;
            if r <= accum {
                chosen = i;
                break;
            }
        }
        let log_prob = probs[chosen].ln();
        (chosen, log_prob)
    }
}
