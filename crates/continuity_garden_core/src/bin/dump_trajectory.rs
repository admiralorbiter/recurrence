//! Dumps exact trajectory for bit-level Python-Rust Parity Fixture.

use continuity_garden_core::environment::{DualLocusRegulatorEnv, EventTapeV2};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::Read;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepDump {
    pub step: usize,
    pub symbol: usize,
    pub sensor_a: f32,
    pub sensor_b: f32,
    pub warning_cue: f32,
    pub is_decision_window: usize,
    pub reward: f32,
    pub i_t: f32,
    pub x_t: f32,
    pub done: bool,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut env = DualLocusRegulatorEnv::new(42, false);

    let tape: EventTapeV2 = if args.len() > 1 {
        let mut file = File::open(&args[1]).expect("Failed to open tape file");
        let mut contents = String::new();
        file.read_to_string(&mut contents).expect("Failed to read tape file");
        serde_json::from_str(&contents).expect("Failed to parse tape JSON")
    } else {
        env.generate_deterministic_tape(env.episode_len, 42)
    };

    let (obs0, gt0) = env.reset(Some(tape));
    let mut dumps = Vec::new();

    let actions = [0, 1, 2, 0, 1, 3, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2];

    for &a in &actions {
        let (obs, rew, done, gt) = env.step(a);
        dumps.push(StepDump {
            step: gt.step_idx,
            symbol: obs.symbol,
            sensor_a: obs.sensor_a,
            sensor_b: obs.sensor_b,
            warning_cue: obs.warning_cue,
            is_decision_window: obs.is_decision_window,
            reward: rew,
            i_t: gt.internal_reliability_i,
            x_t: gt.external_reliability_x,
            done,
        });
    }

    println!("{}", serde_json::to_string_pretty(&dumps).unwrap());
}
