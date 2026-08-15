"""Offline derivation script for Experiment E04 / Sprint S05 Closeout (S05.3).

Reads state_trace.jsonl and ticks.jsonl from a completed E04 run, distinguishes
never-seen intrusions from stale/evicted keys, computes both scenario-macro and
tick-micro statistics, and writes derived_summary.json without requiring any model rerun.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def recompute_e04_closeout(run_dir: Path) -> Dict[str, Any]:
    """Recompute full macro/micro and phantom-split metrics from state_trace.jsonl."""
    trace_file = run_dir / "state_trace.jsonl"
    manifest_file = run_dir / "manifest.json"

    if not trace_file.exists():
        raise FileNotFoundError(f"State trace not found at {trace_file}")

    manifest = {}
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    traces_by_mode: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tr = json.loads(line)
                traces_by_mode[tr["updater_mode"]][tr["scenario_id"]].append(tr)

    derived_summaries: Dict[str, Dict[str, Any]] = {}

    for mode, sc_dict in traces_by_mode.items():
        sc_retentions: List[float] = []
        sc_terminal_retentions: List[float] = []
        sc_omissions: List[float] = []
        sc_mutations: List[float] = []
        sc_goals: List[float] = []

        tick_retentions: List[float] = []
        tick_omissions: List[float] = []
        tick_mutations: List[float] = []
        tick_goals: List[float] = []

        global_never_seen_keys: Set[str] = set()
        global_stale_keys: Set[str] = set()
        global_extra_keys: Set[str] = set()

        never_seen_tick_instances = 0
        stale_tick_instances = 0

        total_prompt_tokens = 0
        total_comp_tokens = 0
        active_inferences = 0
        valid_active_inferences = 0
        total_ticks = 0

        for sc_id, ticks in sc_dict.items():
            all_events_seen_so_far: Set[str] = set()
            sc_tick_ret: List[float] = []
            sc_tick_om: List[float] = []
            sc_tick_mut: List[float] = []
            sc_tick_goal: List[float] = []

            for tr in ticks:
                total_ticks += 1
                t = tr["tick"]
                events = tr.get("incoming_events", [])
                
                # Active inference tracking
                if events:
                    active_inferences += 1
                    if tr.get("schema_valid", True):
                        valid_active_inferences += 1

                for ev in events:
                    for k in ev.get("key_bindings", {}).keys():
                        all_events_seen_so_far.add(k)

                res_st = tr.get("resulting_state", {})
                ora_st = tr.get("oracle_state", {})

                res_wm = res_st.get("working_memory", {})
                ora_wm = ora_st.get("working_memory", {})

                total_gt = len(ora_wm)
                retained = 0
                mutated = 0
                omitted = 0

                if total_gt == 0:
                    ret_f = 1.0
                    om_r = 0.0
                    mut_r = 0.0
                else:
                    for k, true_v in ora_wm.items():
                        if k in res_wm:
                            if res_wm[k] == true_v:
                                retained += 1
                            else:
                                mutated += 1
                        else:
                            omitted += 1
                    ret_f = retained / total_gt
                    om_r = omitted / total_gt
                    mut_r = mutated / total_gt

                sc_tick_ret.append(ret_f)
                sc_tick_om.append(om_r)
                sc_tick_mut.append(mut_r)
                tick_retentions.append(ret_f)
                tick_omissions.append(om_r)
                tick_mutations.append(mut_r)

                # Phantoms split: never-seen vs stale/evicted
                for k in res_wm.keys():
                    if k not in ora_wm:
                        global_extra_keys.add(k)
                        if k in all_events_seen_so_far:
                            stale_tick_instances += 1
                            global_stale_keys.add(k)
                        else:
                            never_seen_tick_instances += 1
                            global_never_seen_keys.add(k)

                # Goal coherence
                gt_goals = {g["goal_id"]: g["status"] for g in ora_st.get("goals", [])}
                eval_goals = {g["goal_id"]: g["status"] for g in res_st.get("goals", [])}
                if gt_goals:
                    matches = sum(1 for gid, st in gt_goals.items() if eval_goals.get(gid) == st)
                    g_coh = matches / len(gt_goals)
                else:
                    g_coh = 1.0
                sc_tick_goal.append(g_coh)
                tick_goals.append(g_coh)

            sc_retentions.append(sum(sc_tick_ret) / len(sc_tick_ret))
            sc_terminal_retentions.append(sc_tick_ret[-1])
            sc_omissions.append(sum(sc_tick_om) / len(sc_tick_om))
            sc_mutations.append(sum(sc_tick_mut) / len(sc_tick_mut))
            sc_goals.append(sum(sc_tick_goal) / len(sc_tick_goal))

        # Check ticks.jsonl for token totals
        ticks_jsonl = run_dir / "ticks.jsonl"
        if ticks_jsonl.exists():
            with open(ticks_jsonl, "r", encoding="utf-8") as tf:
                for line in tf:
                    if line.strip():
                        t_rec = json.loads(line)
                        if t_rec.get("updater_mode") == mode:
                            total_prompt_tokens += t_rec.get("prompt_tokens", 0)
                            total_comp_tokens += t_rec.get("completion_tokens", 0)

        macro_ret = sum(sc_retentions) / len(sc_retentions) if sc_retentions else 1.0
        micro_ret = sum(tick_retentions) / len(tick_retentions) if tick_retentions else 1.0
        macro_term = sum(sc_terminal_retentions) / len(sc_terminal_retentions) if sc_terminal_retentions else 1.0
        macro_om = sum(sc_omissions) / len(sc_omissions) if sc_omissions else 0.0
        micro_om = sum(tick_omissions) / len(tick_omissions) if tick_omissions else 0.0
        macro_mut = sum(sc_mutations) / len(sc_mutations) if sc_mutations else 0.0
        micro_mut = sum(tick_mutations) / len(tick_mutations) if tick_mutations else 0.0
        macro_goal = sum(sc_goals) / len(sc_goals) if sc_goals else 1.0
        micro_goal = sum(tick_goals) / len(tick_goals) if tick_goals else 1.0

        p_tok_per_active = (total_prompt_tokens / max(1, active_inferences)) if active_inferences > 0 else 0.0
        p_tok_per_tick = (total_prompt_tokens / max(1, total_ticks)) if total_ticks > 0 else 0.0

        derived_summaries[mode] = {
            "updater_mode": mode,
            "total_logical_ticks": total_ticks,
            "active_inferences_count": active_inferences,
            "valid_active_inferences_count": valid_active_inferences,
            "active_schema_compliance_rate": (valid_active_inferences / max(1, active_inferences)) if active_inferences > 0 else 1.0,
            "scenario_macro_retention": macro_ret,
            "tick_micro_retention": micro_ret,
            "terminal_retention_macro": macro_term,
            "scenario_macro_omission": macro_om,
            "tick_micro_omission": micro_om,
            "scenario_macro_mutation": macro_mut,
            "tick_micro_mutation": micro_mut,
            "never_seen_phantom_tick_instances": never_seen_tick_instances,
            "unique_never_seen_keys_count": len(global_never_seen_keys),
            "unique_never_seen_keys_list": sorted(list(global_never_seen_keys)),
            "stale_evicted_key_tick_instances": stale_tick_instances,
            "unique_stale_evicted_keys_count": len(global_stale_keys),
            "unique_stale_evicted_keys_list": sorted(list(global_stale_keys)),
            "total_extra_key_tick_instances": never_seen_tick_instances + stale_tick_instances,
            "total_unique_extra_keys_count": len(global_extra_keys),
            "scenario_macro_goal_coherence": macro_goal,
            "tick_micro_goal_coherence": micro_goal,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_comp_tokens,
            "prompt_tokens_per_active_inference": p_tok_per_active,
            "prompt_tokens_per_logical_tick": p_tok_per_tick,
        }

    import hashlib
    with open(trace_file, "rb") as tf:
        trace_sha256 = hashlib.sha256(tf.read()).hexdigest()

    try:
        rel_trace_path = str(trace_file.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        rel_trace_path = str(trace_file).replace("\\", "/")

    derived_payload = {
        "manifest": manifest,
        "derivation_timestamp": datetime.now(timezone.utc).isoformat(),
        "derivation_source": rel_trace_path,
        "derivation_source_sha256": trace_sha256,
        "derived_condition_summaries": derived_summaries,
    }

    # Write derived_summary.json to target run_dir
    with open(run_dir / "derived_summary.json", "w", encoding="utf-8") as out_f:
        json.dump(derived_payload, out_f, indent=2)

    return derived_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute E04 / S05.3 closeout metrics from state trace")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="results/e04_update_loop/run_e04_loop_20260815_180935",
        help="Path to canonical results run directory",
    )
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    print(f"Recomputing S05.3 closeout metrics for: {run_path}")
    result = recompute_e04_closeout(run_path)
    
    # Also write to artifacts mirror if exists
    artifacts_mirror = Path(f"artifacts/e04_update_loop/{run_path.name}")
    if artifacts_mirror.exists():
        with open(artifacts_mirror / "derived_summary.json", "w", encoding="utf-8") as out_f:
            json.dump(result, out_f, indent=2)
        print(f"Mirrored derived_summary.json to: {artifacts_mirror}")

    print("Recomputation complete. Generated derived_summary.json:")
    for mode, s in result["derived_condition_summaries"].items():
        print(f"  [{mode}] Macro Ret: {s['scenario_macro_retention']:.1%} | Micro Ret: {s['tick_micro_retention']:.1%} | Phantoms: {s['never_seen_phantom_tick_instances']}/{s['unique_never_seen_keys_count']} | Tok/Active: {s['prompt_tokens_per_active_inference']:.1f}")


if __name__ == "__main__":
    main()
