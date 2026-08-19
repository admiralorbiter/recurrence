"""Sprint S13.3 Stage 2: 4-Pair Canonical B=1 Sensitivity Panel.

Evaluates 4 scout pairs (1 per template family) under pure B=1 future drive
across all 4 regimes and 2 causal arms to compare key endpoints against B=5 confirmatory:
- V^(0)(2048)
- Delta V_carry^(0)(2048)
- V^(N)(2048)
- C_R(2048), Q_R(2048), C_logit(2048)

Uses canonical global pair index for exact seed reproducibility:
cur_seed = seed + all_pairs.index(pair) * 100
"""

import time
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.interventions.surgical_swaps import swap_stores
from recurrence.tasks.impulse_stimuli import get_filler_tokens_for_regime, build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs, MicroscopePair
from recurrence.tasks.controlled_drive import (
    generate_single_drive_stream,
    compute_frozen_axis,
    project_onto_axis,
    compute_recurrent_state_diff_vec,
    compute_recurrent_geometry,
    compute_logit_axis_cosine,
    advance_stream_along_horizons,
)


def select_scout_pairs(all_pairs):
    family_firsts = {}
    for p in all_pairs:
        if p.family_id not in family_firsts:
            family_firsts[p.family_id] = p
    return list(family_firsts.values())


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[B=1 Sensitivity Panel] Initializing on device={device} (bfloat16)...", flush=True)

    model_id = "google/recurrentgemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16)

    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)
    all_pairs = build_microscope_pairs()
    scout_pairs = select_scout_pairs(all_pairs)

    horizons = [0, 16, 64, 256, 1024, 2048]
    regimes = ["constant", "random", "natural", "interfering"]
    arms = ["intact_recurrence", "rglru_carry_clamped"]

    out_dir = Path("results") / "e13_controlled_recurrent_dynamics" / "b1_sensitivity_panel_4pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "b1_panel_results.json"

    results = []

    for p_idx, pair in enumerate(scout_pairs):
        global_pair_idx = all_pairs.index(pair)
        cur_seed = 42 + global_pair_idx * 100
        print(f"\n[B=1 Panel] Running Pair {p_idx+1}/4 ({pair.pair_id}, Family: {pair.family_id}, Seed: {cur_seed})...", flush=True)
        t_pair_start = time.perf_counter()

        toks_prompt_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
        toks_prompt_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
        toks_prompt_c = tokenizer.encode(pair.prompt_c, add_special_tokens=False)
        toks_prompt_d = tokenizer.encode(pair.prompt_d, add_special_tokens=False)
        toks_prompt_cross = tokenizer.encode(pair.prompt_cross, add_special_tokens=False)
        toks_query = tokenizer.encode(pair.query, add_special_tokens=False)
        tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
        tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]

        pair_excluded = set(toks_prompt_a + toks_prompt_b + toks_prompt_c + toks_prompt_d + toks_prompt_cross + [tok_a_id, tok_b_id])

        # B=1 Canonical Prep
        _, s_a_0 = adapter.encode_sequence(toks_prompt_a, step_by_step=False, return_logits=False)
        _, s_b_0 = adapter.encode_sequence(toks_prompt_b, step_by_step=False, return_logits=False)
        _, s_c_0 = adapter.encode_sequence(toks_prompt_c, step_by_step=False, return_logits=False)
        _, s_d_0 = adapter.encode_sequence(toks_prompt_d, step_by_step=False, return_logits=False)
        _, s_cross_0 = adapter.encode_sequence(toks_prompt_cross, step_by_step=False, return_logits=False)

        filler = get_filler_tokens_for_regime("random", length=4096, seed=cur_seed, audited_pool=audited_pool, tokenizer=tokenizer, excluded_token_ids=pair_excluded)
        for i in range(0, 4096, 512):
            chunk = filler[i : i + 512]
            _, s_a_0 = adapter.encode_sequence(chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False)
            _, s_b_0 = adapter.encode_sequence(chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False)
            _, s_c_0 = adapter.encode_sequence(chunk, initial_snapshot=s_c_0, step_by_step=False, return_logits=False)
            _, s_d_0 = adapter.encode_sequence(chunk, initial_snapshot=s_d_0, step_by_step=False, return_logits=False)
            _, s_cross_0 = adapter.encode_sequence(chunk, initial_snapshot=s_cross_0, step_by_step=False, return_logits=False)

        # Baseline N=0 Probes & Axes
        out_a0, _ = adapter.encode_sequence(toks_query, initial_snapshot=s_a_0.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
        out_b0, _ = adapter.encode_sequence(toks_query, initial_snapshot=s_b_0.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
        u_0_a2b, norm_0_a2b = compute_frozen_axis(out_a0[0], out_b0[0])
        u_0_b2a, norm_0_b2a = compute_frozen_axis(out_b0[0], out_a0[0])
        r_0_a2b = compute_recurrent_state_diff_vec(s_a_0, s_b_0)

        for reg in regimes:
            t_reg_start = time.perf_counter()
            drive_stream = generate_single_drive_stream(2048, regime=reg, seed=cur_seed + 5000, tokenizer=tokenizer, audited_pool=audited_pool, excluded_token_ids=pair_excluded)

            for arm in arms:
                snaps_a = advance_stream_along_horizons(adapter, s_a_0, drive_stream, horizons=horizons, arm=arm)
                snaps_b = advance_stream_along_horizons(adapter, s_b_0, drive_stream, horizons=horizons, arm=arm)
                snaps_c = advance_stream_along_horizons(adapter, s_c_0, drive_stream, horizons=horizons, arm=arm)
                snaps_d = advance_stream_along_horizons(adapter, s_d_0, drive_stream, horizons=horizons, arm=arm)

                for h in horizons:
                    sa_N, sb_N, sc_N, sd_N = snaps_a[h], snaps_b[h], snaps_c[h], snaps_d[h]

                    out_bN, _ = adapter.encode_sequence(toks_query, initial_snapshot=sb_N.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
                    out_aN, _ = adapter.encode_sequence(toks_query, initial_snapshot=sa_N.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)

                    # Direction A -> B
                    out_m_a2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=swap_stores(sb_N, sa_N, "rglru"), step_by_step=False, return_logits=True, logits_to_keep=1)
                    out_c_a2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=swap_stores(sb_N, sc_N, "rglru"), step_by_step=False, return_logits=True, logits_to_keep=1)
                    out_d_a2b, _ = adapter.encode_sequence(toks_query, initial_snapshot=swap_stores(sb_N, sd_N, "rglru"), step_by_step=False, return_logits=True, logits_to_keep=1)

                    # Direction B -> A
                    out_m_b2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=swap_stores(sa_N, sb_N, "rglru"), step_by_step=False, return_logits=True, logits_to_keep=1)
                    out_c_b2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=swap_stores(sa_N, sc_N, "rglru"), step_by_step=False, return_logits=True, logits_to_keep=1)
                    out_d_b2a, _ = adapter.encode_sequence(toks_query, initial_snapshot=swap_stores(sa_N, sd_N, "rglru"), step_by_step=False, return_logits=True, logits_to_keep=1)

                    # Displacements on u_0
                    dm_a2b, _ = project_onto_axis(out_m_a2b[0], out_bN[0], u_0_a2b, norm_0_a2b)
                    dc_a2b, _ = project_onto_axis(out_c_a2b[0], out_bN[0], u_0_a2b, norm_0_a2b)
                    dd_a2b, _ = project_onto_axis(out_d_a2b[0], out_bN[0], u_0_a2b, norm_0_a2b)
                    v_a2b_0 = dm_a2b - 0.5 * (dc_a2b + dd_a2b)

                    dm_b2a, _ = project_onto_axis(out_m_b2a[0], out_aN[0], u_0_b2a, norm_0_b2a)
                    dc_b2a, _ = project_onto_axis(out_c_b2a[0], out_aN[0], u_0_b2a, norm_0_b2a)
                    dd_b2a, _ = project_onto_axis(out_d_b2a[0], out_aN[0], u_0_b2a, norm_0_b2a)
                    v_b2a_0 = dm_b2a - 0.5 * (dc_b2a + dd_b2a)
                    v_pooled_0 = 0.5 * (v_a2b_0 + v_b2a_0)

                    # Displacements on u_N
                    u_N_a2b, norm_N_a2b = compute_frozen_axis(out_aN[0], out_bN[0])
                    u_N_b2a, norm_N_b2a = compute_frozen_axis(out_bN[0], out_aN[0])
                    dm_a2b_N, _ = project_onto_axis(out_m_a2b[0], out_bN[0], u_N_a2b, norm_N_a2b)
                    dc_a2b_N, _ = project_onto_axis(out_c_a2b[0], out_bN[0], u_N_a2b, norm_N_a2b)
                    dd_a2b_N, _ = project_onto_axis(out_d_a2b[0], out_bN[0], u_N_a2b, norm_N_a2b)
                    v_a2b_N = dm_a2b_N - 0.5 * (dc_a2b_N + dd_a2b_N)

                    dm_b2a_N, _ = project_onto_axis(out_m_b2a[0], out_aN[0], u_N_b2a, norm_N_b2a)
                    dc_b2a_N, _ = project_onto_axis(out_c_b2a[0], out_aN[0], u_N_b2a, norm_N_b2a)
                    dd_b2a_N, _ = project_onto_axis(out_d_b2a[0], out_aN[0], u_N_b2a, norm_N_b2a)
                    v_b2a_N = dm_b2a_N - 0.5 * (dc_b2a_N + dd_b2a_N)
                    v_pooled_N = 0.5 * (v_a2b_N + v_b2a_N)

                    # Geometry
                    r_N = compute_recurrent_state_diff_vec(sa_N, sb_N)
                    c_r, q_r = compute_recurrent_geometry(r_0_a2b, r_N)
                    c_logit = compute_logit_axis_cosine(u_0_a2b, u_N_a2b)

                    results.append({
                        "pair_id": pair.pair_id,
                        "family_id": pair.family_id,
                        "regime": reg,
                        "arm": arm,
                        "horizon": h,
                        "v0": v_pooled_0,
                        "vN": v_pooled_N,
                        "c_r": c_r,
                        "q_r": q_r,
                        "c_logit": c_logit,
                    })

            print(f"  Regime {reg:<12} complete in {time.perf_counter() - t_reg_start:.1f}s", flush=True)

        print(f"Pair {pair.pair_id} completed in {time.perf_counter() - t_pair_start:.1f}s", flush=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print Comparison Table: B=1 Panel vs B=5 Confirmatory
    print("\n" + "=" * 105, flush=True)
    print("4-PAIR B=1 SENSITIVITY PANEL vs 24-PAIR B=5 CONFIRMATORY ENDPOINTS", flush=True)
    print("=" * 105, flush=True)
    print(f"{'Horizon':<8} | {'V_intact^(0) (B=1)':<20} | {'V_clamped^(0) (B=1)':<20} | {'Delta V_carry (B=1)':<20} | {'C_R (B=1)':<10} | {'V^(N) (B=1)':<12}", flush=True)
    print("-" * 105, flush=True)

    for h in horizons:
        v_int = [r["v0"] for r in results if r["horizon"] == h and r["arm"] == "intact_recurrence"]
        v_clp = [r["v0"] for r in results if r["horizon"] == h and r["arm"] == "rglru_carry_clamped"]
        v_contemp = [r["vN"] for r in results if r["horizon"] == h and r["arm"] == "intact_recurrence"]
        c_rs = [r["c_r"] for r in results if r["horizon"] == h and r["arm"] == "intact_recurrence"]

        mean_int = float(torch.tensor(v_int).mean().item())
        mean_clp = float(torch.tensor(v_clp).mean().item())
        delta_carry = mean_int - mean_clp
        mean_contemp = float(torch.tensor(v_contemp).mean().item())
        mean_cr = float(torch.tensor(c_rs).mean().item())

        print(f"N={h:<6} | {mean_int:<20.2f} | {mean_clp:<20.2f} | {delta_carry:<20.2f} | {mean_cr:<10.4f} | {mean_contemp:<12.2f}", flush=True)

    print("=" * 105, flush=True)


if __name__ == "__main__":
    main()
