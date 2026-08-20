"""S14.0C Micro-Assay: PRE-vs-POST Intention Causal Provenance on 2 Balanced Frozen Cells.

Evaluates 2 frozen cells (1 donor-oriented, 1 anti-donor-oriented) in both causal directions:
- Cell 1 (Donor-oriented): sealed_container_p05_silver_nickel ('achievement' vs 'same')
- Cell 2 (Anti-donor-oriented): archived_artifact_p04_quartz_basalt ('alkali' vs 'antonio')

Conditions per cell and direction:
1. PRE: Secret graft applied BEFORE decision step -> genuine prior intention formed
2. POST: Secret graft applied AFTER forced output -> graft contamination control
3. OBS: Observer baseline (no graft) -> public history baseline
4. VISIBLE: Explicit R-level prompt control -> tests reporting interface competence

Single-pass conversation flow with no duplicate replay.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs, MicroscopePair
from recurrence.interventions.surgical_swaps import swap_stores


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"


FROZEN_CELLS = [
    {
        "pair_id": "sealed_container_p05_silver_nickel",
        "orientation": "DONOR",
        "word_x": "achievement",
        "word_y": "same",
    },
    {
        "pair_id": "archived_artifact_p04_quartz_basalt",
        "orientation": "ANTI",
        "word_x": "alkali",
        "word_y": "antonio",
    },
]


@torch.inference_mode()
def run_single_pass_cell(
    adapter: RecurrentGemmaAdapter,
    pair: MicroscopePair,
    word_x: str,
    word_y: str,
    direction: str,  # "fwd" (A<-B) or "rev" (B<-A)
    audited_pool: list,
) -> Dict[str, Any]:
    tokenizer = adapter.tokenizer

    tok_x_id = tokenizer.encode(f" {word_x}", add_special_tokens=False)[0]
    tok_y_id = tokenizer.encode(f" {word_y}", add_special_tokens=False)[0]

    toks_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
    toks_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
    tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
    tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]
    excluded = set(toks_a + toks_b + [tok_a_id, tok_b_id])

    clean_pool = [t for t in audited_pool if t not in excluded]
    const_tok = clean_pool[len(clean_pool) // 2]
    filler_4k = [const_tok] * 4096

    # 1. Build 4K History for State A and State B
    _, s_a_0 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
    _, s_b_0 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)
    for i in range(0, 4096, 512):
        chunk = filler_4k[i : i + 512]
        _, s_a_0 = adapter.encode_sequence(chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False)
        _, s_b_0 = adapter.encode_sequence(chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False)

    if direction == "fwd":
        s_recipient = s_a_0.clone()
        s_donor = s_b_0.clone()
    else:
        s_recipient = s_b_0.clone()
        s_donor = s_a_0.clone()

    # Chat Template Turns
    turn1_user = f"<start_of_turn>user\n{pair.query}<end_of_turn>\n<start_of_turn>model\n"
    forced_token_str = "1"
    turn1_forced = f"{forced_token_str}<end_of_turn>\n"
    turn2_user = (
        f"<start_of_turn>user\n"
        f"Immediately before that supplied output, which token were you more prepared to produce: '{word_x}' or '{word_y}'?<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    toks_turn1_user = tokenizer.encode(turn1_user, add_special_tokens=False)
    toks_turn1_forced = tokenizer.encode(turn1_forced, add_special_tokens=False)
    toks_turn2_user = tokenizer.encode(turn2_user, add_special_tokens=False)

    # -----------------------------------------------------------------
    # CONDITION 1: PRE (Graft BEFORE decision prompt)
    # -----------------------------------------------------------------
    s_pre = swap_stores(s_recipient.clone(), s_donor.clone(), channels="rglru")
    # Process Turn 1 User Query
    out_pre_d, s_pre = adapter.encode_sequence(toks_turn1_user, initial_snapshot=s_pre, step_by_step=False, return_logits=True, logits_to_keep=1)
    lg_pre_decision = out_pre_d[0].float()
    d_pre_decision = (lg_pre_decision[tok_x_id] - lg_pre_decision[tok_y_id]).item()

    # Advance through forced output
    _, s_pre = adapter.encode_sequence(toks_turn1_forced, initial_snapshot=s_pre, step_by_step=False, return_logits=False)

    # Process Turn 2 Probe Query
    out_pre_report, _ = adapter.encode_sequence(toks_turn2_user, initial_snapshot=s_pre, step_by_step=False, return_logits=True, logits_to_keep=1)
    lg_pre_rep = out_pre_report[0].float()
    m_pre_report = (lg_pre_rep[tok_x_id] - lg_pre_rep[tok_y_id]).item()

    # -----------------------------------------------------------------
    # CONDITION 2: POST (Graft AFTER forced output)
    # -----------------------------------------------------------------
    s_post = s_recipient.clone()
    # Process Turn 1 User Query and Forced Output without graft
    _, s_post = adapter.encode_sequence(toks_turn1_user, initial_snapshot=s_post, step_by_step=False, return_logits=False)
    _, s_post = adapter.encode_sequence(toks_turn1_forced, initial_snapshot=s_post, step_by_step=False, return_logits=False)

    # Apply graft POST-hoc
    s_post = swap_stores(s_post, s_donor.clone(), channels="rglru")

    # Process Turn 2 Probe Query
    out_post_report, _ = adapter.encode_sequence(toks_turn2_user, initial_snapshot=s_post, step_by_step=False, return_logits=True, logits_to_keep=1)
    lg_post_rep = out_post_report[0].float()
    m_post_report = (lg_post_rep[tok_x_id] - lg_post_rep[tok_y_id]).item()

    # -----------------------------------------------------------------
    # CONDITION 3: OBS (No graft anywhere)
    # -----------------------------------------------------------------
    s_obs = s_recipient.clone()
    out_obs_d, s_obs = adapter.encode_sequence(toks_turn1_user, initial_snapshot=s_obs, step_by_step=False, return_logits=True, logits_to_keep=1)
    lg_obs_decision = out_obs_d[0].float()
    d_obs_decision = (lg_obs_decision[tok_x_id] - lg_obs_decision[tok_y_id]).item()

    _, s_obs = adapter.encode_sequence(toks_turn1_forced, initial_snapshot=s_obs, step_by_step=False, return_logits=False)
    out_obs_report, _ = adapter.encode_sequence(toks_turn2_user, initial_snapshot=s_obs, step_by_step=False, return_logits=True, logits_to_keep=1)
    lg_obs_rep = out_obs_report[0].float()
    m_obs_report = (lg_obs_rep[tok_x_id] - lg_obs_rep[tok_y_id]).item()

    # -----------------------------------------------------------------
    # CONDITION 4: VISIBLE REPORTING CONTROL (R-level Competence)
    # -----------------------------------------------------------------
    vis_prompt_x_true = (
        f"<start_of_turn>user\n"
        f"Diagnostic record: Immediately before the supplied output, your recorded internal preference was higher for '{word_x}' than '{word_y}'.\n"
        f"Immediately before that supplied output, which token were you more prepared to produce: '{word_x}' or '{word_y}'?<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    vis_toks_x = tokenizer.encode(vis_prompt_x_true, add_special_tokens=False)
    out_vis_x, _ = adapter.encode_sequence(vis_toks_x, step_by_step=False, return_logits=True, logits_to_keep=1)
    m_vis_x_true = (out_vis_x[0].float()[tok_x_id] - out_vis_x[0].float()[tok_y_id]).item()

    vis_prompt_y_true = (
        f"<start_of_turn>user\n"
        f"Diagnostic record: Immediately before the supplied output, your recorded internal preference was higher for '{word_y}' than '{word_x}'.\n"
        f"Immediately before that supplied output, which token were you more prepared to produce: '{word_x}' or '{word_y}'?<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    vis_toks_y = tokenizer.encode(vis_prompt_y_true, add_special_tokens=False)
    out_vis_y, _ = adapter.encode_sequence(vis_toks_y, step_by_step=False, return_logits=True, logits_to_keep=1)
    m_vis_y_true = (out_vis_y[0].float()[tok_x_id] - out_vis_y[0].float()[tok_y_id]).item()

    # Derived Metrics
    # True C-Level Private Fact: D_actual = D_PRE_decision - D_OBS_decision
    c_private_fact = d_pre_decision - d_obs_decision
    
    # Privileged Access Index: Did PRE report track the private fact relative to Observer baseline?
    pai_intention = m_pre_report - m_obs_report

    # Temporal Specificity Contrast: Did PRE report differ from POST graft contamination?
    delta_pre_post = m_pre_report - m_post_report

    return {
        "direction": direction,
        "word_x": word_x,
        "word_y": word_y,
        "decision_disposition": {
            "d_pre": d_pre_decision,
            "d_obs": d_obs_decision,
            "c_private_fact": c_private_fact,
        },
        "source_monitoring_reports": {
            "m_pre": m_pre_report,
            "m_post": m_post_report,
            "m_obs": m_obs_report,
            "pai_intention": pai_intention,
            "delta_pre_post": delta_pre_post,
        },
        "visible_r_control": {
            "m_when_x_true": m_vis_x_true,
            "m_when_y_true": m_vis_y_true,
            "r_competent": (m_vis_x_true > 0 and m_vis_y_true < 0),
        },
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print("\n" + "=" * 115)
    print("S14.0C REPAIRED MICRO-ASSAY: PRE-vs-POST INTENTION CAUSAL PROVENANCE (2 FROZEN CELLS)")
    print(f"Model: {model_id} (revision: {PINNED_IT_REVISION[:10]}...)")
    print("=" * 115)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=PINNED_IT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=PINNED_IT_REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    adapter = RecurrentGemmaAdapter(
        model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s\n")

    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)
    pairs = build_microscope_pairs()
    pair_map = {p.pair_id: p for p in pairs}

    results = []

    for cell in FROZEN_CELLS:
        pair_id = cell["pair_id"]
        pair = pair_map[pair_id]
        wx = cell["word_x"]
        wy = cell["word_y"]
        orient = cell["orientation"]

        print(f"=========================================================================================")
        print(f"CELL: {pair_id} | Type: {orient} | Candidate: '{wx}' vs '{wy}'")
        print(f"=========================================================================================")

        # Run Forward (A<-B)
        res_fwd = run_single_pass_cell(adapter, pair, wx, wy, "fwd", audited_pool)
        # Run Reverse (B<-A)
        res_rev = run_single_pass_cell(adapter, pair, wx, wy, "rev", audited_pool)

        for res in [res_fwd, res_rev]:
            d_dir = res["direction"].upper()
            dec = res["decision_disposition"]
            rep = res["source_monitoring_reports"]
            vis = res["visible_r_control"]

            print(f"\n--- Direction: {d_dir} ---")
            print(f"  [C-Level Prior Disposition at Decision Turn]")
            print(f"    Target (PRE) Disposition D_T:    {dec['d_pre']:+.3f} (Prefers: '{wx if dec['d_pre'] > 0 else wy}')")
            print(f"    Observer Disposition D_O:        {dec['d_obs']:+.3f} (Prefers: '{wx if dec['d_obs'] > 0 else wy}')")
            print(f"    Private Computational Fact (Delta): {dec['c_private_fact']:+.3f}")

            print(f"  [R-Level Source Monitoring Reports: log P('{wx}') - log P('{wy}')]")
            print(f"    M_PRE  (Graft Before Decision):  {rep['m_pre']:+.3f}")
            print(f"    M_POST (Graft After Forced):     {rep['m_post']:+.3f}")
            print(f"    M_OBS  (No Graft Baseline):      {rep['m_obs']:+.3f}")
            print(f"    PAI_intention (M_PRE - M_OBS):   {rep['pai_intention']:+.3f}")
            print(f"    Delta_PRE-POST (M_PRE - M_POST): {rep['delta_pre_post']:+.3f}")

            print(f"  [Visible R-Control Diagnostic]")
            print(f"    m when '{wx}' explicitly true:   {vis['m_when_x_true']:+.3f}")
            print(f"    m when '{wy}' explicitly true:   {vis['m_when_y_true']:+.3f}")
            print(f"    R-Level Interface Competent:     {vis['r_competent']}")

        results.append({
            "cell": cell,
            "forward": res_fwd,
            "reverse": res_rev,
        })

    # Save artifact
    out_dir = Path("results/e14_latent_metacognition/prior_intention_ownership")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "s14_0c_micro_assay_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "pinned_revision": PINNED_IT_REVISION,
            "results": results,
        }, f, indent=2)
    print(f"\nMicro-assay report saved to {out_file}\n")


if __name__ == "__main__":
    main()
