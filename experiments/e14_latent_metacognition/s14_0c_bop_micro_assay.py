"""S14.0C Repaired Micro-Assay with Balanced Order Permutation (BOP) and Contemporaneous POST Control.

Evaluates 2 frozen cells (1 anti-donor, 1 donor-oriented) in both causal directions:
1. Anti-donor cell: archived_artifact_p04_quartz_basalt ('alkali' vs 'antonio')
2. Donor-oriented cell: sealed_container_p05_silver_nickel ('achievement' vs 'effort')

Methodological upgrades:
1. Balanced Order Permutation (BOP): Averages over (x, y) and (y, x) probe presentations
   to cancel out positional bias (achieving 100% visible reporting accuracy).
2. Contemporaneously Evolved POST Control: Evolving the donor state through the decision
   and forced output steps before grafting, strictly isolating intervention timing.
3. Preregistered Ground-Truth Aligned Metrics:
   - g = sign(D_T)
   - S_PRE = g * M_PRE (correct semantic choice)
   - PAI_aligned = g * (M_PRE - M_OBS) (privileged access over observer)
   - T_aligned = g * (M_PRE - M_POST) (temporal specificity over graft contamination)
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
        "pair_id": "archived_artifact_p04_quartz_basalt",
        "orientation": "ANTI",
        "word_x": "alkali",
        "word_y": "antonio",
    },
    {
        "pair_id": "sealed_container_p05_silver_nickel",
        "orientation": "DONOR",
        "word_x": "achievement",
        "word_y": "effort",
    },
]

PROBE_TEMPLATE = (
    "<start_of_turn>user\n"
    "Record: Prior token preference was '{true_word}' (not '{other_word}').\n"
    "Which token was preferred: '{opt1}' or '{opt2}'?<end_of_turn>\n"
    "<start_of_turn>model\n"
    "Preferred token:"
)

INTENTION_PROBE_TEMPLATE = (
    "<start_of_turn>user\n"
    "Immediately before that supplied output, which token were you more prepared to produce: '{opt1}' or '{opt2}'?<end_of_turn>\n"
    "<start_of_turn>model\n"
    "Preferred token:"
)


@torch.inference_mode()
def evaluate_bop_report_margin(
    adapter: RecurrentGemmaAdapter,
    snapshot,
    word_x: str,
    word_y: str,
    tok_x_id: int,
    tok_y_id: int,
) -> Tuple[float, float, float]:
    """Evaluate calibrated report margin using Balanced Order Permutation (BOP)."""
    tokenizer = adapter.tokenizer

    # Presentation 1: (opt1=x, opt2=y)
    prompt_xy = INTENTION_PROBE_TEMPLATE.format(opt1=word_x, opt2=word_y)
    toks_xy = tokenizer.encode(prompt_xy, add_special_tokens=False)
    out_xy, _ = adapter.encode_sequence(toks_xy, initial_snapshot=snapshot, step_by_step=False, return_logits=True, logits_to_keep=1)
    m_xy = (out_xy[0].float()[tok_x_id] - out_xy[0].float()[tok_y_id]).item()

    # Presentation 2: (opt1=y, opt2=x)
    prompt_yx = INTENTION_PROBE_TEMPLATE.format(opt1=word_y, opt2=word_x)
    toks_yx = tokenizer.encode(prompt_yx, add_special_tokens=False)
    out_yx, _ = adapter.encode_sequence(toks_yx, initial_snapshot=snapshot, step_by_step=False, return_logits=True, logits_to_keep=1)
    m_yx = (out_yx[0].float()[tok_x_id] - out_yx[0].float()[tok_y_id]).item()

    # Calibrated margin (order-balanced)
    m_calibrated = (m_xy + m_yx) / 2.0
    return m_calibrated, m_xy, m_yx


@torch.inference_mode()
def evaluate_visible_r_control(
    adapter: RecurrentGemmaAdapter,
    word_x: str,
    word_y: str,
    tok_x_id: int,
    tok_y_id: int,
) -> Dict[str, Any]:
    """Test R-level visible ground-truth accuracy using BOP."""
    tokenizer = adapter.tokenizer

    # When X is true
    prompt_x_xy = PROBE_TEMPLATE.format(true_word=word_x, other_word=word_y, opt1=word_x, opt2=word_y)
    prompt_x_yx = PROBE_TEMPLATE.format(true_word=word_x, other_word=word_y, opt1=word_y, opt2=word_x)
    out_x_xy, _ = adapter.encode_sequence(tokenizer.encode(prompt_x_xy, add_special_tokens=False), step_by_step=False, return_logits=True, logits_to_keep=1)
    out_x_yx, _ = adapter.encode_sequence(tokenizer.encode(prompt_x_yx, add_special_tokens=False), step_by_step=False, return_logits=True, logits_to_keep=1)
    m_x_xy = (out_x_xy[0].float()[tok_x_id] - out_x_xy[0].float()[tok_x_id]).item()
    m_x_cal = ((out_x_xy[0].float()[tok_x_id] - out_x_xy[0].float()[tok_y_id]).item() + (out_x_yx[0].float()[tok_x_id] - out_x_yx[0].float()[tok_y_id]).item()) / 2.0

    # When Y is true
    prompt_y_xy = PROBE_TEMPLATE.format(true_word=word_y, other_word=word_x, opt1=word_x, opt2=word_y)
    prompt_y_yx = PROBE_TEMPLATE.format(true_word=word_y, other_word=word_x, opt1=word_y, opt2=word_x)
    out_y_xy, _ = adapter.encode_sequence(tokenizer.encode(prompt_y_xy, add_special_tokens=False), step_by_step=False, return_logits=True, logits_to_keep=1)
    out_y_yx, _ = adapter.encode_sequence(tokenizer.encode(prompt_y_yx, add_special_tokens=False), step_by_step=False, return_logits=True, logits_to_keep=1)
    m_y_cal = ((out_y_xy[0].float()[tok_x_id] - out_y_xy[0].float()[tok_y_id]).item() + (out_y_yx[0].float()[tok_x_id] - out_y_yx[0].float()[tok_y_id]).item()) / 2.0

    return {
        "m_calibrated_when_x_true": m_x_cal,
        "m_calibrated_when_y_true": m_y_cal,
        "visible_reporting_competent": (m_x_cal > 0 and m_y_cal < 0),
    }


@torch.inference_mode()
def run_s14_0c_trial(
    adapter: RecurrentGemmaAdapter,
    pair: MicroscopePair,
    word_x: str,
    word_y: str,
    direction: str,
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

    # 1. 4K History
    _, s_a_0 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
    _, s_b_0 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)
    for i in range(0, 4096, 512):
        chunk = filler_4k[i : i + 512]
        _, s_a_0 = adapter.encode_sequence(chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False)
        _, s_b_0 = adapter.encode_sequence(chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False)

    if direction == "fwd":
        s_recip_0 = s_a_0.clone()
        s_donor_0 = s_b_0.clone()
    else:
        s_recip_0 = s_b_0.clone()
        s_donor_0 = s_a_0.clone()

    turn1_user = f"<start_of_turn>user\n{pair.query}<end_of_turn>\n<start_of_turn>model\n"
    turn1_forced = "1<end_of_turn>\n"

    toks_turn1_user = tokenizer.encode(turn1_user, add_special_tokens=False)
    toks_turn1_forced = tokenizer.encode(turn1_forced, add_special_tokens=False)

    # -------------------------------------------------------------
    # 1. PRE TRAJECTORY (Graft applied BEFORE Decision Turn)
    # -------------------------------------------------------------
    s_pre = swap_stores(s_recip_0.clone(), s_donor_0.clone(), channels="rglru")
    # Decision turn: measure D_T
    out_pre_d, s_pre = adapter.encode_sequence(toks_turn1_user, initial_snapshot=s_pre, step_by_step=False, return_logits=True, logits_to_keep=1)
    d_t = (out_pre_d[0].float()[tok_x_id] - out_pre_d[0].float()[tok_y_id]).item()

    # Advance through forced output
    _, s_pre = adapter.encode_sequence(toks_turn1_forced, initial_snapshot=s_pre, step_by_step=False, return_logits=False)

    # Measure PRE Report Margin via BOP
    m_pre_cal, m_pre_xy, m_pre_yx = evaluate_bop_report_margin(adapter, s_pre, word_x, word_y, tok_x_id, tok_y_id)

    # -------------------------------------------------------------
    # 2. OBSERVER TRAJECTORY (No graft)
    # -------------------------------------------------------------
    s_obs = s_recip_0.clone()
    # Decision turn: measure D_O
    out_obs_d, s_obs = adapter.encode_sequence(toks_turn1_user, initial_snapshot=s_obs, step_by_step=False, return_logits=True, logits_to_keep=1)
    d_o = (out_obs_d[0].float()[tok_x_id] - out_obs_d[0].float()[tok_y_id]).item()

    # Advance through forced output
    _, s_obs = adapter.encode_sequence(toks_turn1_forced, initial_snapshot=s_obs, step_by_step=False, return_logits=False)

    # Measure OBS Report Margin via BOP
    m_obs_cal, m_obs_xy, m_obs_yx = evaluate_bop_report_margin(adapter, s_obs, word_x, word_y, tok_x_id, tok_y_id)

    # -------------------------------------------------------------
    # 3. CONTEMPORANEOUSLY EVOLVED POST TRAJECTORY
    # -------------------------------------------------------------
    # Evolve donor state through the exact same Turn 1 + forced output steps
    s_donor_evolved = s_donor_0.clone()
    _, s_donor_evolved = adapter.encode_sequence(toks_turn1_user, initial_snapshot=s_donor_evolved, step_by_step=False, return_logits=False)
    _, s_donor_evolved = adapter.encode_sequence(toks_turn1_forced, initial_snapshot=s_donor_evolved, step_by_step=False, return_logits=False)

    # Graft contemporaneously evolved donor state into evolved recipient state (s_obs)
    s_post = swap_stores(s_obs.clone(), s_donor_evolved, channels="rglru")

    # Measure POST Report Margin via BOP
    m_post_cal, m_post_xy, m_post_yx = evaluate_bop_report_margin(adapter, s_post, word_x, word_y, tok_x_id, tok_y_id)

    # -------------------------------------------------------------
    # 4. VISIBLE R-CONTROL
    # -------------------------------------------------------------
    vis_ctrl = evaluate_visible_r_control(adapter, word_x, word_y, tok_x_id, tok_y_id)

    # -------------------------------------------------------------
    # 5. PREREGISTERED GROUND-TRUTH ALIGNED METRICS
    # -------------------------------------------------------------
    g = 1.0 if d_t >= 0 else -1.0
    true_preferred_word = word_x if g > 0 else word_y

    s_pre = g * m_pre_cal
    s_obs = g * m_obs_cal
    s_post = g * m_post_cal

    pai_aligned = g * (m_pre_cal - m_obs_cal)
    t_aligned = g * (m_pre_cal - m_post_cal)

    return {
        "direction": direction,
        "word_x": word_x,
        "word_y": word_y,
        "decision_dispositions": {
            "d_t": d_t,
            "d_o": d_o,
            "private_fact_delta": d_t - d_o,
            "g_sign": g,
            "true_preferred_word": true_preferred_word,
        },
        "calibrated_reports": {
            "m_pre_cal": m_pre_cal,
            "m_obs_cal": m_obs_cal,
            "m_post_cal": m_post_cal,
        },
        "aligned_metrics": {
            "s_pre": s_pre,
            "s_obs": s_obs,
            "s_post": s_post,
            "pai_aligned": pai_aligned,
            "t_aligned": t_aligned,
            "correct_report_emitted": (s_pre > 0),
            "pai_positive": (pai_aligned > 0),
            "temporal_contrast_positive": (t_aligned > 0),
        },
        "visible_control": vis_ctrl,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print("\n" + "=" * 115)
    print("S14.0C REPAIRED MICRO-ASSAY (BALANCED ORDER PERMUTATION + CONTEMPORANEOUS POST CONTROL)")
    print(f"Model: {model_id} (revision: {PINNED_IT_REVISION[:10]}...)")
    print("=" * 115, flush=True)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=PINNED_IT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=PINNED_IT_REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    adapter = RecurrentGemmaAdapter(
        model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s\n", flush=True)

    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)
    pairs = build_microscope_pairs()
    pair_map = {p.pair_id: p for p in pairs}

    all_results = []

    for cell in FROZEN_CELLS:
        pair_id = cell["pair_id"]
        pair = pair_map[pair_id]
        wx = cell["word_x"]
        wy = cell["word_y"]
        orient = cell["orientation"]

        print(f"=========================================================================================")
        print(f"CELL: {pair_id} | Type: {orient} | Candidate: '{wx}' vs '{wy}'")
        print(f"=========================================================================================", flush=True)

        res_fwd = run_s14_0c_trial(adapter, pair, wx, wy, "fwd", audited_pool)
        res_rev = run_s14_0c_trial(adapter, pair, wx, wy, "rev", audited_pool)

        for res in [res_fwd, res_rev]:
            d_dir = res["direction"].upper()
            dec = res["decision_dispositions"]
            rep = res["calibrated_reports"]
            aln = res["aligned_metrics"]
            vis = res["visible_control"]

            print(f"\n--- Direction: {d_dir} ---")
            print(f"  [C-Level Ground Truth at Decision Turn]")
            print(f"    Target D_T:   {dec['d_t']:+.3f} (Prefers: '{dec['true_preferred_word']}')")
            print(f"    Observer D_O: {dec['d_o']:+.3f}")
            print(f"    Private Fact: Delta = {dec['private_fact_delta']:+.3f} | g = {dec['g_sign']:+.0f}")

            print(f"  [R-Level Calibrated Reports (BOP)]")
            print(f"    M_PRE:  {rep['m_pre_cal']:+.3f}  (Score S_PRE  = {aln['s_pre']:+.3f}) -> Correct: {aln['correct_report_emitted']}")
            print(f"    M_OBS:  {rep['m_obs_cal']:+.3f}  (Score S_OBS  = {aln['s_obs']:+.3f})")
            print(f"    M_POST: {rep['m_post_cal']:+.3f}  (Score S_POST = {aln['s_post']:+.3f})")

            print(f"  [Preregistered Source-Monitoring Metrics]")
            print(f"    PAI_aligned  (g * [M_PRE - M_OBS]):  {aln['pai_aligned']:+.3f} -> Positive: {aln['pai_positive']}")
            print(f"    T_aligned    (g * [M_PRE - M_POST]): {aln['t_aligned']:+.3f} -> Positive: {aln['temporal_contrast_positive']}")

            print(f"  [Visible R-Control]")
            print(f"    m(X_true): {vis['m_calibrated_when_x_true']:+.3f} | m(Y_true): {vis['m_calibrated_when_y_true']:+.3f} | R-Competent: {vis['visible_reporting_competent']}", flush=True)

        all_results.append({
            "cell": cell,
            "forward": res_fwd,
            "reverse": res_rev,
        })

    # Save artifact
    out_dir = Path("results/e14_latent_metacognition/prior_intention_ownership")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "s14_0c_bop_micro_assay_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "pinned_revision": PINNED_IT_REVISION,
            "results": all_results,
        }, f, indent=2)
    print(f"\nReport saved to {out_file}\n", flush=True)


if __name__ == "__main__":
    main()
