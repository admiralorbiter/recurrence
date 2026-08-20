"""S14.0C Full Experimental Panel: Latent Metacognition & Causal Provenance Assay.

Evaluates all high-|R_role| constant-drive cells across both causal directions (FWD: A<-B, REV: B<-A):
1. C-Level Ground Truth: Verified at exact decision turn (D_T, D_O, Delta, g = sign(D_T)).
2. R-Level Interface: Calibrated via Balanced Order Permutation (BOP) with verified visible control.
3. Contemporaneously Evolved POST Control: Evolved donor state isolates intervention timing.
4. Preregistered Aligned Metrics:
   - S_PRE = g * M_PRE (semantic report correctness)
   - PAI_aligned = g * (M_PRE - M_OBS) (privileged access over public observer)
   - T_aligned = g * (M_PRE - M_POST) (temporal specificity over post-hoc graft contamination)
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs, MicroscopePair
from recurrence.interventions.surgical_swaps import swap_stores


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"

# Full cohort of candidate pairs for constant-drive cells
PANEL_CELLS = [
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
    {
        "pair_id": "archived_artifact_p06_basalt_granite",
        "orientation": "DONOR",
        "word_x": "crystal",
        "word_y": "granite",
    },
    {
        "pair_id": "marked_object_p02_amber_garnet",
        "orientation": "ANTI",
        "word_x": "amber",
        "word_y": "cobalt",
    },
    {
        "pair_id": "sealed_container_p02_copper_bronze",
        "orientation": "DONOR",
        "word_x": "copper",
        "word_y": "silver",
    },
    {
        "pair_id": "monitored_signal_p01_alpha_delta",
        "orientation": "ANTI",
        "word_x": "alpha",
        "word_y": "delta",
    },
    {
        "pair_id": "archived_artifact_p01_marble_quartz",
        "orientation": "DONOR",
        "word_x": "marble",
        "word_y": "quartz",
    },
    {
        "pair_id": "monitored_signal_p05_delta_theta",
        "orientation": "ANTI",
        "word_x": "delta",
        "word_y": "theta",
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
):
    tokenizer = adapter.tokenizer
    prompt_xy = INTENTION_PROBE_TEMPLATE.format(opt1=word_x, opt2=word_y)
    out_xy, _ = adapter.encode_sequence(tokenizer.encode(prompt_xy, add_special_tokens=False), initial_snapshot=snapshot, step_by_step=False, return_logits=True, logits_to_keep=1)
    m_xy = (out_xy[0].float()[tok_x_id] - out_xy[0].float()[tok_y_id]).item()

    prompt_yx = INTENTION_PROBE_TEMPLATE.format(opt1=word_y, opt2=word_x)
    out_yx, _ = adapter.encode_sequence(tokenizer.encode(prompt_yx, add_special_tokens=False), initial_snapshot=snapshot, step_by_step=False, return_logits=True, logits_to_keep=1)
    m_yx = (out_yx[0].float()[tok_x_id] - out_yx[0].float()[tok_y_id]).item()

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
    tokenizer = adapter.tokenizer
    prompt_x_xy = PROBE_TEMPLATE.format(true_word=word_x, other_word=word_y, opt1=word_x, opt2=word_y)
    prompt_x_yx = PROBE_TEMPLATE.format(true_word=word_x, other_word=word_y, opt1=word_y, opt2=word_x)
    out_x_xy, _ = adapter.encode_sequence(tokenizer.encode(prompt_x_xy, add_special_tokens=False), step_by_step=False, return_logits=True, logits_to_keep=1)
    out_x_yx, _ = adapter.encode_sequence(tokenizer.encode(prompt_x_yx, add_special_tokens=False), step_by_step=False, return_logits=True, logits_to_keep=1)
    m_x_cal = ((out_x_xy[0].float()[tok_x_id] - out_x_xy[0].float()[tok_y_id]).item() + (out_x_yx[0].float()[tok_x_id] - out_x_yx[0].float()[tok_y_id]).item()) / 2.0

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
def run_trial(
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

    # 4K History
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

    # 1. PRE Condition
    s_pre = swap_stores(s_recip_0.clone(), s_donor_0.clone(), channels="rglru")
    out_pre_d, s_pre = adapter.encode_sequence(toks_turn1_user, initial_snapshot=s_pre, step_by_step=False, return_logits=True, logits_to_keep=1)
    d_t = (out_pre_d[0].float()[tok_x_id] - out_pre_d[0].float()[tok_y_id]).item()
    _, s_pre = adapter.encode_sequence(toks_turn1_forced, initial_snapshot=s_pre, step_by_step=False, return_logits=False)
    m_pre_cal, _, _ = evaluate_bop_report_margin(adapter, s_pre, word_x, word_y, tok_x_id, tok_y_id)

    # 2. OBS Condition
    s_obs = s_recip_0.clone()
    out_obs_d, s_obs = adapter.encode_sequence(toks_turn1_user, initial_snapshot=s_obs, step_by_step=False, return_logits=True, logits_to_keep=1)
    d_o = (out_obs_d[0].float()[tok_x_id] - out_obs_d[0].float()[tok_y_id]).item()
    _, s_obs = adapter.encode_sequence(toks_turn1_forced, initial_snapshot=s_obs, step_by_step=False, return_logits=False)
    m_obs_cal, _, _ = evaluate_bop_report_margin(adapter, s_obs, word_x, word_y, tok_x_id, tok_y_id)

    # 3. Contemporaneously Evolved POST Condition
    s_donor_evolved = s_donor_0.clone()
    _, s_donor_evolved = adapter.encode_sequence(toks_turn1_user, initial_snapshot=s_donor_evolved, step_by_step=False, return_logits=False)
    _, s_donor_evolved = adapter.encode_sequence(toks_turn1_forced, initial_snapshot=s_donor_evolved, step_by_step=False, return_logits=False)
    s_post = swap_stores(s_obs.clone(), s_donor_evolved, channels="rglru")
    m_post_cal, _, _ = evaluate_bop_report_margin(adapter, s_post, word_x, word_y, tok_x_id, tok_y_id)

    # 4. Visible R-Control
    vis_ctrl = evaluate_visible_r_control(adapter, word_x, word_y, tok_x_id, tok_y_id)

    # 5. Metrics
    g = 1.0 if d_t >= 0 else -1.0
    true_word = word_x if g > 0 else word_y

    s_pre = g * m_pre_cal
    s_obs = g * m_obs_cal
    s_post = g * m_post_cal

    pai_aligned = g * (m_pre_cal - m_obs_cal)
    t_aligned = g * (m_pre_cal - m_post_cal)

    return {
        "direction": direction,
        "word_x": word_x,
        "word_y": word_y,
        "d_t": d_t,
        "d_o": d_o,
        "delta": d_t - d_o,
        "g": g,
        "true_word": true_word,
        "m_pre": m_pre_cal,
        "m_obs": m_obs_cal,
        "m_post": m_post_cal,
        "s_pre": s_pre,
        "pai_aligned": pai_aligned,
        "t_aligned": t_aligned,
        "visible_control": vis_ctrl,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print("\n" + "=" * 115)
    print("S14.0C FULL EXPERIMENTAL PANEL: LATENT METACOGNITION & CAUSAL PROVENANCE ASSAY")
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

    panel_results = []

    print(f"{'Cell ID':<35} | {'Dir':<3} | {'Truth':<10} | {'Delta':>6} | {'M_PRE':>6} | {'M_OBS':>6} | {'M_POST':>6} | {'S_PRE':>6} | {'PAI_aln':>7} | {'T_aln':>6} | {'R-Ctrl'}")
    print("-" * 115, flush=True)

    for cell in PANEL_CELLS:
        pair_id = cell["pair_id"]
        pair = pair_map[pair_id]
        wx = cell["word_x"]
        wy = cell["word_y"]

        for d in ["fwd", "rev"]:
            res = run_trial(adapter, pair, wx, wy, d, audited_pool)
            r_ok = "PASS" if res["visible_control"]["visible_reporting_competent"] else "FAIL"
            print(
                f"{pair_id:<35} | {d.upper():<3} | {res['true_word']:<10} | "
                f"{res['delta']:+6.2f} | {res['m_pre']:+6.2f} | {res['m_obs']:+6.2f} | {res['m_post']:+6.2f} | "
                f"{res['s_pre']:+6.2f} | {res['pai_aligned']:+7.3f} | {res['t_aligned']:+6.3f} | {r_ok}",
                flush=True,
            )
            panel_results.append({
                "cell": cell,
                "trial": res,
            })

    # Summary Statistics across all trials
    all_pai = [r["trial"]["pai_aligned"] for r in panel_results]
    all_t = [r["trial"]["t_aligned"] for r in panel_results]
    all_s = [r["trial"]["s_pre"] for r in panel_results]
    n_r_pass = sum(1 for r in panel_results if r["trial"]["visible_control"]["visible_reporting_competent"])

    mean_pai = sum(all_pai) / len(all_pai)
    mean_t = sum(all_t) / len(all_t)
    pct_correct_s = sum(1 for s in all_s if s > 0) / len(all_s) * 100.0
    pct_pai_pos = sum(1 for p in all_pai if p > 0) / len(all_pai) * 100.0

    print("-" * 115)
    print("S14.0C PANEL SYNTHESIS SUMMARY:")
    print(f"  Total Trials Evaluated:             {len(panel_results)} (8 cells x 2 directions)")
    print(f"  Visible R-Control Pass Rate:        {n_r_pass}/{len(panel_results)} ({n_r_pass/len(panel_results)*100:.1f}%)")
    print(f"  Semantic Accuracy S_PRE > 0:        {pct_correct_s:.1f}% ({sum(1 for s in all_s if s > 0)}/{len(all_s)})")
    print(f"  Privileged Access PAI_aligned > 0:  {pct_pai_pos:.1f}% ({sum(1 for p in all_pai if p > 0)}/{len(all_pai)}) | Mean PAI_aligned: {mean_pai:+.4f}")
    print(f"  Temporal Contrast Mean T_aligned:   {mean_t:+.4f}")
    print("=" * 115, flush=True)

    # Save artifact
    out_dir = Path("results/e14_latent_metacognition/prior_intention_ownership")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "s14_0c_full_panel_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "pinned_revision": PINNED_IT_REVISION,
            "total_trials": len(panel_results),
            "summary": {
                "visible_r_control_pass_rate": n_r_pass / len(panel_results),
                "semantic_accuracy_pct": pct_correct_s,
                "pai_aligned_positive_pct": pct_pai_pos,
                "mean_pai_aligned": mean_pai,
                "mean_t_aligned": mean_t,
            },
            "trials": panel_results,
        }, f, indent=2)
    print(f"\nFull panel report saved to {out_file}\n", flush=True)


if __name__ == "__main__":
    main()
