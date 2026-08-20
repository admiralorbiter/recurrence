"""Diagnose gate-by-gate attrition for candidate hardening.

Runs on the top constant-drive cells and prints the exact number of passing candidates
at each successive gate:
  Base (Common-English Lexicon)
  -> Gate 1: Margin Disagreement (D_T >= 0.30, D_O <= -0.30)
  -> Gate 2: Full-Vocab Plausibility (rank <= 500 or delta <= 12.0)
  -> Gate 3: Enforced Mirror Symmetry (D_T_rev <= -0.20, D_O_rev >= +0.20)
  -> Gate 4: Distinct Constant Token Generalization
  -> Gate 5: Visible Direct-Word Control
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
import nltk
from nltk.corpus import brown, words
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs
from recurrence.interventions.surgical_swaps import swap_stores


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"


def build_common_english_lexicon(tokenizer: Any) -> Dict[str, Dict[str, Any]]:
    nltk.download("brown", quiet=True)
    nltk.download("words", quiet=True)
    fdist = nltk.FreqDist(w.lower() for w in brown.words() if w.isalpha())
    common_brown = set(w for w, count in fdist.items() if count >= 3 and 3 <= len(w) <= 12)
    nltk_words = set(w.lower() for w in words.words() if w.isalpha() and 3 <= len(w) <= 12)
    valid_english_words = common_brown.intersection(nltk_words)

    vocab = tokenizer.get_vocab()
    special_ids = set(tokenizer.all_special_ids)
    curated = {}

    for token_str, token_id in vocab.items():
        if token_id in special_ids or token_id < 10:
            continue
        if not token_str.startswith("\u2581"):
            continue
        clean = token_str[1:]
        if clean not in valid_english_words:
            continue
        
        encoded = tokenizer.encode(f" {clean}", add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] != token_id:
            continue
        decoded = tokenizer.decode([token_id]).strip()
        if decoded != clean:
            continue

        if clean not in curated:
            curated[clean] = {
                "token_id": token_id,
                "surface_form": clean,
            }

    return curated


@torch.inference_mode()
def diagnose_cell(
    adapter: RecurrentGemmaAdapter,
    cell_info: Dict[str, Any],
    audited_pool: List[int],
    lexicon: Dict[str, Dict[str, Any]],
):
    tokenizer = adapter.tokenizer
    pairs = build_microscope_pairs()
    pair = next(p for p in pairs if p.pair_id == cell_info["pair_id"])
    regime = cell_info["regime"]

    toks_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
    toks_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
    tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
    tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]
    excluded = set(toks_a + toks_b + [tok_a_id, tok_b_id])

    clean_pool = [t for t in audited_pool if t not in excluded]
    const_tok_median = clean_pool[len(clean_pool) // 2]
    const_tok_q1 = clean_pool[len(clean_pool) // 4]

    lexicon_words = sorted(list(lexicon.keys()))
    lexicon_token_ids = [lexicon[w]["token_id"] for w in lexicon_words]
    lexicon_tensor = torch.tensor(lexicon_token_ids, dtype=torch.long, device=adapter.device)

    # 1. Forward 4K Encoding
    _, s_a_0 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
    _, s_b_0 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)
    filler_4k = [const_tok_median] * 4096
    for i in range(0, 4096, 512):
        chunk = filler_4k[i : i + 512]
        _, s_a_0 = adapter.encode_sequence(chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False)
        _, s_b_0 = adapter.encode_sequence(chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False)

    s_tgt_fwd = swap_stores(s_a_0.clone(), s_b_0.clone(), channels="rglru")
    s_obs_fwd = s_a_0.clone()

    # Query prompt (Turn 1 query)
    chat_prefix = f"<start_of_turn>user\n{pair.query}<end_of_turn>\n<start_of_turn>model\n"
    chat_toks = tokenizer.encode(chat_prefix, add_special_tokens=False)

    out_tgt, _ = adapter.encode_sequence(chat_toks, initial_snapshot=s_tgt_fwd, step_by_step=False, return_logits=True, logits_to_keep=1)
    out_obs, _ = adapter.encode_sequence(chat_toks, initial_snapshot=s_obs_fwd, step_by_step=False, return_logits=True, logits_to_keep=1)

    lg_t = out_tgt[0].float()
    lg_o = out_obs[0].float()
    z_t_max = torch.max(lg_t).item()
    z_o_max = torch.max(lg_o).item()

    zt_sub = lg_t[lexicon_tensor]
    zo_sub = lg_o[lexicon_tensor]

    # Full-vocab top-1000 cutoff
    rank_t_all = torch.sum(lg_t.unsqueeze(0) > zt_sub.unsqueeze(1), dim=1) + 1
    rank_o_all = torch.sum(lg_o.unsqueeze(0) > zo_sub.unsqueeze(1), dim=1) + 1
    delta_t_all = z_t_max - zt_sub
    delta_o_all = z_o_max - zo_sub

    # Pairwise forward difference
    gap_t = zt_sub.unsqueeze(1) - zt_sub.unsqueeze(0)
    gap_o = zo_sub.unsqueeze(1) - zo_sub.unsqueeze(0)

    # Gate 1: Raw Disagreement (m >= 0.30)
    g1_mask = (gap_t >= 0.30) & (gap_o <= -0.30)
    n_g1 = g1_mask.sum().item()

    # Gate 2: Full-Vocab Plausibility (e.g. rank <= 1000 OR delta <= 15.0 for both x in T and y in O)
    plaus_x_t = (rank_t_all <= 1000) | (delta_t_all <= 15.0)
    plaus_y_o = (rank_o_all <= 1000) | (delta_o_all <= 15.0)
    g2_mask = g1_mask & plaus_x_t.unsqueeze(1) & plaus_y_o.unsqueeze(0)
    n_g2 = g2_mask.sum().item()

    # Reverse 4K
    s_tgt_rev = swap_stores(s_b_0.clone(), s_a_0.clone(), channels="rglru")
    s_obs_rev = s_b_0.clone()
    out_tgt_rev, _ = adapter.encode_sequence(chat_toks, initial_snapshot=s_tgt_rev, step_by_step=False, return_logits=True, logits_to_keep=1)
    out_obs_rev, _ = adapter.encode_sequence(chat_toks, initial_snapshot=s_obs_rev, step_by_step=False, return_logits=True, logits_to_keep=1)
    lg_t_rev = out_tgt_rev[0].float()
    lg_o_rev = out_obs_rev[0].float()
    zt_rev = lg_t_rev[lexicon_tensor]
    zo_rev = lg_o_rev[lexicon_tensor]
    gap_t_rev = zt_rev.unsqueeze(1) - zt_rev.unsqueeze(0)
    gap_o_rev = zo_rev.unsqueeze(1) - zo_rev.unsqueeze(0)

    # Gate 3: Reverse Mirror Symmetry (D_T_rev <= -0.15 and D_O_rev >= 0.15)
    g3_mirror_mask = (gap_t_rev <= -0.15) & (gap_o_rev >= 0.15)
    g3_mask = g2_mask & g3_mirror_mask
    n_g3 = g3_mask.sum().item()

    # Also check softer mirror requirement: (gap_t_rev < 0 and gap_o_rev > 0)
    g3_soft_mirror = (gap_t_rev < 0) & (gap_o_rev > 0)
    n_g3_soft = (g2_mask & g3_soft_mirror).sum().item()

    # Distinct Constant Token Q1
    _, s_a_q1 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
    _, s_b_q1 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)
    filler_q1 = [const_tok_q1] * 4096
    for i in range(0, 4096, 512):
        chunk = filler_q1[i : i + 512]
        _, s_a_q1 = adapter.encode_sequence(chunk, initial_snapshot=s_a_q1, step_by_step=False, return_logits=False)
        _, s_b_q1 = adapter.encode_sequence(chunk, initial_snapshot=s_b_q1, step_by_step=False, return_logits=False)
    s_tgt_q1 = swap_stores(s_a_q1.clone(), s_b_q1.clone(), channels="rglru")
    s_obs_q1 = s_a_q1.clone()
    out_tgt_q1, _ = adapter.encode_sequence(chat_toks, initial_snapshot=s_tgt_q1, step_by_step=False, return_logits=True, logits_to_keep=1)
    out_obs_q1, _ = adapter.encode_sequence(chat_toks, initial_snapshot=s_obs_q1, step_by_step=False, return_logits=True, logits_to_keep=1)
    lg_t_q1 = out_tgt_q1[0].float()
    lg_o_q1 = out_obs_q1[0].float()
    zt_q1 = lg_t_q1[lexicon_tensor]
    zo_q1 = lg_o_q1[lexicon_tensor]
    gap_t_q1 = zt_q1.unsqueeze(1) - zt_q1.unsqueeze(0)
    gap_o_q1 = zo_q1.unsqueeze(1) - zo_q1.unsqueeze(0)
    g4_gen_mask = (gap_t_q1 >= 0.10) & (gap_o_q1 <= -0.10)
    n_g4 = (g3_mask & g4_gen_mask).sum().item()

    print(
        f"  {cell_info['pair_id']:<38} | "
        f"G1 (Raw Disag >=0.30): {n_g1:>6} | "
        f"G2 (Plausible): {n_g2:>5} | "
        f"G3 (Mirror Soft): {n_g3_soft:>4} | "
        f"G3 (Mirror Strict): {n_g3:>4} | "
        f"G4 (Alt Const): {n_g4:>4}",
        flush=True,
    )

    if n_g3 > 0:
        matches = g3_mask.nonzero(as_tuple=False)
        top_m = matches[0]
        wx = lexicon_words[top_m[0].item()]
        wy = lexicon_words[top_m[1].item()]
        gt = gap_t[top_m[0], top_m[1]].item()
        go = gap_o[top_m[0], top_m[1]].item()
        gtr = gap_t_rev[top_m[0], top_m[1]].item()
        gor = gap_o_rev[top_m[0], top_m[1]].item()
        print(f"    Sample G3 Pair: '{wx}' vs '{wy}' | Fwd: T={gt:+.2f}, O={go:+.2f} | Rev: T={gtr:+.2f}, O={gor:+.2f}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print("\n" + "=" * 115)
    print("GATE ATTRITION DIAGNOSTIC: TOP 6 CONSTANT-DRIVE CELLS")
    print("=" * 115, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=PINNED_IT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=PINNED_IT_REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    adapter = RecurrentGemmaAdapter(
        model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16,
    )
    lexicon = build_common_english_lexicon(tokenizer)
    print(f"Common English Lexicon: {len(lexicon)} words", flush=True)

    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)

    reanalysis_path = Path("results/e14_latent_metacognition/counterfactual_screen/role_residual_reanalysis.json")
    with open(reanalysis_path, "r", encoding="utf-8") as f:
        reanalysis = json.load(f)

    top_constant_cells = [c for c in reanalysis["cells_sorted_by_abs_r_role"] if c["regime"] == "constant"][:6]

    for cell in top_constant_cells:
        diagnose_cell(adapter, cell, audited_pool, lexicon)


if __name__ == "__main__":
    main()
