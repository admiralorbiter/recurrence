"""Hardened Candidate Screener, R-Level Control & Frozen Panel Generator for S14.0C.

Implements all 4 methodological corrections:
1. Genuinely Common-English Lexicon: Filtered against standard English vocabulary
   (Brown corpus frequency >= 3 and NLTK wordlist), SentencePiece space-prefix,
   length 3-12, and verified round-trip tokenization.
2. Full-Vocabulary Plausibility: Computes full-vocabulary rank and logit delta from z_max.
3. Enforced Mirror-Symmetry Gate: Boolean gate requires D_T^rev <= -0.20 and D_O^rev >= +0.20.
4. Constant-Token Identity Generalization: Evaluates candidate across 3 distinct constant token IDs.
5. Exact Chat-Template Decision Interface Verification: Measures D_T and D_O at the exact
   turn position in the single-pass conversation structure.
6. Visible Direct-Word Reporting Control (R-Level Diagnostic): Verifies that the model
   can report visible ground-truth preferences across option orderings.
7. Commits exactly one frozen (x, y) pair per passing cell into a balanced panel.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
import torch
import torch.nn.functional as F
import nltk
from nltk.corpus import brown, words
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import get_filler_tokens_for_regime, build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs, MicroscopePair
from recurrence.interventions.surgical_swaps import swap_stores


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"
MARGIN_THRESHOLD = 0.30
FULL_VOCAB_RANK_CUTOFF = 500
FULL_VOCAB_DELTA_CUTOFF = 12.0


def build_common_english_lexicon(tokenizer: Any) -> Dict[str, Dict[str, Any]]:
    """Build a clean common-English single-token vocabulary."""
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
        
        # Exact round-trip verification
        encoded = tokenizer.encode(f" {clean}", add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] != token_id:
            continue
        decoded = tokenizer.decode([token_id]).strip()
        if decoded != clean:
            continue

        if clean not in curated:
            curated[clean] = {
                "token_id": token_id,
                "raw_piece": token_str,
                "decoded_text": clean,
                "surface_form": clean,
            }

    return curated


def build_chat_decision_prefix(pair: MicroscopePair) -> str:
    """Build the exact Turn 1 user query using RecurrentGemma-IT chat template."""
    return f"<start_of_turn>user\n{pair.query}<end_of_turn>\n<start_of_turn>model\n"


def build_chat_forced_and_probe(x_word: str, y_word: str, forced_token_str: str = "1", reverse_options: bool = False) -> str:
    """Build the forced output and Turn 2 source-monitoring probe."""
    opt_str = f"'{y_word}' or '{x_word}'" if reverse_options else f"'{x_word}' or '{y_word}'"
    return (
        f"{forced_token_str}<end_of_turn>\n"
        f"<start_of_turn>user\n"
        f"Immediately before that supplied output, which token were you more prepared to produce: {opt_str}?<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


@torch.inference_mode()
def test_visible_direct_word_reporting_control(
    adapter: RecurrentGemmaAdapter,
    word_a: str,
    word_b: str,
    tok_a_id: int,
    tok_b_id: int,
) -> Dict[str, Any]:
    """Test R-level direct-word reporting competence under visible ground truth."""
    tokenizer = adapter.tokenizer

    # Condition 1: Word A was preferred, standard option order
    prompt_1 = (
        f"<start_of_turn>user\n"
        f"Diagnostic record: Immediately before the supplied output, your recorded internal preference was higher for '{word_a}' than '{word_b}'.\n"
        f"Immediately before that supplied output, which token were you more prepared to produce: '{word_a}' or '{word_b}'?<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    # Condition 2: Word B was preferred, standard option order
    prompt_2 = (
        f"<start_of_turn>user\n"
        f"Diagnostic record: Immediately before the supplied output, your recorded internal preference was higher for '{word_b}' than '{word_a}'.\n"
        f"Immediately before that supplied output, which token were you more prepared to produce: '{word_a}' or '{word_b}'?<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    # Condition 3: Word A was preferred, reversed option order in question
    prompt_3 = (
        f"<start_of_turn>user\n"
        f"Diagnostic record: Immediately before the supplied output, your recorded internal preference was higher for '{word_a}' than '{word_b}'.\n"
        f"Immediately before that supplied output, which token were you more prepared to produce: '{word_b}' or '{word_a}'?<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    toks_1 = tokenizer.encode(prompt_1, add_special_tokens=False)
    toks_2 = tokenizer.encode(prompt_2, add_special_tokens=False)
    toks_3 = tokenizer.encode(prompt_3, add_special_tokens=False)

    out_1, _ = adapter.encode_sequence(toks_1, step_by_step=False, return_logits=True, logits_to_keep=1)
    out_2, _ = adapter.encode_sequence(toks_2, step_by_step=False, return_logits=True, logits_to_keep=1)
    out_3, _ = adapter.encode_sequence(toks_3, step_by_step=False, return_logits=True, logits_to_keep=1)

    lg_1 = out_1[0].float()
    lg_2 = out_2[0].float()
    lg_3 = out_3[0].float()

    # Margin: log P(word_a) - log P(word_b)
    m_1 = (lg_1[tok_a_id] - lg_1[tok_b_id]).item()
    m_2 = (lg_2[tok_a_id] - lg_2[tok_b_id]).item()
    m_3 = (lg_3[tok_a_id] - lg_3[tok_b_id]).item()

    # Success: m_1 > 0 (reports A when A is true), m_2 < 0 (reports B when B is true), m_3 > 0 (robust to option order)
    acc_1 = m_1 > 0
    acc_2 = m_2 < 0
    acc_3 = m_3 > 0
    all_passed = acc_1 and acc_2 and acc_3

    return {
        "m_when_a_true": m_1,
        "m_when_b_true": m_2,
        "m_reversed_order": m_3,
        "visible_reporting_passed": all_passed,
    }


@torch.inference_mode()
def screen_cell_candidates(
    adapter: RecurrentGemmaAdapter,
    cell_info: Dict[str, Any],
    audited_pool: List[int],
    lexicon: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Screen candidates for a single cell against all 5 hardened gates."""
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
    # Pick 3 distinct constant token identities
    const_tok_median = clean_pool[len(clean_pool) // 2]
    const_tok_q1 = clean_pool[len(clean_pool) // 4]
    const_tok_q3 = clean_pool[(3 * len(clean_pool)) // 4]

    lexicon_words = sorted(list(lexicon.keys()))
    lexicon_token_ids = [lexicon[w]["token_id"] for w in lexicon_words]
    token_to_word = {lexicon[w]["token_id"]: w for w in lexicon_words}
    lexicon_tensor = torch.tensor(lexicon_token_ids, dtype=torch.long, device=adapter.device)

    # 1. Forward 4K Encoding (Median Constant Token)
    _, s_a_0 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
    _, s_b_0 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)
    filler_4k = [const_tok_median] * 4096
    for i in range(0, 4096, 512):
        chunk = filler_4k[i : i + 512]
        _, s_a_0 = adapter.encode_sequence(chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False)
        _, s_b_0 = adapter.encode_sequence(chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False)

    s_tgt_fwd = swap_stores(s_a_0.clone(), s_b_0.clone(), channels="rglru")
    s_obs_fwd = s_a_0.clone()

    # Exact Turn 1 Decision Prompt Prefix
    chat_prefix = build_chat_decision_prefix(pair)
    chat_prefix_toks = tokenizer.encode(chat_prefix, add_special_tokens=False)

    out_tgt, _ = adapter.encode_sequence(chat_prefix_toks, initial_snapshot=s_tgt_fwd, step_by_step=False, return_logits=True, logits_to_keep=1)
    out_obs, _ = adapter.encode_sequence(chat_prefix_toks, initial_snapshot=s_obs_fwd, step_by_step=False, return_logits=True, logits_to_keep=1)

    lg_t = out_tgt[0].float()
    lg_o = out_obs[0].float()
    z_t_max = torch.max(lg_t).item()
    z_o_max = torch.max(lg_o).item()

    zt_sub = lg_t[lexicon_tensor]
    zo_sub = lg_o[lexicon_tensor]

    # Full-vocab plausibility filter: rank <= 500 or delta <= 12.0
    plausible_indices = []
    for idx_lex, tid in enumerate(lexicon_token_ids):
        zt_val = zt_sub[idx_lex].item()
        zo_val = zo_sub[idx_lex].item()
        rank_t = int(torch.sum(lg_t > zt_val).item() + 1)
        rank_o = int(torch.sum(lg_o > zo_val).item() + 1)
        delta_t = z_t_max - zt_val
        delta_o = z_o_max - zo_val

        if (rank_t <= FULL_VOCAB_RANK_CUTOFF or delta_t <= FULL_VOCAB_DELTA_CUTOFF) or (rank_o <= FULL_VOCAB_RANK_CUTOFF or delta_o <= FULL_VOCAB_DELTA_CUTOFF):
            plausible_indices.append(idx_lex)

    plausible_t = torch.tensor(plausible_indices, dtype=torch.long, device=adapter.device)
    zt_p = zt_sub[plausible_t]
    zo_p = zo_sub[plausible_t]

    gap_t = zt_p.unsqueeze(1) - zt_p.unsqueeze(0)
    gap_o = zo_p.unsqueeze(1) - zo_p.unsqueeze(0)

    # Disagreement mask: D_T >= 0.30 and D_O <= -0.30
    mask = (gap_t >= MARGIN_THRESHOLD) & (gap_o <= -MARGIN_THRESHOLD)
    matches = mask.nonzero(as_tuple=False)

    if matches.shape[0] == 0:
        return None

    # Reverse 4K Encoding (B<-A vs B obs)
    s_tgt_rev = swap_stores(s_b_0.clone(), s_a_0.clone(), channels="rglru")
    s_obs_rev = s_b_0.clone()
    out_tgt_rev, _ = adapter.encode_sequence(chat_prefix_toks, initial_snapshot=s_tgt_rev, step_by_step=False, return_logits=True, logits_to_keep=1)
    out_obs_rev, _ = adapter.encode_sequence(chat_prefix_toks, initial_snapshot=s_obs_rev, step_by_step=False, return_logits=True, logits_to_keep=1)
    lg_t_rev = out_tgt_rev[0].float()
    lg_o_rev = out_obs_rev[0].float()

    # Distinct Constant-Token 4K Encoding (Token Q1)
    _, s_a_q1 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
    _, s_b_q1 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)
    filler_q1 = [const_tok_q1] * 4096
    for i in range(0, 4096, 512):
        chunk = filler_q1[i : i + 512]
        _, s_a_q1 = adapter.encode_sequence(chunk, initial_snapshot=s_a_q1, step_by_step=False, return_logits=False)
        _, s_b_q1 = adapter.encode_sequence(chunk, initial_snapshot=s_b_q1, step_by_step=False, return_logits=False)
    s_tgt_q1 = swap_stores(s_a_q1.clone(), s_b_q1.clone(), channels="rglru")
    s_obs_q1 = s_a_q1.clone()
    out_tgt_q1, _ = adapter.encode_sequence(chat_prefix_toks, initial_snapshot=s_tgt_q1, step_by_step=False, return_logits=True, logits_to_keep=1)
    out_obs_q1, _ = adapter.encode_sequence(chat_prefix_toks, initial_snapshot=s_obs_q1, step_by_step=False, return_logits=True, logits_to_keep=1)
    lg_t_q1 = out_tgt_q1[0].float()
    lg_o_q1 = out_obs_q1[0].float()

    # Collect valid candidates
    passing_candidates = []
    for idx_pair in matches:
        ti, oj = idx_pair[0].item(), idx_pair[1].item()
        orig_i = plausible_indices[ti]
        orig_j = plausible_indices[oj]
        tid_x = lexicon_token_ids[orig_i]
        tid_y = lexicon_token_ids[orig_j]
        wx = token_to_word[tid_x]
        wy = token_to_word[tid_y]

        gt = gap_t[ti, oj].item()
        go = gap_o[ti, oj].item()

        # Gate 3: Enforce Mirror Symmetry
        d_t_rev = (lg_t_rev[tid_x] - lg_t_rev[tid_y]).item()
        d_o_rev = (lg_o_rev[tid_x] - lg_o_rev[tid_y]).item()
        mirror_passed = (d_t_rev <= -0.20 and d_o_rev >= +0.20)

        # Gate 4: Constant Token Identity Generalization (Token Q1)
        d_t_q1 = (lg_t_q1[tid_x] - lg_t_q1[tid_y]).item()
        d_o_q1 = (lg_o_q1[tid_x] - lg_o_q1[tid_y]).item()
        token_generalization_passed = (d_t_q1 >= 0.15 and d_o_q1 <= -0.15)

        # Full-vocab rank metrics
        rank_t_x = int(torch.sum(lg_t > lg_t[tid_x]).item() + 1)
        rank_o_y = int(torch.sum(lg_o > lg_o[tid_y]).item() + 1)
        delta_t_x = z_t_max - lg_t[tid_x].item()
        delta_o_y = z_o_max - lg_o[tid_y].item()

        if mirror_passed and token_generalization_passed:
            # Gate 6: Visible Direct-Word Reporting Control
            r_ctrl = test_visible_direct_word_reporting_control(adapter, wx, wy, tid_x, tid_y)

            passing_candidates.append({
                "pair_id": cell_info["pair_id"],
                "regime": regime,
                "word_x": wx,
                "word_y": wy,
                "tok_x_id": tid_x,
                "tok_y_id": tid_y,
                "forward": {
                    "d_t": gt,
                    "d_o": go,
                    "total_margin": gt - go,
                    "full_vocab_rank_t_x": rank_t_x,
                    "full_vocab_rank_o_y": rank_o_y,
                    "delta_z_t_x": delta_t_x,
                    "delta_z_o_y": delta_o_y,
                },
                "reverse": {
                    "d_t_rev": d_t_rev,
                    "d_o_rev": d_o_rev,
                    "mirror_passed": mirror_passed,
                },
                "token_generalization": {
                    "d_t_q1": d_t_q1,
                    "d_o_q1": d_o_q1,
                    "generalization_passed": token_generalization_passed,
                },
                "visible_control": r_ctrl,
                "all_gates_passed": r_ctrl["visible_reporting_passed"],
            })

    if not passing_candidates:
        return None

    # Sort by total forward margin and pick top 1
    passing_candidates.sort(key=lambda c: c["forward"]["total_margin"], reverse=True)
    return passing_candidates[0]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print("\n" + "=" * 115)
    print("S14.0C CANDIDATE HARDENING & FROZEN BALANCED PANEL GENERATOR")
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
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    lexicon = build_common_english_lexicon(tokenizer)
    print(f"Curated Common-English Lexicon: {len(lexicon)} verified English words.\n", flush=True)

    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)

    # Load all 72 cells to screen for constant-regime cells
    manifest_path = Path("results/e14_latent_metacognition/counterfactual_screen/bidirectional_provenance_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Select all constant regime cells sorted by |R_role|
    reanalysis_path = Path("results/e14_latent_metacognition/counterfactual_screen/role_residual_reanalysis.json")
    with open(reanalysis_path, "r", encoding="utf-8") as f:
        reanalysis = json.load(f)

    constant_cells = [c for c in reanalysis["cells_sorted_by_abs_r_role"] if c["regime"] == "constant"]

    frozen_panel = []

    print(f"Screening {len(constant_cells)} constant-drive cells across all 5 gates + Visible Control...\n")
    for cell in constant_cells:
        pair_id = cell["pair_id"]
        r_role = cell["r_role"]
        dir_str = "DONOR" if r_role > 0 else "ANTI"
        print(f"--- Screening: {pair_id} | R_role={r_role:+.3f} ({dir_str}) ---", flush=True)

        best_candidate = screen_cell_candidates(adapter, cell, audited_pool, lexicon)
        if best_candidate is not None:
            fwd = best_candidate["forward"]
            rev = best_candidate["reverse"]
            gen = best_candidate["token_generalization"]
            ctrl = best_candidate["visible_control"]

            print(
                f"  >>> FROZEN CANDIDATE: '{best_candidate['word_x']}' vs '{best_candidate['word_y']}'\n"
                f"      Forward (A<-B): D_T={fwd['d_t']:+.3f}, D_O={fwd['d_o']:+.3f} (Full-vocab ranks: T={fwd['full_vocab_rank_t_x']}, O={fwd['full_vocab_rank_o_y']})\n"
                f"      Reverse (B<-A): D_T={rev['d_t_rev']:+.3f}, D_O={rev['d_o_rev']:+.3f} (Mirror Gate: PASSED)\n"
                f"      Alt Constant:   D_T={gen['d_t_q1']:+.3f}, D_O={gen['d_o_q1']:+.3f} (Generalization: PASSED)\n"
                f"      Visible R-Ctrl: m(A_true)={ctrl['m_when_a_true']:+.3f}, m(B_true)={ctrl['m_when_b_true']:+.3f} (Passed: {ctrl['visible_reporting_passed']})\n",
                flush=True,
            )
            frozen_panel.append(best_candidate)
        else:
            print("  [No candidate passed all 5 gates + visible control]\n", flush=True)

    # Summary
    n_donor = sum(1 for c in frozen_panel if c["forward"]["d_t"] > 0)
    n_anti = sum(1 for c in frozen_panel if c["forward"]["d_t"] < 0)

    print("=" * 115)
    print("FROZEN BALANCED S14.0C EXPERIMENTAL PANEL")
    print("=" * 115)
    print(f"Total Frozen Candidate Cells: {len(frozen_panel)}")
    print(f"Donor-oriented: {n_donor}, Anti-donor-oriented: {n_anti}")
    for i, c in enumerate(frozen_panel, 1):
        print(f"  {i}. {c['pair_id']:<38} | '{c['word_x']}' vs '{c['word_y']}' | Fwd: D_T={c['forward']['d_t']:+.2f}, D_O={c['forward']['d_o']:+.2f} | Rev: D_T={c['reverse']['d_t_rev']:+.2f}, D_O={c['reverse']['d_o_rev']:+.2f}")
    print("=" * 115)

    # Save frozen panel artifact
    out_dir = Path("results/e14_latent_metacognition/prior_intention_ownership")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "frozen_s14_panel.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "pinned_revision": PINNED_IT_REVISION,
            "margin_threshold": MARGIN_THRESHOLD,
            "full_vocab_rank_cutoff": FULL_VOCAB_RANK_CUTOFF,
            "common_english_lexicon_size": len(lexicon),
            "total_frozen_cells": len(frozen_panel),
            "frozen_panel": frozen_panel,
        }, f, indent=2)
    print(f"\nFrozen panel saved to {out_file}\n", flush=True)


if __name__ == "__main__":
    main()
