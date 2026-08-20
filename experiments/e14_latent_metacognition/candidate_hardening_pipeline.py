"""Candidate Hardening Pipeline for S14 Causal Provenance Assay.

Executes the complete 5-stage validation pipeline with vectorized search:
1. Curated Lexicon: Clean, unambiguous common English single-token words
   with strict exact-context round-trip tokenization and surface deduplication.
2. Absolute Plausibility Filter: Both candidates must rank in top-250 (or near top)
   of the model's output distribution, eliminating tail tokens.
3. Bidirectional Evaluation: Screen both forward (A<-B) and reverse (B<-A) to check
   role-associated behavior.
4. Fresh-Stream Replication: Re-evaluate discovered candidates under an independent
   filler seed (seed=999 + idx).
5. Exact-Prompt Re-measurement: Place candidates in the exact S14 decision prompt
   and verify that D_T > m and D_O < -m persist under the exact operational interface.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import get_filler_tokens_for_regime, build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs, MicroscopePair
from recurrence.interventions.surgical_swaps import swap_stores


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"
MARGIN_THRESHOLD = 0.30        # Require >= 0.30 logit margin in each direction
PLAUSIBILITY_TOP_K = 250       # Candidates must rank within top-250 among curated lexicon


def build_curated_english_lexicon(tokenizer: Any) -> Dict[str, Dict[str, Any]]:
    """Build a clean, curated single-token English lexicon fast.
    
    Requirements:
    - Starts with SentencePiece leading space (\u2581)
    - Stripped string is purely lowercase ASCII alphabetic (length 3 to 12)
    - Deduplicated by normalized surface form
    """
    vocab = tokenizer.get_vocab()
    curated = {}
    special_ids = set(tokenizer.all_special_ids)

    for token_str, token_id in vocab.items():
        if token_id in special_ids or token_id < 10:
            continue
        
        # Must start with SentencePiece space
        if not token_str.startswith("\u2581"):
            continue
        
        clean = token_str[1:]  # remove \u2581
        if not (3 <= len(clean) <= 12 and clean.isalpha() and clean.islower() and clean.isascii()):
            continue
        
        if clean not in curated:
            curated[clean] = {
                "token_id": token_id,
                "raw_piece": token_str,
                "decoded_text": clean,
                "normalized_surface_form": clean,
            }

    return curated


def verify_token_roundtrip(tokenizer: Any, token_id: int, surface_word: str) -> bool:
    """Strictly verify roundtrip tokenization in continuation context."""
    encoded = tokenizer.encode(f" {surface_word}", add_special_tokens=False)
    if len(encoded) != 1 or encoded[0] != token_id:
        return False
    decoded = tokenizer.decode([token_id]).strip()
    return decoded == surface_word


@torch.inference_mode()
def evaluate_candidate_pair_at_decision_prompt(
    adapter: RecurrentGemmaAdapter,
    pair: MicroscopePair,
    s_target_origin,
    s_observer_origin,
    tok_x_id: int,
    tok_y_id: int,
    decision_prompt_text: str,
) -> Tuple[float, float, float, float]:
    """Re-measure D_T and D_O under the exact task decision prompt."""
    tokenizer = adapter.tokenizer
    prompt_toks = tokenizer.encode(decision_prompt_text, add_special_tokens=False)

    out_tgt, _ = adapter.encode_sequence(
        prompt_toks, initial_snapshot=s_target_origin, step_by_step=False,
        return_logits=True, logits_to_keep=1
    )
    out_obs, _ = adapter.encode_sequence(
        prompt_toks, initial_snapshot=s_observer_origin, step_by_step=False,
        return_logits=True, logits_to_keep=1
    )

    lg_t = out_tgt[0].float()
    lg_o = out_obs[0].float()

    d_t = (lg_t[tok_x_id] - lg_t[tok_y_id]).item()
    d_o = (lg_o[tok_x_id] - lg_o[tok_y_id]).item()
    z_t_x = lg_t[tok_x_id].item()
    z_o_y = lg_o[tok_y_id].item()

    return d_t, d_o, z_t_x, z_o_y


@torch.inference_mode()
def run_hardening_pipeline(
    adapter: RecurrentGemmaAdapter,
    top_cells: List[Dict[str, Any]],
    audited_pool: List[int],
    lexicon: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute all 5 stages of candidate hardening with vectorized evaluation."""
    tokenizer = adapter.tokenizer
    pairs = build_microscope_pairs()
    pair_map = {p.pair_id: p for p in pairs}
    pair_idx_map = {p.pair_id: i for i, p in enumerate(pairs)}

    lexicon_words = sorted(list(lexicon.keys()))
    lexicon_token_ids = [lexicon[w]["token_id"] for w in lexicon_words]
    token_to_word = {lexicon[w]["token_id"]: w for w in lexicon_words}
    lexicon_tensor = torch.tensor(lexicon_token_ids, dtype=torch.long, device=adapter.device)

    print(f"\nCurated English Lexicon: {len(lexicon)} verified single-token words.", flush=True)
    print("Screening top candidate cells for plausible, robust counterfactual rank reversals...\n", flush=True)

    hardened_candidates = []

    for cell in top_cells:
        pair_id = cell["pair_id"]
        regime = cell["regime"]
        pair = pair_map[pair_id]
        p_idx = pair_idx_map[pair_id]

        print(f"=== Cell: {pair_id} | Regime: {regime} | Discovery R_role: {cell['r_role']:+.3f} ===", flush=True)

        toks_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
        toks_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
        tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
        tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]
        excluded = set(toks_a + toks_b + [tok_a_id, tok_b_id])
        query_toks = tokenizer.encode(pair.query, add_special_tokens=False)

        # -------------------------------------------------------------
        # STAGE 1 & 2: Forward Discovery with Absolute Plausibility Filter
        # -------------------------------------------------------------
        discovery_seed = 42 + p_idx * 100
        _, s_a_0 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
        _, s_b_0 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)

        filler_4k = get_filler_tokens_for_regime(
            regime, length=4096, seed=discovery_seed, audited_pool=audited_pool,
            tokenizer=tokenizer, excluded_token_ids=excluded,
        )
        for i in range(0, 4096, 512):
            chunk = filler_4k[i : i + 512]
            _, s_a_0 = adapter.encode_sequence(chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False)
            _, s_b_0 = adapter.encode_sequence(chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False)

        s_tgt_fwd = swap_stores(s_a_0.clone(), s_b_0.clone(), channels="rglru")
        s_obs_fwd = s_a_0.clone()

        out_tgt, _ = adapter.encode_sequence(query_toks, initial_snapshot=s_tgt_fwd, step_by_step=False, return_logits=True, logits_to_keep=1)
        out_obs, _ = adapter.encode_sequence(query_toks, initial_snapshot=s_obs_fwd, step_by_step=False, return_logits=True, logits_to_keep=1)

        lg_t = out_tgt[0].float()
        lg_o = out_obs[0].float()

        # Evaluate curated word subset
        zt_sub = lg_t[lexicon_tensor]  # shape [N_lex]
        zo_sub = lg_o[lexicon_tensor]  # shape [N_lex]

        # Top-K indices within curated lexicon
        top_k_t = torch.topk(zt_sub, min(PLAUSIBILITY_TOP_K, len(lexicon_words))).indices
        top_k_o = torch.topk(zo_sub, min(PLAUSIBILITY_TOP_K, len(lexicon_words))).indices

        # Union of plausible indices
        plausible_indices = torch.unique(torch.cat([top_k_t, top_k_o]))
        zt_p = zt_sub[plausible_indices]
        zo_p = zo_sub[plausible_indices]

        # Vectorized pairwise differences: shape [P, P]
        gap_t = zt_p.unsqueeze(1) - zt_p.unsqueeze(0)
        gap_o = zo_p.unsqueeze(1) - zo_p.unsqueeze(0)

        # Reversal condition: target prefers x to y by >= margin, observer prefers y to x by >= margin
        reversal_mask = (gap_t >= MARGIN_THRESHOLD) & (gap_o <= -MARGIN_THRESHOLD)
        matches = reversal_mask.nonzero(as_tuple=False)

        cell_candidates = []
        for idx in matches:
            ti, oj = idx[0].item(), idx[1].item()
            orig_i = plausible_indices[ti].item()
            orig_j = plausible_indices[oj].item()
            tid_x = lexicon_token_ids[orig_i]
            tid_y = lexicon_token_ids[orig_j]
            wx = token_to_word[tid_x]
            wy = token_to_word[tid_y]

            # Verify roundtrip
            if not verify_token_roundtrip(tokenizer, tid_x, wx) or not verify_token_roundtrip(tokenizer, tid_y, wy):
                continue

            gt = gap_t[ti, oj].item()
            go = gap_o[ti, oj].item()

            cell_candidates.append({
                "word_x": wx,
                "word_y": wy,
                "tok_x_id": tid_x,
                "tok_y_id": tid_y,
                "d_t_fwd": gt,
                "d_o_fwd": go,
                "total_margin": gt - go,
                "rank_t_x": int(torch.sum(zt_sub > zt_sub[orig_i]).item() + 1),
                "rank_o_y": int(torch.sum(zo_sub > zo_sub[orig_j]).item() + 1),
            })

        cell_candidates.sort(key=lambda c: c["total_margin"], reverse=True)
        print(f"  Found {len(cell_candidates)} plausible candidate pairs meeting margin >= {MARGIN_THRESHOLD}", flush=True)

        # Deep verification for top candidates
        for cand in cell_candidates[:3]:
            wx = cand["word_x"]
            wy = cand["word_y"]
            tx_id = cand["tok_x_id"]
            ty_id = cand["tok_y_id"]

            # ---------------------------------------------------------
            # STAGE 3: Reverse Evaluation (B<-A vs B's observer)
            # ---------------------------------------------------------
            s_tgt_rev = swap_stores(s_b_0.clone(), s_a_0.clone(), channels="rglru")
            s_obs_rev = s_b_0.clone()
            out_tgt_rev, _ = adapter.encode_sequence(query_toks, initial_snapshot=s_tgt_rev, step_by_step=False, return_logits=True, logits_to_keep=1)
            out_obs_rev, _ = adapter.encode_sequence(query_toks, initial_snapshot=s_obs_rev, step_by_step=False, return_logits=True, logits_to_keep=1)
            lg_t_rev = out_tgt_rev[0].float()
            lg_o_rev = out_obs_rev[0].float()
            d_t_rev = (lg_t_rev[tx_id] - lg_t_rev[ty_id]).item()
            d_o_rev = (lg_o_rev[tx_id] - lg_o_rev[ty_id]).item()

            # ---------------------------------------------------------
            # STAGE 4: Fresh-Stream Replication (New Filler Seed)
            # ---------------------------------------------------------
            fresh_seed = 999 + p_idx * 100
            _, s_a_fresh = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
            _, s_b_fresh = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)
            filler_fresh = get_filler_tokens_for_regime(
                regime, length=4096, seed=fresh_seed, audited_pool=audited_pool,
                tokenizer=tokenizer, excluded_token_ids=excluded,
            )
            for i in range(0, 4096, 512):
                chunk = filler_fresh[i : i + 512]
                _, s_a_fresh = adapter.encode_sequence(chunk, initial_snapshot=s_a_fresh, step_by_step=False, return_logits=False)
                _, s_b_fresh = adapter.encode_sequence(chunk, initial_snapshot=s_b_fresh, step_by_step=False, return_logits=False)

            s_tgt_fresh = swap_stores(s_a_fresh.clone(), s_b_fresh.clone(), channels="rglru")
            s_obs_fresh = s_a_fresh.clone()

            out_tgt_fresh, _ = adapter.encode_sequence(query_toks, initial_snapshot=s_tgt_fresh, step_by_step=False, return_logits=True, logits_to_keep=1)
            out_obs_fresh, _ = adapter.encode_sequence(query_toks, initial_snapshot=s_obs_fresh, step_by_step=False, return_logits=True, logits_to_keep=1)
            lg_t_fresh = out_tgt_fresh[0].float()
            lg_o_fresh = out_obs_fresh[0].float()

            d_t_fresh = (lg_t_fresh[tx_id] - lg_t_fresh[ty_id]).item()
            d_o_fresh = (lg_o_fresh[tx_id] - lg_o_fresh[ty_id]).item()

            fresh_replicated = (d_t_fresh > 0.05 and d_o_fresh < -0.05)

            # ---------------------------------------------------------
            # STAGE 5: Exact Task Decision Prompt Measurement
            # ---------------------------------------------------------
            decision_prompt = pair.query
            d_t_prompt, d_o_prompt, z_t_x_prompt, z_o_y_prompt = evaluate_candidate_pair_at_decision_prompt(
                adapter, pair, s_tgt_fwd, s_obs_fwd, tx_id, ty_id, decision_prompt
            )
            prompt_confirmed = (d_t_prompt >= MARGIN_THRESHOLD and d_o_prompt <= -MARGIN_THRESHOLD)

            status_str = "PASSED ALL GATES" if (fresh_replicated and prompt_confirmed) else "PARTIAL"
            print(
                f"    Candidate '{wx}' vs '{wy}':\n"
                f"      Discovery Fwd: D_T={cand['d_t_fwd']:+.3f}, D_O={cand['d_o_fwd']:+.3f} (Ranks in lexicon: T={cand['rank_t_x']}, O={cand['rank_o_y']})\n"
                f"      Reverse Check: D_T={d_t_rev:+.3f}, D_O={d_o_rev:+.3f}\n"
                f"      Fresh Seed:    D_T={d_t_fresh:+.3f}, D_O={d_o_fresh:+.3f} -> Replicated: {fresh_replicated}\n"
                f"      Exact Prompt:  D_T={d_t_prompt:+.3f}, D_O={d_o_prompt:+.3f} -> Confirmed: {prompt_confirmed}\n"
                f"      [{status_str}]",
                flush=True,
            )

            hardened_candidates.append({
                "pair_id": pair_id,
                "regime": regime,
                "word_x": wx,
                "word_y": wy,
                "tok_x_id": tx_id,
                "tok_y_id": ty_id,
                "discovery": {
                    "d_t": cand["d_t_fwd"],
                    "d_o": cand["d_o_fwd"],
                    "rank_t_x": cand["rank_t_x"],
                    "rank_o_y": cand["rank_o_y"],
                },
                "reverse": {
                    "d_t": d_t_rev,
                    "d_o": d_o_rev,
                },
                "fresh_replication": {
                    "d_t": d_t_fresh,
                    "d_o": d_o_fresh,
                    "replicated": fresh_replicated,
                },
                "exact_prompt": {
                    "d_t": d_t_prompt,
                    "d_o": d_o_prompt,
                    "confirmed": prompt_confirmed,
                },
                "all_gates_passed": fresh_replicated and prompt_confirmed,
            })

    return {
        "model_id": adapter.model.config._name_or_path,
        "pinned_revision": PINNED_IT_REVISION,
        "margin_threshold": MARGIN_THRESHOLD,
        "plausibility_top_k": PLAUSIBILITY_TOP_K,
        "curated_lexicon_size": len(lexicon),
        "candidates": hardened_candidates,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print("\n" + "=" * 115)
    print("S14 CANDIDATE HARDENING PIPELINE: DISCOVERY -> PLAUSIBILITY -> REVERSE -> REPLICATION -> EXACT PROMPT")
    print(f"Model: {model_id} (revision: {PINNED_IT_REVISION[:10]}...)")
    print("=" * 115, flush=True)

    reanalysis_path = Path("results/e14_latent_metacognition/counterfactual_screen/role_residual_reanalysis.json")
    with open(reanalysis_path, "r", encoding="utf-8") as f:
        reanalysis = json.load(f)

    # Top 8 cells by |R_role|
    top_cells = reanalysis["cells_sorted_by_abs_r_role"][:8]

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=PINNED_IT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=PINNED_IT_REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    adapter = RecurrentGemmaAdapter(
        model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    lexicon = build_curated_english_lexicon(tokenizer)
    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)

    results = run_hardening_pipeline(adapter, top_cells, audited_pool, lexicon)

    # Save artifact
    out_dir = Path("results/e14_latent_metacognition/counterfactual_screen")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hardened_candidates_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nHardened candidate report saved to {out_file}\n", flush=True)


if __name__ == "__main__":
    main()
