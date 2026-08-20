"""Mine audited pairwise lexical rank reversals from top bidirectional-reversal cells.

For the top N cells by |R_role|, re-run target and observer logit extraction,
then search an audited vocabulary of clean single-token words for pairs (x, y) where:

  z_T(x) - z_T(y) > margin   AND   z_O(x) - z_O(y) < -margin

Uses vectorized torch operations for the pairwise comparison.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import get_filler_tokens_for_regime, build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs
from recurrence.interventions.surgical_swaps import swap_stores


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"
MARGIN_THRESHOLD = 0.30
TOP_N_CELLS = 12


def build_audited_word_set(tokenizer) -> Dict[int, str]:
    """Build vocabulary of clean printable single-token words. Skip re-encode check."""
    word_set = {}
    vocab = tokenizer.get_vocab()
    special_ids = set(tokenizer.all_special_ids)
    for token_str, token_id in vocab.items():
        if token_id in special_ids:
            continue
        clean = token_str.lstrip("\u2581").strip()
        if (
            len(clean) >= 3
            and clean.isalpha()
            and clean.islower()
            and clean.isascii()
        ):
            word_set[token_id] = clean
    return word_set


def find_pairwise_reversals_prefiltered(
    lg_t: torch.Tensor,
    lg_o: torch.Tensor,
    word_ids: List[int],
    audited_words: Dict[int, str],
    margin: float,
    prefilter_k: int = 500,
) -> List[Dict[str, Any]]:
    """Find pairwise rank reversals by pre-filtering to tokens with large T-O diffs.

    Strategy: A rank reversal on pair (x, y) requires z_T(x) > z_T(y) + margin
    AND z_O(y) > z_O(x) + margin. This means x must be relatively target-preferred
    and y must be relatively observer-preferred. So we pre-filter to the top-K
    tokens by (z_T - z_O) and bottom-K tokens by (z_T - z_O), then do pairwise
    comparison only within those ~2K candidates.
    """
    ids = torch.tensor(word_ids, dtype=torch.long)
    zt = lg_t[ids]
    zo = lg_o[ids]
    diff = zt - zo  # positive = target-preferred, negative = observer-preferred

    k = min(prefilter_k, len(word_ids) // 2)

    # Top-K target-preferred tokens (candidates for "x" in reversal)
    top_vals, top_idx = torch.topk(diff, k)
    # Top-K observer-preferred tokens (candidates for "y" in reversal)
    bot_vals, bot_idx = torch.topk(-diff, k)

    reversals = []
    for ti in range(k):
        i = top_idx[ti].item()
        wid_x = word_ids[i]
        zt_x = zt[i].item()
        zo_x = zo[i].item()

        for bi in range(k):
            j = bot_idx[bi].item()
            if i == j:
                continue
            wid_y = word_ids[j]
            zt_y = zt[j].item()
            zo_y = zo[j].item()

            target_gap = zt_x - zt_y
            observer_gap = zo_x - zo_y

            if target_gap > margin and observer_gap < -margin:
                reversals.append({
                    "word_x": audited_words[wid_x],
                    "word_y": audited_words[wid_y],
                    "tok_x": wid_x,
                    "tok_y": wid_y,
                    "target_gap": target_gap,
                    "observer_gap": observer_gap,
                    "total_disagreement": target_gap - observer_gap,
                })

    reversals.sort(key=lambda r: r["total_disagreement"], reverse=True)
    return reversals


@torch.inference_mode()
def mine_cell_disagreements(
    adapter: RecurrentGemmaAdapter,
    cell_info: Dict[str, Any],
    pair_idx: int,
    audited_pool: List[int],
    audited_words: Dict[int, str],
    word_ids: List[int],
    margin: float = MARGIN_THRESHOLD,
) -> Dict[str, Any]:
    """For one cell, extract target/observer logits and find pairwise rank reversals."""
    tokenizer = adapter.tokenizer
    pairs = build_microscope_pairs()
    pair = next(p for p in pairs if p.pair_id == cell_info["pair_id"])
    regime = cell_info["regime"]

    toks_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
    toks_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
    tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
    tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]
    excluded = set(toks_a + toks_b + [tok_a_id, tok_b_id])
    query_toks = tokenizer.encode(pair.query, add_special_tokens=False)

    seed = 42 + pair_idx * 100

    _, s_a_0 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
    _, s_b_0 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)

    filler_4k = get_filler_tokens_for_regime(
        regime, length=4096, seed=seed, audited_pool=audited_pool,
        tokenizer=tokenizer, excluded_token_ids=excluded,
    )
    for i in range(0, 4096, 512):
        chunk = filler_4k[i : i + 512]
        _, s_a_0 = adapter.encode_sequence(chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False)
        _, s_b_0 = adapter.encode_sequence(chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False)

    # Forward direction: A_recipient <- B_donor
    s_target = swap_stores(s_a_0.clone(), s_b_0.clone(), channels="rglru")
    s_observer = s_a_0.clone()

    out_tgt, _ = adapter.encode_sequence(query_toks, initial_snapshot=s_target, step_by_step=False, return_logits=True, logits_to_keep=1)
    out_obs, _ = adapter.encode_sequence(query_toks, initial_snapshot=s_observer, step_by_step=False, return_logits=True, logits_to_keep=1)

    lg_t = out_tgt[0].float()
    lg_o = out_obs[0].float()

    reversals = find_pairwise_reversals_prefiltered(lg_t, lg_o, word_ids, audited_words, margin)

    # Top-1 audited words
    ids_tensor = torch.tensor(word_ids, dtype=torch.long)
    top_t_idx = torch.argmax(lg_t[ids_tensor]).item()
    top_o_idx = torch.argmax(lg_o[ids_tensor]).item()
    top_t_wid = word_ids[top_t_idx]
    top_o_wid = word_ids[top_o_idx]

    return {
        "pair_id": cell_info["pair_id"],
        "regime": regime,
        "r_role": cell_info["r_role"],
        "top_target_word": audited_words[top_t_wid],
        "top_observer_word": audited_words[top_o_wid],
        "audited_argmax_flip": top_t_wid != top_o_wid,
        "n_pairwise_reversals": len(reversals),
        "top_reversals": reversals[:10],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print("\n" + "=" * 115)
    print(f"AUDITED PAIRWISE LEXICAL RANK-REVERSAL MINER (Top {TOP_N_CELLS} cells by |R_role|)")
    print(f"Model: {model_id} (revision: {PINNED_IT_REVISION[:10]}...)")
    print(f"Margin threshold: {MARGIN_THRESHOLD} logits on both sides")
    print("=" * 115)

    reanalysis_path = Path("results/e14_latent_metacognition/counterfactual_screen/role_residual_reanalysis.json")
    with open(reanalysis_path, "r", encoding="utf-8") as f:
        reanalysis = json.load(f)

    top_cells = reanalysis["cells_sorted_by_abs_r_role"][:TOP_N_CELLS]

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=PINNED_IT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=PINNED_IT_REVISION, torch_dtype=torch.bfloat16,
    ).to(device)
    adapter = RecurrentGemmaAdapter(
        model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)
    audited_words = build_audited_word_set(tokenizer)
    word_ids = sorted(audited_words.keys())
    print(f"Audited word vocabulary: {len(audited_words)} clean single-token words", flush=True)

    pairs = build_microscope_pairs()
    pair_id_to_idx = {p.pair_id: i for i, p in enumerate(pairs)}

    all_results = []
    for cell in top_cells:
        pair_idx = pair_id_to_idx[cell["pair_id"]]
        result = mine_cell_disagreements(
            adapter, cell, pair_idx, audited_pool, audited_words, word_ids,
        )
        all_results.append(result)

        n_rev = result["n_pairwise_reversals"]
        top_str = ""
        if result["top_reversals"]:
            r = result["top_reversals"][0]
            top_str = f" Best: '{r['word_x']}' vs '{r['word_y']}' (T:{r['target_gap']:+.2f}, O:{r['observer_gap']:+.2f})"
        print(
            f"  {result['pair_id']:<34} {result['regime']:<9} | "
            f"R_role={result['r_role']:+.3f} | "
            f"T='{result['top_target_word']}' O='{result['top_observer_word']}' | "
            f"Reversals: {n_rev}"
            f"{top_str}",
            flush=True,
        )

    total_reversals = sum(r["n_pairwise_reversals"] for r in all_results)
    cells_with_reversals = sum(1 for r in all_results if r["n_pairwise_reversals"] > 0)

    print("\n" + "=" * 115)
    print("AUDITED PAIRWISE RANK-REVERSAL SUMMARY")
    print("=" * 115)
    print(f"Cells screened:                    {len(all_results)}")
    print(f"Cells with pairwise reversals:     {cells_with_reversals}")
    print(f"Total pairwise reversals found:    {total_reversals}")
    print(f"Margin threshold:                  {MARGIN_THRESHOLD} logits (both sides)")
    print("=" * 115)

    if total_reversals > 0:
        print("\nTop Candidate Disagreement Pairs (across all cells):")
        all_top = []
        for r in all_results:
            for rev in r["top_reversals"][:3]:
                all_top.append({**rev, "pair_id": r["pair_id"], "regime": r["regime"], "r_role": r["r_role"]})
        all_top.sort(key=lambda x: x["total_disagreement"], reverse=True)
        for t in all_top[:20]:
            print(
                f"  {t['pair_id']:<34} {t['regime']:<9} | "
                f"'{t['word_x']}' vs '{t['word_y']}' | "
                f"T_gap: {t['target_gap']:+.3f}, O_gap: {t['observer_gap']:+.3f} | "
                f"Total: {t['total_disagreement']:.3f}"
            )

    out_dir = Path("results/e14_latent_metacognition/counterfactual_screen")
    out_file = out_dir / "audited_pairwise_reversals.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "pinned_revision": PINNED_IT_REVISION,
            "margin_threshold": MARGIN_THRESHOLD,
            "audited_vocab_size": len(audited_words),
            "cells_screened": len(all_results),
            "cells_with_reversals": cells_with_reversals,
            "total_reversals": total_reversals,
            "cell_results": all_results,
        }, f, indent=2)
    print(f"\nSaved to {out_file}\n")


if __name__ == "__main__":
    main()
