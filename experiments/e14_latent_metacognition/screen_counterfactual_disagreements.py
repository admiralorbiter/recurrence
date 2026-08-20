"""Sprint S14: Bidirectional Causal-Provenance Disagreement Screen.

Systematically screens all 24 canonical value pairs across filler regimes
to find eligible diagnostic cells where secret RG-LRU transplantation creates
a genuine behavioral disagreement between the Target's actual prior disposition
and the Replay Observer's prediction.

Key Design (Borrowed from GENE's Bidirectional Role-Swap Logic):
For each pair (val_a, val_b), we run BOTH causal directions:
  - Forward:  A_recipient <- B_donor  (transplant B's RG-LRU into A)
  - Reverse:  B_recipient <- A_donor  (transplant A's RG-LRU into B)

A convincing private-provenance effect should REVERSE with causal role,
rather than consistently favoring a particular token or lexical prior.

Eligibility Criteria:
1. Binary Sign Disagreement: sign(D_T) != sign(D_O) in at least one direction.
2. Bidirectional Consistency: margin shifts move in opposite directions for
   forward vs reverse transplants.
3. Full-Vocab Argmax Divergence: argmax P_T(y) != argmax P_O(y).
4. Target-Observer Margin Shift: |D_T - D_O| >= threshold.

Framing Note (Cross-Pollination from GENE Project):
This is a causal-provenance / source-monitoring assay, not an "ownership" test.
The question is: Can the model correctly report the causal provenance of its own
current or prior output disposition when that provenance differs from what an
observer can reconstruct from public history?

Decomposition Framework (C/D/R/A):
  C - Causal fact exists: target and observer genuinely have different dispositions.
  D - Discrimination/access: target behavior contains information about which
      private causal state occurred.
  R - Reporting competence: the model can map that discrimination into the
      requested reporting format.
  A - Answer correctness: emitted report matches ground truth.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.impulse_stimuli import get_filler_tokens_for_regime, build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs, MicroscopePair
from recurrence.interventions.surgical_swaps import swap_stores


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"


@torch.inference_mode()
def screen_single_direction(
    adapter: RecurrentGemmaAdapter,
    pair: MicroscopePair,
    s_recipient_0,
    s_donor_0,
    query_toks: List[int],
    tok_a_id: int,
    tok_b_id: int,
    direction_label: str,
):
    """Screen one causal direction: transplant donor RG-LRU into recipient."""
    s_target = swap_stores(s_recipient_0, s_donor_0, channels="rglru")
    s_observer = s_recipient_0.clone()

    out_tgt, _ = adapter.encode_sequence(
        query_toks, initial_snapshot=s_target, step_by_step=False,
        return_logits=True, logits_to_keep=1,
    )
    out_obs, _ = adapter.encode_sequence(
        query_toks, initial_snapshot=s_observer, step_by_step=False,
        return_logits=True, logits_to_keep=1,
    )

    lg_t = out_tgt[0].float()
    lg_o = out_obs[0].float()

    # Paired value margin: D = z(val_b) - z(val_a)
    d_t = (lg_t[tok_b_id] - lg_t[tok_a_id]).item()
    d_o = (lg_o[tok_b_id] - lg_o[tok_a_id]).item()
    margin_shift = d_t - d_o

    sign_flip = (d_t > 0) != (d_o > 0) and abs(d_t) > 0.01 and abs(d_o) > 0.01

    # Full output vocabulary argmax
    top_t_id = torch.argmax(lg_t).item()
    top_o_id = torch.argmax(lg_o).item()
    top_t_word = repr(adapter.tokenizer.decode([top_t_id]))
    top_o_word = repr(adapter.tokenizer.decode([top_o_id]))
    vocab_argmax_flip = (top_t_id != top_o_id)

    # Full-vocab KL divergence
    kl_div = F.kl_div(
        F.log_softmax(lg_o, dim=-1), F.softmax(lg_t, dim=-1), reduction="sum"
    ).item()
    max_logit_delta = torch.max(torch.abs(lg_t - lg_o)).item()

    return {
        "direction": direction_label,
        "d_target": d_t,
        "d_observer": d_o,
        "margin_shift": margin_shift,
        "sign_flip": sign_flip,
        "top_target_word": top_t_word,
        "top_observer_word": top_o_word,
        "vocab_argmax_flip": vocab_argmax_flip,
        "kl_div_nats": kl_div,
        "max_logit_delta": max_logit_delta,
    }


@torch.inference_mode()
def screen_pair_bidirectional(
    adapter: RecurrentGemmaAdapter,
    pair: MicroscopePair,
    pair_idx: int,
    regime: str,
    audited_pool: List[int],
) -> Dict[str, Any]:
    """Screen one pair in both causal directions under one filler regime."""
    tokenizer = adapter.tokenizer

    toks_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
    toks_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
    tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
    tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]
    excluded = set(toks_a + toks_b + [tok_a_id, tok_b_id])
    query_toks = tokenizer.encode(pair.query, add_special_tokens=False)

    seed = 42 + pair_idx * 100

    # Build 4,096-token origin states
    _, s_a_0 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
    _, s_b_0 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)

    filler_4k = get_filler_tokens_for_regime(
        regime, length=4096, seed=seed, audited_pool=audited_pool,
        tokenizer=tokenizer, excluded_token_ids=excluded,
    )
    for i in range(0, 4096, 512):
        chunk = filler_4k[i : i + 512]
        _, s_a_0 = adapter.encode_sequence(
            chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False,
        )
        _, s_b_0 = adapter.encode_sequence(
            chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False,
        )

    # Forward: A_recipient <- B_donor
    fwd = screen_single_direction(
        adapter, pair, s_a_0.clone(), s_b_0.clone(), query_toks,
        tok_a_id, tok_b_id, direction_label="A<-B",
    )

    # Reverse: B_recipient <- A_donor
    rev = screen_single_direction(
        adapter, pair, s_b_0.clone(), s_a_0.clone(), query_toks,
        tok_a_id, tok_b_id, direction_label="B<-A",
    )

    # Bidirectional consistency check:
    # If the effect is causal-role-dependent (not lexical), margin shifts should
    # move in opposite directions for forward vs reverse transplants.
    bidirectional_reversal = (fwd["margin_shift"] > 0) != (rev["margin_shift"] > 0)
    bidirectional_magnitude = abs(fwd["margin_shift"]) + abs(rev["margin_shift"])

    return {
        "pair_id": pair.pair_id,
        "family_id": pair.family_id,
        "regime": regime,
        "val_a": pair.val_a,
        "val_b": pair.val_b,
        "forward": fwd,
        "reverse": rev,
        "bidirectional_reversal": bidirectional_reversal,
        "bidirectional_magnitude": bidirectional_magnitude,
        "any_sign_flip": fwd["sign_flip"] or rev["sign_flip"],
        "any_argmax_flip": fwd["vocab_argmax_flip"] or rev["vocab_argmax_flip"],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print("\n" + "=" * 115)
    print("BIDIRECTIONAL CAUSAL-PROVENANCE DISAGREEMENT SCREEN (24 Pairs x 3 Regimes x 2 Directions)")
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
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    pairs = build_microscope_pairs()
    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)

    regimes = ["random", "natural", "constant"]
    all_cells = []

    print(
        f"\nScreening {len(pairs)} pairs across {len(regimes)} regimes "
        f"({len(pairs) * len(regimes)} cells, 2 directions each)...\n"
    )

    for p_idx, pair in enumerate(pairs):
        for reg in regimes:
            cell = screen_pair_bidirectional(adapter, pair, p_idx, reg, audited_pool)
            all_cells.append(cell)
            fwd = cell["forward"]
            rev = cell["reverse"]
            marks = []
            if cell["any_sign_flip"]:
                marks.append("SIGN-FLIP")
            if cell["bidirectional_reversal"]:
                marks.append("BIDIR-REV")
            if cell["any_argmax_flip"]:
                marks.append("ARGMAX-FLIP")
            tag = f" [{', '.join(marks)}]" if marks else ""
            print(
                f"P{p_idx+1:02d} {pair.pair_id:<34} {reg:<9} | "
                f"Fwd(A<-B): {fwd['margin_shift']:+6.2f} | "
                f"Rev(B<-A): {rev['margin_shift']:+6.2f} | "
                f"BiMag: {cell['bidirectional_magnitude']:5.2f}"
                f"{tag}",
                flush=True,
            )

    # Aggregate summary
    sign_flip_cells = [c for c in all_cells if c["any_sign_flip"]]
    bidir_cells = [c for c in all_cells if c["bidirectional_reversal"]]
    argmax_cells = [c for c in all_cells if c["any_argmax_flip"]]
    strong_bidir = [
        c for c in all_cells
        if c["bidirectional_reversal"] and c["bidirectional_magnitude"] >= 0.30
    ]

    print("\n" + "=" * 115)
    print("BIDIRECTIONAL CAUSAL-PROVENANCE SCREEN SUMMARY")
    print("=" * 115)
    print(f"Total Cells Evaluated:                          {len(all_cells)}")
    print(f"Cells with Binary Sign Flip (either direction): {len(sign_flip_cells)}")
    print(f"Cells with Bidirectional Reversal:               {len(bidir_cells)}")
    print(f"Cells with Strong Bidirectional Reversal (>=0.30):{len(strong_bidir)}")
    print(f"Cells with Full-Vocab Argmax Flip:               {len(argmax_cells)}")
    print("=" * 115)

    if strong_bidir:
        print("\nEligible Strong Bidirectional-Reversal Cells:")
        for c in strong_bidir:
            f = c["forward"]
            r = c["reverse"]
            print(
                f"  {c['pair_id']} ({c['regime']}): "
                f"Fwd shift={f['margin_shift']:+.3f}, "
                f"Rev shift={r['margin_shift']:+.3f}, "
                f"BiMag={c['bidirectional_magnitude']:.3f}"
            )

    if sign_flip_cells:
        print("\nCells with Sign Flip:")
        for c in sign_flip_cells:
            f = c["forward"]
            r = c["reverse"]
            which = []
            if f["sign_flip"]:
                which.append(f"Fwd D_T={f['d_target']:+.3f} vs D_O={f['d_observer']:+.3f}")
            if r["sign_flip"]:
                which.append(f"Rev D_T={r['d_target']:+.3f} vs D_O={r['d_observer']:+.3f}")
            print(f"  {c['pair_id']} ({c['regime']}): {'; '.join(which)}")

    # Save manifest
    out_dir = Path("results") / "e14_latent_metacognition" / "counterfactual_screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bidirectional_provenance_manifest.json"

    # Strip non-serializable fields
    serializable_cells = []
    for c in all_cells:
        sc = {k: v for k, v in c.items()}
        serializable_cells.append(sc)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": model_id,
                "pinned_revision": PINNED_IT_REVISION,
                "total_cells": len(all_cells),
                "sign_flip_count": len(sign_flip_cells),
                "bidir_reversal_count": len(bidir_cells),
                "strong_bidir_count": len(strong_bidir),
                "argmax_flip_count": len(argmax_cells),
                "strong_bidir_cells": strong_bidir,
                "sign_flip_cells": sign_flip_cells,
                "all_cells": serializable_cells,
            },
            f,
            indent=2,
        )
    print(f"\nManifest saved to {out_file}\n")


if __name__ == "__main__":
    main()
