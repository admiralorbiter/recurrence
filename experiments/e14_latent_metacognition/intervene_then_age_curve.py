"""Sprint S14: Intervene-Then-Age Potency Retention Curve.

Measures how much causal and physical signal survives at a common endpoint T=256
when an RG-LRU transplant is applied at t_intervention in {0, 64, 128, 192, 256}
and then evolved under a single frozen neutral token stream.

Endpoints measured at T=256:
1. Target vs Observer Full-Vocab KL divergence
2. Max absolute logit difference
3. Donor-directed cloze shift [(z_B - z_A)_target - (z_B - z_A)_observer]
4. Target-Observer RG-LRU Euclidean distance and cosine similarity (C_R)
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
from recurrence.tasks.specificity_microscope import build_microscope_pairs
from recurrence.interventions.surgical_swaps import swap_stores


PINNED_IT_REVISION = "2766eb5d4264c6c0357803990791f9ab9cd50f8e"


@torch.inference_mode()
def compute_rglru_distance(s_target, s_observer):
    dists = []
    cosines = []
    for l in s_target.rglru.keys():
        if l in s_observer.rglru:
            ht = s_target.rglru[l].float().flatten()
            ho = s_observer.rglru[l].float().flatten()
            dists.append(torch.norm(ht - ho).item())
            nt = torch.norm(ht).item()
            no = torch.norm(ho).item()
            if nt > 1e-8 and no > 1e-8:
                cosines.append(torch.dot(ht, ho).item() / (nt * no))
    return {
        "euclidean_dist": float(sum(dists) / len(dists)) if dists else 0.0,
        "cosine_sim": float(sum(cosines) / len(cosines)) if cosines else 1.0,
    }


@torch.inference_mode()
def measure_cloze_and_divergence(adapter, snapshot_target, snapshot_observer, query_tokens, tok_a_id, tok_b_id):
    out_target, _ = adapter.encode_sequence(query_tokens, initial_snapshot=snapshot_target.clone(), return_logits=True, logits_to_keep=1)
    out_obs, _ = adapter.encode_sequence(query_tokens, initial_snapshot=snapshot_observer.clone(), return_logits=True, logits_to_keep=1)

    lt = out_target[0].float()
    lo = out_obs[0].float()

    pt = F.softmax(lt, dim=-1)
    lpt = F.log_softmax(lt, dim=-1)
    lpo = F.log_softmax(lo, dim=-1)

    kl_t_o = F.kl_div(lpo, pt, reduction="sum").item()
    max_delta = torch.max(torch.abs(lt - lo)).item()

    cloze_shift = (lt[tok_b_id] - lt[tok_a_id]).item() - (lo[tok_b_id] - lo[tok_a_id]).item()

    return {
        "kl_nats": float(kl_t_o),
        "max_logit_delta": float(max_delta),
        "donor_cloze_shift": float(cloze_shift),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/recurrentgemma-2b-it"
    print(f"\n" + "=" * 95)
    print(f"INTERVENE-THEN-AGE POTENCY RETENTION CURVE (Endpoint T=256 tokens)")
    print(f"Model: {model_id} (revision: {PINNED_IT_REVISION[:10]}...)")
    print("=" * 95)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=PINNED_IT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=PINNED_IT_REVISION, torch_dtype=torch.bfloat16).to(device)
    adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16)
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    pair = build_microscope_pairs()[0] # marked_object_p01_amber_cobalt
    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)

    toks_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
    toks_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
    tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
    tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]
    excluded = set(toks_a + toks_b + [tok_a_id, tok_b_id])
    query_toks = tokenizer.encode(pair.query, add_special_tokens=False)

    # 1. Prepare 4,096-token canonical origin states
    print(f"\n[1] Constructing canonical B=1 4,096-token origin states S_A(0) and S_B(0)...")
    _, s_a_0 = adapter.encode_sequence(toks_a, step_by_step=False, return_logits=False)
    _, s_b_0 = adapter.encode_sequence(toks_b, step_by_step=False, return_logits=False)

    filler_4k = get_filler_tokens_for_regime("random", length=4096, seed=42, audited_pool=audited_pool, tokenizer=tokenizer, excluded_token_ids=excluded)
    for i in range(0, 4096, 512):
        chunk = filler_4k[i : i + 512]
        _, s_a_0 = adapter.encode_sequence(chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False)
        _, s_b_0 = adapter.encode_sequence(chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False)

    # 2. Frozen 256-token single neutral stream
    stream_256 = get_filler_tokens_for_regime("random", length=256, seed=1042, audited_pool=audited_pool, tokenizer=tokenizer, excluded_token_ids=excluded)

    # 3. Compute Observer (Sham) state at T=256
    print(f"[2] Evolving unperturbed Observer trajectory across full 256 tokens...")
    _, s_observer_256 = adapter.encode_sequence(stream_256, initial_snapshot=s_a_0.clone(), step_by_step=False, return_logits=False)

    # 4. Intervene at t in {0, 64, 128, 192, 256} and age to T=256
    intervention_times = [0, 64, 128, 192, 256]
    results = []

    print(f"\n[3] Running Intervene-Then-Age Sweep across t in {intervention_times}...")
    for t_step in intervention_times:
        # Evolve recipient and donor up to step t
        tokens_pre = stream_256[:t_step]
        tokens_post = stream_256[t_step:]

        if t_step > 0:
            _, s_rec_t = adapter.encode_sequence(tokens_pre, initial_snapshot=s_a_0.clone(), step_by_step=False, return_logits=False)
            _, s_don_t = adapter.encode_sequence(tokens_pre, initial_snapshot=s_b_0.clone(), step_by_step=False, return_logits=False)
        else:
            s_rec_t = s_a_0.clone()
            s_don_t = s_b_0.clone()

        # Surgical RG-LRU transplant at step t
        s_graft_t = swap_stores(s_rec_t, s_don_t, channels="rglru")

        # Evolve grafted state with remaining (256 - t) tokens to reach common T=256 endpoint
        if len(tokens_post) > 0:
            _, s_target_256 = adapter.encode_sequence(tokens_post, initial_snapshot=s_graft_t, step_by_step=False, return_logits=False)
        else:
            s_target_256 = s_graft_t

        # Measure at common T=256 endpoint against the Observer
        geom = compute_rglru_distance(s_target_256, s_observer_256)
        div = measure_cloze_and_divergence(adapter, s_target_256, s_observer_256, query_toks, tok_a_id, tok_b_id)

        rec = {
            "t_intervention": t_step,
            "aging_tokens": 256 - t_step,
            "rglru_euclidean_dist": geom["euclidean_dist"],
            "rglru_cosine_sim": geom["cosine_sim"],
            "full_vocab_kl_nats": div["kl_nats"],
            "max_logit_delta": div["max_logit_delta"],
            "donor_cloze_shift": div["donor_cloze_shift"],
        }
        results.append(rec)

    print("\n" + "=" * 95)
    print("INTERVENE-THEN-AGE POTENCY RETENTION SUMMARY (Measured at Endpoint T=256)")
    print("=" * 95)
    print(f"{'t_intervene':<12} | {'Aging Steps':<12} | {'RG-LRU Dist':<12} | {'Cosine Sim':<12} | {'KL (nats)':<12} | {'Max Logit Delta':<16} | {'Cloze Shift':<12}")
    print("-" * 95)
    for r in results:
        print(f"t = {r['t_intervention']:<8} | {r['aging_tokens']:<12} | {r['rglru_euclidean_dist']:<12.4f} | {r['rglru_cosine_sim']:<12.4f} | {r['full_vocab_kl_nats']:<12.6f} | {r['max_logit_delta']:<16.4f} | {r['donor_cloze_shift']:<+12.4f}")
    print("=" * 95)

    # Save results
    out_dir = Path("results") / "e14_latent_metacognition" / "aging_potency_curve"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "intervene_then_age_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"pair_id": pair.pair_id, "endpoint_T": 256, "curve": results}, f, indent=2)
    print(f"\nReport saved to {out_file}\n")


if __name__ == "__main__":
    main()
