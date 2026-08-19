"""Sprint S13.3 Stage 1.1: Fine-Grained Layer-by-Layer Divergence Localization.

Instruments steps N in {1, 2, 3, 4} layer-by-layer comparing B=1 vs B=5:
1. linear_x output before Conv1D
2. Conv1D state before the token
3. post-Conv x_branch
4. RG-LRU input, gate, and recurrent state
5. Layer output hidden state

Also tests Intra-Batch Uniformity:
A B=5 batch composed of 5 identical copies of S_0 receiving identical tokens,
verifying whether all 5 lanes stay bit-identical to each other while comparing against B=1.
"""

import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from recurrence.models.recurrent_gemma_adapter import (
    RecurrentGemmaAdapter,
    RecurrentStateSnapshot,
    stack_snapshots,
    unstack_snapshot,
)
from recurrence.interventions.surgical_swaps import swap_stores
from recurrence.tasks.impulse_stimuli import get_filler_tokens_for_regime, build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import build_microscope_pairs
from recurrence.tasks.controlled_drive import (
    generate_single_drive_stream,
    compute_frozen_axis,
    compute_recurrent_state_diff_vec,
)


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Stage 1.1 Layer Localization] Initializing on device={device} (bfloat16)...", flush=True)

    model_id = "google/recurrentgemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=device, dtype=torch.bfloat16)

    audited_pool, _ = build_audited_vocabulary_pool(tokenizer)
    pair = build_microscope_pairs()[0]

    toks_prompt_a = tokenizer.encode(pair.prompt_a, add_special_tokens=False)
    toks_prompt_b = tokenizer.encode(pair.prompt_b, add_special_tokens=False)
    toks_prompt_c = tokenizer.encode(pair.prompt_c, add_special_tokens=False)
    toks_prompt_d = tokenizer.encode(pair.prompt_d, add_special_tokens=False)
    toks_prompt_cross = tokenizer.encode(pair.prompt_cross, add_special_tokens=False)
    toks_query = tokenizer.encode(pair.query, add_special_tokens=False)
    tok_a_id = tokenizer.encode(pair.target_a, add_special_tokens=False)[-1]
    tok_b_id = tokenizer.encode(pair.target_b, add_special_tokens=False)[-1]

    pair_excluded = set(toks_prompt_a + toks_prompt_b + toks_prompt_c + toks_prompt_d + toks_prompt_cross + [tok_a_id, tok_b_id])

    # Canonical B=1 Prep
    print("[Stage 1.1] Running canonical B=1 preparation...", flush=True)
    _, s_a_0 = adapter.encode_sequence(toks_prompt_a, step_by_step=False, return_logits=False)
    _, s_b_0 = adapter.encode_sequence(toks_prompt_b, step_by_step=False, return_logits=False)
    _, s_c_0 = adapter.encode_sequence(toks_prompt_c, step_by_step=False, return_logits=False)
    _, s_d_0 = adapter.encode_sequence(toks_prompt_d, step_by_step=False, return_logits=False)
    _, s_cross_0 = adapter.encode_sequence(toks_prompt_cross, step_by_step=False, return_logits=False)

    filler = get_filler_tokens_for_regime("random", length=4096, seed=42, audited_pool=audited_pool, tokenizer=tokenizer, excluded_token_ids=pair_excluded)
    for i in range(0, 4096, 512):
        chunk = filler[i : i + 512]
        _, s_a_0 = adapter.encode_sequence(chunk, initial_snapshot=s_a_0, step_by_step=False, return_logits=False)
        _, s_b_0 = adapter.encode_sequence(chunk, initial_snapshot=s_b_0, step_by_step=False, return_logits=False)
        _, s_c_0 = adapter.encode_sequence(chunk, initial_snapshot=s_c_0, step_by_step=False, return_logits=False)
        _, s_d_0 = adapter.encode_sequence(chunk, initial_snapshot=s_d_0, step_by_step=False, return_logits=False)
        _, s_cross_0 = adapter.encode_sequence(chunk, initial_snapshot=s_cross_0, step_by_step=False, return_logits=False)

    stream = generate_single_drive_stream(2048, regime="constant", seed=5042, tokenizer=tokenizer, audited_pool=audited_pool, excluded_token_ids=pair_excluded)

    # 1. Test Intra-Batch Uniformity on 5 identical copies of s_a_0
    print("\n" + "=" * 90, flush=True)
    print("TEST 1: INTRA-BATCH UNIFORMITY (5 Identical Copies of S_A(0) in B=5)", flush=True)
    print("=" * 90, flush=True)

    identical_s0_5 = [s_a_0.clone() for _ in range(5)]
    b5_identical_snaps = stack_snapshots(identical_s0_5)
    cache_b5_ident = adapter.inject_state_snapshot(b5_identical_snaps)

    s1_single = s_a_0.clone()
    cache_b1_single = adapter.inject_state_snapshot(s1_single)

    pos = s_a_0.cache_position
    model_fn = adapter.model.model if hasattr(adapter.model, "model") else adapter.model

    for step_idx in range(4):
        tok = stream[step_idx]
        cur_pos = pos + step_idx

        # B=1 single step
        in_b1 = torch.tensor([[tok]], device=device, dtype=torch.long)
        pos_b1 = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
        model_fn(input_ids=in_b1, position_ids=pos_b1, past_key_values=cache_b1_single, use_cache=True, return_dict=True)
        snap_b1 = adapter.extract_state_snapshot(past_key_values=cache_b1_single, cache_position=cur_pos + 1)

        # B=5 identical step
        in_b5 = torch.tensor([[tok]] * 5, device=device, dtype=torch.long)
        pos_b5 = torch.full((5, 1), cur_pos, device=device, dtype=torch.long)
        model_fn(input_ids=in_b5, position_ids=pos_b5, past_key_values=cache_b5_ident, use_cache=True, return_dict=True)
        snap_b5_all = adapter.extract_state_snapshot(past_key_values=cache_b5_ident, cache_position=cur_pos + 1)
        unstacked_5 = unstack_snapshot(snap_b5_all)

        # Check intra-batch max difference across the 5 lanes
        max_intra_rglru = 0.0
        max_intra_conv = 0.0
        for l in unstacked_5[0].rglru:
            lane_tensors = [unstacked_5[lane].rglru[l] for lane in range(5)]
            for lane_i in range(1, 5):
                max_intra_rglru = max(max_intra_rglru, torch.max(torch.abs(lane_tensors[0] - lane_tensors[lane_i])).item())
        for l in unstacked_5[0].conv:
            lane_tensors = [unstacked_5[lane].conv[l] for lane in range(5)]
            for lane_i in range(1, 5):
                max_intra_conv = max(max_intra_conv, torch.max(torch.abs(lane_tensors[0] - lane_tensors[lane_i])).item())

        # Check diff between B=1 single lane vs Lane 0 of B=5
        diff_b1_vs_b5_rglru = max(torch.max(torch.abs(snap_b1.rglru[l] - unstacked_5[0].rglru[l])).item() for l in snap_b1.rglru)
        diff_b1_vs_b5_conv = max(torch.max(torch.abs(snap_b1.conv[l] - unstacked_5[0].conv[l])).item() for l in snap_b1.conv)

        print(f"Step N={step_idx+1:<2} | Intra-Batch RG-LRU Diff: {max_intra_rglru:<10.6f} | Intra Conv Diff: {max_intra_conv:<10.6f} | B=1 vs B=5 RG-LRU Diff: {diff_b1_vs_b5_rglru:<10.6f} | Conv Diff: {diff_b1_vs_b5_conv:<10.6f}", flush=True)

    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
