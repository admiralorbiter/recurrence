"""Sprint S13.3 Stage 1: Diagnostic Microscope on B=1 vs B=5 Future Trajectory Divergence.

Evaluates marked_object_p01 under constant drive starting from the canonical B=1 N=0 state.
Checkpoints N in {1, 2, 4, 8, 16, 32, 64} comparing:
- Layer-by-layer RG-LRU recurrent states (max abs diff & cosine similarity)
- Conv1D states (max abs diff)
- KV cache keys/values (max abs diff)
- C_R, Q_R, output axis cosine
- A->B matching and wrong-value displacements

Evaluated under both BF16 and FP32 to isolate numerical accumulation vs state/cache plumbing.
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
    project_onto_axis,
    compute_recurrent_state_diff_vec,
    compute_recurrent_geometry,
    compute_logit_axis_cosine,
    advance_stream_along_horizons,
    advance_batched_stream_along_horizons,
)


@torch.inference_mode()
def run_diagnostic_for_dtype(dtype_str: str = "bfloat16"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
    print(f"\n==========================================================================================")
    print(f"RUNNING DIAGNOSTIC MICROSCOPE: DTYPE = {dtype_str.upper()} on {device.upper()}")
    print(f"==========================================================================================")

    model_id = "google/recurrentgemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    adapter = RecurrentGemmaAdapter(model=model, tokenizer=tokenizer, device=device, dtype=dtype)

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

    # N=0 Frozen Ruler & Baseline Vectors
    out_a0, _ = adapter.encode_sequence(toks_query, initial_snapshot=s_a_0.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
    out_b0, _ = adapter.encode_sequence(toks_query, initial_snapshot=s_b_0.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
    u_0_a2b, norm_0_a2b = compute_frozen_axis(out_a0[0], out_b0[0])
    r_0_a2b = compute_recurrent_state_diff_vec(s_a_0, s_b_0)

    # Drive Stream: Constant Drive
    stream = generate_single_drive_stream(2048, regime="constant", seed=5042, tokenizer=tokenizer, audited_pool=audited_pool, excluded_token_ids=pair_excluded)
    checkpoints = [1, 2, 4, 8, 16, 32, 64]
    horizons = [0] + checkpoints

    # Run B=1 Reference Future (Intact Arm)
    snaps_a_b1 = advance_stream_along_horizons(adapter, s_a_0, stream, horizons=horizons, arm="intact_recurrence")
    snaps_b_b1 = advance_stream_along_horizons(adapter, s_b_0, stream, horizons=horizons, arm="intact_recurrence")
    snaps_c_b1 = advance_stream_along_horizons(adapter, s_c_0, stream, horizons=horizons, arm="intact_recurrence")
    snaps_d_b1 = advance_stream_along_horizons(adapter, s_d_0, stream, horizons=horizons, arm="intact_recurrence")

    # Run B=5 Optimized Future (Intact Arm)
    init_snapshots = [s_a_0, s_b_0, s_c_0, s_d_0, s_cross_0]
    b5_snaps = advance_batched_stream_along_horizons(adapter, init_snapshots, stream, horizons=horizons, arm="intact_recurrence")
    snaps_a_b5 = b5_snaps[0]
    snaps_b_b5 = b5_snaps[1]
    snaps_c_b5 = b5_snaps[2]
    snaps_d_b5 = b5_snaps[3]

    print(f"\n{'Step N':<8} | {'Max RG-LRU Diff':<16} | {'Max Conv Diff':<14} | {'Max KV Diff':<12} | {'Disp Match B=1':<16} | {'Disp Match B=5':<16} | {'Abs Diff':<10}")
    print("-" * 105)

    for n in checkpoints:
        sa_b1 = snaps_a_b1[n]
        sa_b5 = snaps_a_b5[n]
        sb_b1 = snaps_b_b1[n]
        sb_b5 = snaps_b_b5[n]

        # Max diff across layers for branch A
        max_rglru = max(torch.max(torch.abs(sa_b1.rglru[l].float() - sa_b5.rglru[l].float())).item() for l in sa_b1.rglru)
        max_conv = max(torch.max(torch.abs(sa_b1.conv[l].float() - sa_b5.conv[l].float())).item() for l in sa_b1.conv)
        max_kv = 0.0
        for l in sa_b1.kv:
            d_k = torch.max(torch.abs(sa_b1.kv[l]["key"].float() - sa_b5.kv[l]["key"].float())).item()
            d_v = torch.max(torch.abs(sa_b1.kv[l]["value"].float() - sa_b5.kv[l]["value"].float())).item()
            max_kv = max(max_kv, d_k, d_v)

        # Probes
        out_b_b1, _ = adapter.encode_sequence(toks_query, initial_snapshot=sb_b1.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
        out_m_b1, _ = adapter.encode_sequence(toks_query, initial_snapshot=swap_stores(sb_b1, sa_b1, "rglru"), step_by_step=False, return_logits=True, logits_to_keep=1)
        disp_m_b1, _ = project_onto_axis(out_m_b1[0], out_b_b1[0], u_0_a2b, norm_0_a2b)

        out_b_b5, _ = adapter.encode_sequence(toks_query, initial_snapshot=sb_b5.clone(), step_by_step=False, return_logits=True, logits_to_keep=1)
        out_m_b5, _ = adapter.encode_sequence(toks_query, initial_snapshot=swap_stores(sb_b5, sa_b5, "rglru"), step_by_step=False, return_logits=True, logits_to_keep=1)
        disp_m_b5, _ = project_onto_axis(out_m_b5[0], out_b_b5[0], u_0_a2b, norm_0_a2b)

        print(f"N={n:<6} | {max_rglru:<16.6f} | {max_conv:<14.6f} | {max_kv:<12.6f} | {disp_m_b1:<+16.2f} | {disp_m_b5:<+16.2f} | {abs(disp_m_b1 - disp_m_b5):<10.2f}")

    print("=" * 105)


def main():
    run_diagnostic_for_dtype("bfloat16")
    run_diagnostic_for_dtype("float32")


if __name__ == "__main__":
    main()
