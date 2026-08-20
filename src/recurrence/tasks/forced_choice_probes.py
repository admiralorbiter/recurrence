"""Sprint S14: Forced-Choice Logit Scoring & Mapping Equivariance Probes (Repaired).

Provides generic utilities for K-way forced-choice evaluation:
- Chat-template aware prompt formatting for instruction-tuned models
- Randomized letter label remapping (e.g. Interval 1 -> C, Interval 2 -> A, etc.)
- Exact next-token logit and probability extraction over candidate letter tokens
- Semantic distribution unpermuting
- Multi-remapping probe helper & Mapping-Equivariance calculation (Jensen-Shannon divergence)
- Fixed-token label bias detection
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.state.temporal_inventory import RecurrentStateSnapshot


@dataclass
class SemanticOption:
    key: str
    description: str


@dataclass
class ForcedChoiceMapping:
    """Represents a randomized assignment of semantic options to letter labels."""
    options: List[SemanticOption]
    key_to_label: Dict[str, str]       # e.g. {"interval_1": "C", ...}
    label_to_key: Dict[str, str]       # e.g. {"C": "interval_1", ...}
    label_token_ids: Dict[str, int]    # e.g. {"A": 235280, ...}
    letters: List[str]                 # e.g. ["A", "B", "C", "D", "E"]


def create_forced_choice_mapping(
    options: Sequence[SemanticOption],
    tokenizer: Any,
    seed: Optional[int] = None,
    custom_perm: Optional[Sequence[int]] = None,
) -> ForcedChoiceMapping:
    """Create a randomized mapping of semantic options to single-token letters [A, B, C, D, ...]."""
    k = len(options)
    all_letters = [chr(65 + i) for i in range(k)]  # ['A', 'B', 'C', ...]

    if custom_perm is not None:
        perm = list(custom_perm)
    elif seed is not None:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(k, generator=g).tolist()
    else:
        perm = list(range(k))

    key_to_label = {}
    label_to_key = {}
    label_token_ids = {}

    for opt_idx, p_idx in enumerate(perm):
        opt = options[opt_idx]
        letter = all_letters[p_idx]
        key_to_label[opt.key] = letter
        label_to_key[letter] = opt.key

    for letter in all_letters:
        # Encode single letter token (without leading space / prefix)
        tok_id = tokenizer.encode(letter, add_special_tokens=False)[-1]
        label_token_ids[letter] = tok_id

    return ForcedChoiceMapping(
        options=list(options),
        key_to_label=key_to_label,
        label_to_key=label_to_key,
        label_token_ids=label_token_ids,
        letters=all_letters,
    )


def format_forced_choice_prompt(
    mapping: ForcedChoiceMapping,
    preamble: str,
    question: str,
    tokenizer: Optional[Any] = None,
    use_chat_template: bool = True,
) -> str:
    """Format the evaluation prompt listing options in alphabetical label order (A, B, C, ...)."""
    lines = [preamble.strip(), "", question.strip(), ""]
    for letter in sorted(mapping.letters):
        key = mapping.label_to_key[letter]
        opt = next(o for o in mapping.options if o.key == key)
        lines.append(f"{letter}) {opt.description}")
    lines.append("")
    lines.append("Answer with only the single option letter:")
    raw_text = "\n".join(lines)

    if use_chat_template and tokenizer is not None and getattr(tokenizer, "chat_template", None) is not None:
        messages = [{"role": "user", "content": raw_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return raw_text


@torch.inference_mode()
def score_forced_choice_prompt(
    adapter: RecurrentGemmaAdapter,
    snapshot: RecurrentStateSnapshot,
    mapping: ForcedChoiceMapping,
    prompt_text: str,
) -> Dict[str, Any]:
    """Score next-token probabilities over forced-choice candidate labels from a snapshot.

    Returns:
        Dict containing:
        - raw_label_probs: Dict[letter, float]
        - raw_label_logits: Dict[letter, float]
        - semantic_probs: Dict[key, float]
        - predicted_letter: str
        - predicted_key: str
        - margin: float (difference between top prob and 2nd prob)
    """
    toks = adapter.tokenizer.encode(prompt_text, add_special_tokens=False)
    out_logits, _ = adapter.encode_sequence(
        toks,
        initial_snapshot=snapshot.clone(),
        step_by_step=False,
        return_logits=True,
        logits_to_keep=1,
    )
    last_logits = out_logits[0].float()  # Shape [vocab_size]

    # Gather logits for candidate letter tokens
    candidate_tokens = [mapping.label_token_ids[l] for l in mapping.letters]
    candidate_logits = torch.stack([last_logits[t] for t in candidate_tokens])
    candidate_probs = F.softmax(candidate_logits, dim=-1).tolist()

    raw_label_probs = {l: float(p) for l, p in zip(mapping.letters, candidate_probs)}
    raw_label_logits = {l: float(v.item()) for l, v in zip(mapping.letters, candidate_logits)}
    semantic_probs = {mapping.label_to_key[l]: float(p) for l, p in zip(mapping.letters, candidate_probs)}

    best_letter = max(raw_label_probs.keys(), key=lambda l: raw_label_probs[l])
    best_key = mapping.label_to_key[best_letter]

    sorted_probs = sorted(candidate_probs, reverse=True)
    margin = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)

    return {
        "raw_label_probs": raw_label_probs,
        "raw_label_logits": raw_label_logits,
        "semantic_probs": semantic_probs,
        "predicted_letter": best_letter,
        "predicted_key": best_key,
        "margin": float(margin),
    }


def compute_js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute Jensen-Shannon Divergence between two probability distributions over the same keys."""
    keys = sorted(p.keys())
    p_vec = [max(1e-12, p[k]) for k in keys]
    q_vec = [max(1e-12, q[k]) for k in keys]

    # Normalize
    sum_p = sum(p_vec)
    sum_q = sum(q_vec)
    p_vec = [x / sum_p for x in p_vec]
    q_vec = [x / sum_q for x in q_vec]

    m_vec = [0.5 * (pv + qv) for pv, qv in zip(p_vec, q_vec)]

    kl_pm = sum(pv * math.log2(pv / mv) for pv, mv in zip(p_vec, m_vec))
    kl_qm = sum(qv * math.log2(qv / mv) for qv, mv in zip(q_vec, m_vec))

    return 0.5 * (kl_pm + kl_qm)


@torch.inference_mode()
def evaluate_mapping_equivariance(
    adapter: RecurrentGemmaAdapter,
    snapshot: RecurrentStateSnapshot,
    options: Sequence[SemanticOption],
    preamble: str,
    question: str,
    seed_1: int = 100,
    seed_2: int = 200,
    true_key: Optional[str] = None,
    use_chat_template: bool = True,
) -> Dict[str, Any]:
    """Probe the same final snapshot under two independently randomized label permutations."""
    map_1 = create_forced_choice_mapping(options, adapter.tokenizer, seed=seed_1)
    prompt_1 = format_forced_choice_prompt(map_1, preamble, question, tokenizer=adapter.tokenizer, use_chat_template=use_chat_template)
    res_1 = score_forced_choice_prompt(adapter, snapshot, map_1, prompt_1)

    map_2 = create_forced_choice_mapping(options, adapter.tokenizer, seed=seed_2)
    prompt_2 = format_forced_choice_prompt(map_2, preamble, question, tokenizer=adapter.tokenizer, use_chat_template=use_chat_template)
    res_2 = score_forced_choice_prompt(adapter, snapshot, map_2, prompt_2)

    js_div = compute_js_divergence(res_1["semantic_probs"], res_2["semantic_probs"])
    semantic_agreement = (res_1["predicted_key"] == res_2["predicted_key"])
    fixed_letter_chosen = (res_1["predicted_letter"] == res_2["predicted_letter"])

    acc_1 = (res_1["predicted_key"] == true_key) if true_key is not None else None
    acc_2 = (res_2["predicted_key"] == true_key) if true_key is not None else None

    log_margin_1 = None
    log_margin_2 = None
    if true_key is not None:
        p_true_1 = res_1["semantic_probs"].get(true_key, 1e-12)
        p_max_other_1 = max(v for k, v in res_1["semantic_probs"].items() if k != true_key)
        log_margin_1 = math.log(max(1e-12, p_true_1)) - math.log(max(1e-12, p_max_other_1))

        p_true_2 = res_2["semantic_probs"].get(true_key, 1e-12)
        p_max_other_2 = max(v for k, v in res_2["semantic_probs"].items() if k != true_key)
        log_margin_2 = math.log(max(1e-12, p_true_2)) - math.log(max(1e-12, p_max_other_2))

    return {
        "m1_predicted_key": res_1["predicted_key"],
        "m1_predicted_letter": res_1["predicted_letter"],
        "m1_semantic_probs": res_1["semantic_probs"],
        "m1_raw_probs": res_1["raw_label_probs"],
        "m1_raw_logits": res_1["raw_label_logits"],
        "m1_acc": acc_1,
        "m1_log_margin": log_margin_1,
        "m2_predicted_key": res_2["predicted_key"],
        "m2_predicted_letter": res_2["predicted_letter"],
        "m2_semantic_probs": res_2["semantic_probs"],
        "m2_raw_probs": res_2["raw_label_probs"],
        "m2_raw_logits": res_2["raw_label_logits"],
        "m2_acc": acc_2,
        "m2_log_margin": log_margin_2,
        "js_divergence": js_div,
        "semantic_agreement": semantic_agreement,
        "fixed_letter_chosen": fixed_letter_chosen,
    }
