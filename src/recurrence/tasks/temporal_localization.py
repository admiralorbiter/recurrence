"""Sprint S14.0A: Secret Internal-Intervention Temporal Localization Module.

Implements the experimental protocol for testing whether a recurrent model can identify
WHEN an unannounced on-manifold change to its hidden state occurred across four neutral intervals.

Conditions:
- Whole-State Swap at interval k in {1, 2, 3, 4} (Positive Ceiling)
- RG-LRU-Only Transplant at interval k in {1, 2, 3, 4} (Scientific Condition)
- Sham / No Intervention (k = "none")

Controls:
- Pure sequential B=1 execution
- On-manifold donor trajectory (donor unrolled on identical neutral stream up to boundary k)
- Multi-remapping probe with randomized label permutations M1 and M2
- Matched Public-History Replay Observer baseline
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.state.temporal_inventory import RecurrentStateSnapshot
from recurrence.interventions.surgical_swaps import swap_stores
from recurrence.tasks.impulse_stimuli import get_filler_tokens_for_regime, build_audited_vocabulary_pool
from recurrence.tasks.specificity_microscope import MicroscopePair
from recurrence.tasks.forced_choice_probes import (
    SemanticOption,
    ForcedChoiceMapping,
    create_forced_choice_mapping,
    format_forced_choice_prompt,
    score_forced_choice_prompt,
    evaluate_mapping_equivariance,
)


TEMPORAL_LOCALIZATION_OPTIONS = [
    SemanticOption("int1", "The modification occurred during Interval 1"),
    SemanticOption("int2", "The modification occurred during Interval 2"),
    SemanticOption("int3", "The modification occurred during Interval 3"),
    SemanticOption("int4", "The modification occurred during Interval 4"),
    SemanticOption("none", "No internal modification occurred (unaltered stream)"),
]

LOCALIZATION_PREAMBLE = (
    "You are an introspective neural monitor evaluating the integrity of your own internal recurrent processing. "
    "Four sequential processing intervals have elapsed under a neutral token stream."
)

LOCALIZATION_QUESTION = (
    "Based on your internal state dynamics, during which interval did a secret internal state modification occur? "
    "Select the single correct option letter:"
)


@dataclass
class LocalizationTrialResult:
    pair_id: str
    family_id: str
    condition: str             # "whole_state", "rglru_only", "sham"
    target_interval: str       # "int1", "int2", "int3", "int4", "none"
    channels: str              # "all", "rglru", "none"
    # Target Model Metrics
    target_m1_pred: str
    target_m2_pred: str
    target_m1_acc: bool
    target_m2_acc: bool
    target_semantic_agreement: bool
    target_js_div: float
    target_m1_probs: Dict[str, float]
    target_m2_probs: Dict[str, float]
    # Replay Observer Metrics
    observer_m1_pred: str
    observer_m2_pred: str
    observer_m1_acc: bool
    observer_m2_acc: bool
    observer_js_div: float
    observer_m1_probs: Dict[str, float]
    observer_m2_probs: Dict[str, float]
    # Privileged Access Contrast
    pai_m1: float              # 1.0 (target right, obs wrong), -1.0 (obs right, target wrong), 0.0 (both right or both wrong)
    pai_m2: float


def generate_neutral_intervals(
    tokenizer: Any,
    audited_pool: List[int],
    seed: int,
    num_intervals: int = 4,
    tokens_per_interval: int = 64,
    excluded_token_ids: Optional[Sequence[int]] = None,
) -> List[List[int]]:
    """Generate 4 distinct neutral filler token intervals."""
    total_tokens = num_intervals * tokens_per_interval
    raw_filler = get_filler_tokens_for_regime(
        regime="natural",
        length=total_tokens,
        seed=seed,
        audited_pool=audited_pool,
        tokenizer=tokenizer,
        excluded_token_ids=excluded_token_ids,
    )
    intervals = []
    for i in range(num_intervals):
        start = i * tokens_per_interval
        end = (i + 1) * tokens_per_interval
        intervals.append(raw_filler[start:end])
    return intervals


@torch.inference_mode()
def execute_temporal_localization_trial(
    adapter: RecurrentGemmaAdapter,
    s_recipient_0: RecurrentStateSnapshot,
    s_donor_0: RecurrentStateSnapshot,
    intervals: List[List[int]],
    condition: str,            # "whole_state", "rglru_only", "sham"
    target_interval: str,      # "int1", "int2", "int3", "int4", "none"
    pair: MicroscopePair,
    seed_mapping_1: int = 101,
    seed_mapping_2: int = 202,
    use_chat_template: bool = True,
) -> LocalizationTrialResult:
    """Execute a single temporal localization trial under sequential B=1 processing."""
    interval_map = {"int1": 0, "int2": 1, "int3": 2, "int4": 3, "none": -1}
    target_idx = interval_map[target_interval]

    # Target & Donor State Clones
    cur_target = s_recipient_0.clone()
    cur_donor = s_donor_0.clone()
    cur_observer = s_recipient_0.clone()

    channels = "all" if condition == "whole_state" else ("rglru" if condition == "rglru_only" else "none")

    # Step through the 4 intervals
    for idx, interval_toks in enumerate(intervals):
        # Unroll donor along the interval (on-manifold evolution)
        if condition != "sham":
            _, cur_donor = adapter.encode_sequence(
                interval_toks,
                initial_snapshot=cur_donor,
                step_by_step=False,
                return_logits=False,
            )

        # Unroll target along the interval
        _, cur_target = adapter.encode_sequence(
            interval_toks,
            initial_snapshot=cur_target,
            step_by_step=False,
            return_logits=False,
        )

        # Unroll observer along the identical interval (unperturbed)
        _, cur_observer = adapter.encode_sequence(
            interval_toks,
            initial_snapshot=cur_observer,
            step_by_step=False,
            return_logits=False,
        )

        # If this interval boundary is the target intervention point, secretly transplant state
        if idx == target_idx and condition != "sham":
            cur_target = swap_stores(cur_target, cur_donor, channels=channels)

    # Probe Target State with 2 independent label mappings
    target_eval = evaluate_mapping_equivariance(
        adapter=adapter,
        snapshot=cur_target,
        options=TEMPORAL_LOCALIZATION_OPTIONS,
        preamble=LOCALIZATION_PREAMBLE,
        question=LOCALIZATION_QUESTION,
        seed_1=seed_mapping_1,
        seed_2=seed_mapping_2,
        true_key=target_interval,
    )

    # Probe Public-History Replay Observer with the exact same 2 mappings
    observer_eval = evaluate_mapping_equivariance(
        adapter=adapter,
        snapshot=cur_observer,
        options=TEMPORAL_LOCALIZATION_OPTIONS,
        preamble=LOCALIZATION_PREAMBLE,
        question=LOCALIZATION_QUESTION,
        seed_1=seed_mapping_1,
        seed_2=seed_mapping_2,
        true_key=target_interval,
    )

    t_acc1 = float(target_eval["m1_acc"])
    obs_acc1 = float(observer_eval["m1_acc"])
    pai_1 = t_acc1 - obs_acc1

    t_acc2 = float(target_eval["m2_acc"])
    obs_acc2 = float(observer_eval["m2_acc"])
    pai_2 = t_acc2 - obs_acc2

    return LocalizationTrialResult(
        pair_id=pair.pair_id,
        family_id=pair.family_id,
        condition=condition,
        target_interval=target_interval,
        channels=channels,
        target_m1_pred=target_eval["m1_predicted_key"],
        target_m2_pred=target_eval["m2_predicted_key"],
        target_m1_acc=target_eval["m1_acc"],
        target_m2_acc=target_eval["m2_acc"],
        target_semantic_agreement=target_eval["semantic_agreement"],
        target_js_div=target_eval["js_divergence"],
        target_m1_probs=target_eval["m1_semantic_probs"],
        target_m2_probs=target_eval["m2_semantic_probs"],
        observer_m1_pred=observer_eval["m1_predicted_key"],
        observer_m2_pred=observer_eval["m2_predicted_key"],
        observer_m1_acc=observer_eval["m1_acc"],
        observer_m2_acc=observer_eval["m2_acc"],
        observer_js_div=observer_eval["js_divergence"],
        observer_m1_probs=observer_eval["m1_semantic_probs"],
        observer_m2_probs=observer_eval["m2_semantic_probs"],
        pai_m1=pai_1,
        pai_m2=pai_2,
    )
