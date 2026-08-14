"""Smoke test for JSON Schema structured outputs with OllamaBackend across all 6 primary evaluator conditions."""

import json
from recurrence.backends.ollama import OllamaBackend
from recurrence.tasks.kv_retrieval import KVRetrievalTask
from recurrence.observers.visible import (
    VisibleAnswerOnlyObserver,
    VisibleFullTranscriptObserver,
)
from recurrence.observers.reconstruction import ReconstructionObserver
from recurrence.observers.ablated import EqualComputeReviewObserver
from recurrence.core.schemas import (
    TARGET_FORCED_CHOICE_SCHEMA,
    PROBABILITY_ONLY_SCHEMA,
    RECONSTRUCTION_DISTRIBUTION_SCHEMA,
)


def smoke_test_structured_schemas(items_count: int = 5) -> None:
    print(f"Starting Structured JSON Schema Smoke Test with {items_count} items per condition...")
    backend = OllamaBackend(model_name="qwen2.5:3b", seed=42)

    task = KVRetrievalTask(mode="forced_choice", ask_confidence=True, confidence_format="probability")
    raw = task.generate_raw_pairs(count=items_count, distractor_count=5, identifier_type="semantic", seed=42)
    items = task.generate_items_from_raw(raw, seed=42)

    obs_self_rev = EqualComputeReviewObserver(backend=backend, framing="self")
    obs_oth_rev = EqualComputeReviewObserver(backend=backend, framing="other")
    obs_vis_ans = VisibleAnswerOnlyObserver(backend=backend)
    obs_vis_full = VisibleFullTranscriptObserver(backend=backend)
    obs_recon = ReconstructionObserver(backend=backend)

    results = {
        "target": {"total": 0, "schema_valid": 0, "ans_valid": 0, "prob_valid": 0},
        "self_review": {"total": 0, "valid_prob": 0},
        "other_review": {"total": 0, "valid_prob": 0},
        "visible_answer_only": {"total": 0, "valid_prob": 0},
        "visible_full_transcript": {"total": 0, "valid_prob": 0},
        "reconstruction": {"total": 0, "complete_dist": 0, "valid_prob": 0},
    }

    for i, item in enumerate(items):
        print(f"\n--- Item {i+1}/{items_count} ({item.item_id}) ---")
        
        # 1. Target with TARGET_FORCED_CHOICE_SCHEMA
        messages = [{"role": "user", "content": item.prompt}]
        target_resp, meta = backend.chat(messages=messages, temperature=0.0, seed=42, format=TARGET_FORCED_CHOICE_SCHEMA)
        score = task.score_response(item, target_resp)
        print(f"Target raw: {target_resp!r}")
        print(f"Target score: correct={score['correct']}, ans={score['parsed_answer']}, prob={score['probability']}, schema_valid={score['schema_valid']}")

        results["target"]["total"] += 1
        if score["schema_valid"]:
            results["target"]["schema_valid"] += 1
        if score["answer_parse_valid"]:
            results["target"]["ans_valid"] += 1
        if score["probability_parse_valid"]:
            results["target"]["prob_valid"] += 1

        # 2. Observers
        eval_self = obs_self_rev.evaluate(item.prompt, target_resp, item_metadata=item.metadata, seed=42)
        results["self_review"]["total"] += 1
        if eval_self.predicted_probability is not None:
            results["self_review"]["valid_prob"] += 1
        print(f"Self-Review: {eval_self.predicted_probability} (raw: {eval_self.raw_response!r})")

        eval_oth = obs_oth_rev.evaluate(item.prompt, target_resp, item_metadata=item.metadata, seed=42)
        results["other_review"]["total"] += 1
        if eval_oth.predicted_probability is not None:
            results["other_review"]["valid_prob"] += 1
        print(f"Other-Review: {eval_oth.predicted_probability} (raw: {eval_oth.raw_response!r})")

        eval_vis_ans = obs_vis_ans.evaluate(item.prompt, target_resp, item_metadata=item.metadata, seed=42)
        results["visible_answer_only"]["total"] += 1
        if eval_vis_ans.predicted_probability is not None:
            results["visible_answer_only"]["valid_prob"] += 1
        print(f"Visible Answer-Only: {eval_vis_ans.predicted_probability} (raw: {eval_vis_ans.raw_response!r})")

        eval_vis_full = obs_vis_full.evaluate(item.prompt, target_resp, item_metadata=item.metadata, seed=42)
        results["visible_full_transcript"]["total"] += 1
        if eval_vis_full.predicted_probability is not None:
            results["visible_full_transcript"]["valid_prob"] += 1
        print(f"Visible Full-Transcript: {eval_vis_full.predicted_probability} (raw: {eval_vis_full.raw_response!r})")

        eval_recon = obs_recon.evaluate(item.prompt, target_resp, item_metadata=item.metadata, seed=42)
        results["reconstruction"]["total"] += 1
        if eval_recon.metadata.get("distribution_complete"):
            results["reconstruction"]["complete_dist"] += 1
        if eval_recon.predicted_probability is not None:
            results["reconstruction"]["valid_prob"] += 1
        print(f"Reconstruction: {eval_recon.predicted_probability} (dist: {eval_recon.metadata.get('distribution_raw')}, raw: {eval_recon.raw_response!r})")

    print("\n==========================================")
    print("SMOKE TEST COMPLIANCE SUMMARY:")
    print("==========================================")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    smoke_test_structured_schemas(items_count=5)
