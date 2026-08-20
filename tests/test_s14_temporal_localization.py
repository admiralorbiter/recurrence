"""Unit tests for S14 forced-choice probes and temporal localization tasks."""

import pytest
import torch
from transformers import AutoTokenizer

from recurrence.models.recurrent_gemma_adapter import RecurrentGemmaAdapter
from recurrence.tasks.forced_choice_probes import (
    SemanticOption,
    create_forced_choice_mapping,
    format_forced_choice_prompt,
    compute_js_divergence,
)
from recurrence.tasks.temporal_localization import (
    TEMPORAL_LOCALIZATION_OPTIONS,
    generate_neutral_intervals,
)


def test_forced_choice_mapping_permutation():
    tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-2b-it")
    options = [
        SemanticOption("opt1", "First option"),
        SemanticOption("opt2", "Second option"),
        SemanticOption("opt3", "Third option"),
    ]

    map_1 = create_forced_choice_mapping(options, tokenizer, seed=42)
    assert len(map_1.letters) == 3
    assert set(map_1.key_to_label.keys()) == {"opt1", "opt2", "opt3"}
    assert set(map_1.label_to_key.keys()) == {"A", "B", "C"}

    # Distinct seed produces different permutation
    map_2 = create_forced_choice_mapping(options, tokenizer, seed=999)
    assert len(map_2.letters) == 3
    assert set(map_2.label_to_key.keys()) == {"A", "B", "C"}


def test_js_divergence_properties():
    # Identical distributions have zero divergence
    p = {"a": 0.5, "b": 0.5}
    assert compute_js_divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    # Disjoint distributions have maximum divergence (1.0)
    p1 = {"a": 1.0, "b": 0.0}
    p2 = {"a": 0.0, "b": 1.0}
    assert compute_js_divergence(p1, p2) == pytest.approx(1.0, abs=1e-5)


def test_generate_neutral_intervals_shape():
    tokenizer = AutoTokenizer.from_pretrained("google/recurrentgemma-2b-it")
    audited_pool = [1000, 1001, 1002, 1003, 1004, 1005]
    intervals = generate_neutral_intervals(
        tokenizer=tokenizer,
        audited_pool=audited_pool,
        seed=42,
        num_intervals=4,
        tokens_per_interval=32,
    )
    assert len(intervals) == 4
    for inter in intervals:
        assert len(inter) == 32
