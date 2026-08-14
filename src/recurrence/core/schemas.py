"""Canonical JSON Schema definitions for structured model elicitation in Level-0 and memory baselines."""

from typing import Any, Dict


TARGET_FORCED_CHOICE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
        },
        "probability": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
    },
    "required": ["answer", "probability"],
    "additionalProperties": False,
}

PROBABILITY_ONLY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "probability": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
    },
    "required": ["probability"],
    "additionalProperties": False,
}

RECONSTRUCTION_DISTRIBUTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "A": {"type": "integer", "minimum": 0, "maximum": 100},
        "B": {"type": "integer", "minimum": 0, "maximum": 100},
        "C": {"type": "integer", "minimum": 0, "maximum": 100},
        "D": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["A", "B", "C", "D"],
    "additionalProperties": False,
}
