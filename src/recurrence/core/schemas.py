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

TARGET_4AFC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
        },
    },
    "required": ["answer"],
    "additionalProperties": False,
}

TARGET_3AFC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["A", "B", "C"],
        },
    },
    "required": ["answer"],
    "additionalProperties": False,
}

TARGET_2AFC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["A", "B"],
        },
    },
    "required": ["answer"],
    "additionalProperties": False,
}

TARGET_2AFC_CONFIDENCE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["A", "B"],
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

TARGET_2AFC_LIKERT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["A", "B"],
        },
        "confidence": {
            "type": "integer",
            "minimum": 1,
            "maximum": 4,
        },
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}

TARGET_ANSWER_ONLY_SCHEMA: Dict[str, Any] = TARGET_4AFC_SCHEMA

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

STATE_UPDATE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "working_memory": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "active", "completed", "suspended"],
                    },
                },
                "required": ["goal_id", "description", "status"],
                "additionalProperties": False,
            },
        },
        "source_ledger": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": ["environment", "self", "experimenter"],
            },
        },
        "unresolved_items": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["working_memory", "goals", "source_ledger", "unresolved_items"],
    "additionalProperties": False,
}

