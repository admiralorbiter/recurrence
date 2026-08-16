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

# Schema for single-pass retrospective state reconstruction (Experiment E05d)
STATE_RECONSTRUCTION_SCHEMA: Dict[str, Any] = STATE_UPDATE_SCHEMA

# Schema for selective null-interval reflection (Experiment E06 / Sprint S07)
# Protected evidence (working_memory, source_ledger) is clamped; the model writes to derived channels
STATE_SELECTIVE_REFLECTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "derived_inferences": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Derived deductions, multi-hop conclusions, or resolved entity relations",
        },
        "unresolved_items": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Items, keys, or hypotheses currently unresolved or in conflict",
        },
        "goal_status_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "active", "completed", "suspended"],
                    },
                },
                "required": ["goal_id", "status"],
                "additionalProperties": False,
            },
            "description": "Updated statuses for existing goals",
        },
    },
    "required": ["derived_inferences", "unresolved_items", "goal_status_updates"],
    "additionalProperties": False,
}

STATE_DELTA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "working_memory_upserts": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Key-value bindings to insert or update in working memory",
        },
        "working_memory_deletions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Keys to remove from working memory",
        },
        "source_upserts": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": ["environment", "self", "experimenter"],
            },
            "description": "Source attribution updates for target entity keys",
        },
        "goal_updates": {
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
            "description": "Goal status transitions or new goals asserted in this tick",
        },
        "unresolved_items_add": {
            "type": "array",
            "items": {"type": "string"},
        },
        "unresolved_items_remove": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "working_memory_upserts",
        "working_memory_deletions",
        "source_upserts",
        "goal_updates",
        "unresolved_items_add",
        "unresolved_items_remove",
    ],
    "additionalProperties": False,
}


def make_2afc_direct_value_schema(val_a: str, val_b: str, ask_confidence: bool = True) -> Dict[str, Any]:
    """Construct dynamic JSON schema restricting answer to exact candidate values."""
    if not ask_confidence:
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": [val_a, val_b],
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": [val_a, val_b],
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
