"""Multi-stage episodic memory benchmark battery for Experiment E03 (Horizon 1)."""

import json
import random
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from recurrence.core.schemas import TARGET_ANSWER_ONLY_SCHEMA
from recurrence.memory.schemas import (
    EventSource,
    GoalState,
    MemoryEvent,
    MemoryFormat,
    StructuredSelfState,
)
from recurrence.memory.adapters import get_memory_adapter
from recurrence.tasks.base import BaseTask, TaskItem


class MemoryProbeItem(TaskItem):
    """An individual evaluation item within the episodic memory battery."""
    probe_type: Literal["delayed_kv", "source_attribution", "goal_resumption"]
    target_key: str
    target_value: str
    target_source: Optional[str] = None
    target_goal_id: Optional[str] = None
    position_stratum: Literal["early", "middle", "late"] = "middle"
    distractor_count: int = 5
    memory_format: MemoryFormat = MemoryFormat.FRESH
    raw_events: List[MemoryEvent] = Field(default_factory=list)
    structured_state: Optional[StructuredSelfState] = None
    cached_summary: Optional[str] = None


class EpisodeData(BaseModel):
    """A complete synthetic episode with events, state, and probes."""
    episode_id: str
    events: List[MemoryEvent]
    structured_state: StructuredSelfState
    kv_targets: Dict[str, Dict[str, Any]]
    goals: List[GoalState]


class MemoryBatteryTask(BaseTask):
    """Multi-stage memory battery generating controlled episodic streams and probes."""

    def __init__(
        self,
        identifier_type: Literal["semantic", "opaque"] = "semantic",
        mode: Literal["forced_choice"] = "forced_choice",
        ask_confidence: bool = False,
        default_memory_format: MemoryFormat = MemoryFormat.TRANSCRIPT,
    ):
        super().__init__(
            name="e03_memory_battery",
            description="Multi-stage episodic memory benchmark battery testing delayed retrieval, source memory, and goal resumption across memory formats."
        )
        self.identifier_type = identifier_type
        self.mode = mode
        self.ask_confidence = ask_confidence
        self.default_memory_format = default_memory_format

    def generate_items(
        self,
        count: int = 10,
        seed: int = 42,
        memory_format: Optional[MemoryFormat] = None,
        cached_summaries: Optional[Dict[str, str]] = None,
    ) -> List[TaskItem]:
        """Generate benchmark task items across synthetic episodes for the specified memory format."""
        fmt = memory_format or self.default_memory_format
        episodes = [
            self.generate_episode(episode_idx=i, seed=seed)
            for i in range(count)
        ]
        return list(self.generate_probe_items(episodes, memory_format=fmt, cached_summaries=cached_summaries, seed=seed))

    def generate_episode(
        self,
        episode_idx: int,
        target_kv_count: int = 3,
        distractor_count: int = 6,
        seed: int = 42,
    ) -> EpisodeData:
        """Generate a single episodic stream with interleaved targets, sources, distractors, and goals."""
        rng = random.Random(seed + episode_idx * 1000)

        # 1. Semantic word pools for realistic entity binding
        nouns = [
            "falcon", "canyon", "river", "glacier", "prism", "tempest", "harbor", "citadel",
            "volcano", "compass", "meadow", "cascade", "spire", "cavern", "lagoon", "monolith"
        ]
        adjectives = [
            "obsidian", "velvet", "golden", "emerald", "crimson", "sapphire", "amber", "silver",
            "celestial", "shadow", "radiant", "frozen", "solar", "lunar", "mystic", "ancient"
        ]

        def make_pair(idx: int) -> Tuple[str, str]:
            if self.identifier_type == "semantic":
                adj = adjectives[(episode_idx * 7 + idx) % len(adjectives)]
                noun = nouns[(episode_idx * 11 + idx * 3) % len(nouns)]
                k = f"key_{adj}_{noun}"
                v_adj = adjectives[(episode_idx * 3 + idx * 5 + 1) % len(adjectives)]
                v_noun = nouns[(episode_idx * 5 + idx * 2 + 1) % len(nouns)]
                v = f"val_{v_adj}_{v_noun}"
                return k, v
            else:
                return f"key_0x{rng.randint(0x1000, 0xffff):04x}", f"val_0x{rng.randint(0x1000, 0xffff):04x}"

        # 2. Build target key-value pairs with source assignments
        sources_pool = [EventSource.ENVIRONMENT, EventSource.SELF, EventSource.EXPERIMENTER]
        kv_targets: Dict[str, Dict[str, Any]] = {}
        strata: List[Literal["early", "middle", "late"]] = ["early", "middle", "late"]

        for i in range(target_kv_count):
            k, v = make_pair(i)
            src = sources_pool[i % len(sources_pool)]
            pos = strata[i % len(strata)]
            kv_targets[k] = {
                "key": k,
                "value": v,
                "source": src.value,
                "stratum": pos,
            }

        # 3. Interleave target events and distractor events into a chronological timeline
        total_steps = target_kv_count + distractor_count + 2  # plus goal events
        events: List[MemoryEvent] = []
        step = 1

        # Target placement by stratum
        target_keys = list(kv_targets.keys())
        early_target = target_keys[0] if len(target_keys) > 0 else None
        mid_target = target_keys[1] if len(target_keys) > 1 else None
        late_target = target_keys[2] if len(target_keys) > 2 else None

        # Helper to format event content based on source
        def format_event_content(k: str, v: str, src: EventSource) -> str:
            if src == EventSource.ENVIRONMENT:
                return f"Sensor telemetry observed that {k} is mapped to {v}."
            elif src == EventSource.SELF:
                return f"I executed an internal computation and asserted {k} = {v}."
            else:
                return f"Experimenter instruction declared that {k} must resolve to {v}."

        # Early Goal Assertion
        active_goal = GoalState(
            goal_id=f"goal_ep{episode_idx:02d}_alpha",
            description="Calibrate sensor array parameters",
            status="completed",
            created_at_step=step,
            updated_at_step=step + 4,
        )
        suspended_goal = GoalState(
            goal_id=f"goal_ep{episode_idx:02d}_beta",
            description="Process background telemetry archive",
            status="suspended",
            created_at_step=step + 2,
            updated_at_step=step + 2,
        )

        # Step 1: Goal event
        events.append(MemoryEvent(
            event_id=f"ev_ep{episode_idx:02d}_{step:02d}",
            step_index=step,
            source=EventSource.EXPERIMENTER,
            event_type="goal_assertion",
            content=f"Primary objective assigned: {active_goal.description}.",
            metadata={"goal_id": active_goal.goal_id}
        ))
        step += 1

        # Step 2: Early target (if any)
        if early_target:
            t = kv_targets[early_target]
            events.append(MemoryEvent(
                event_id=f"ev_ep{episode_idx:02d}_{step:02d}",
                step_index=step,
                source=EventSource(t["source"]),
                event_type="binding_assertion",
                content=format_event_content(t["key"], t["value"], EventSource(t["source"])),
                key_bindings={t["key"]: t["value"]},
            ))
            step += 1

        # Distractor batch 1
        for d_i in range(distractor_count // 3):
            dk, dv = make_pair(100 + d_i)
            events.append(MemoryEvent(
                event_id=f"ev_ep{episode_idx:02d}_{step:02d}",
                step_index=step,
                source=EventSource.ENVIRONMENT,
                event_type="distractor",
                content=f"Routine ambient observation: {dk} is currently {dv}.",
                key_bindings={dk: dv},
            ))
            step += 1

        # Suspended goal event
        events.append(MemoryEvent(
            event_id=f"ev_ep{episode_idx:02d}_{step:02d}",
            step_index=step,
            source=EventSource.EXPERIMENTER,
            event_type="goal_update",
            content=f"Secondary task suspended due to priority interruption: {suspended_goal.description}.",
            metadata={"goal_id": suspended_goal.goal_id, "status": "suspended"}
        ))
        step += 1

        # Middle target
        if mid_target:
            t = kv_targets[mid_target]
            events.append(MemoryEvent(
                event_id=f"ev_ep{episode_idx:02d}_{step:02d}",
                step_index=step,
                source=EventSource(t["source"]),
                event_type="binding_assertion",
                content=format_event_content(t["key"], t["value"], EventSource(t["source"])),
                key_bindings={t["key"]: t["value"]},
            ))
            step += 1

        # Distractor batch 2
        for d_i in range(distractor_count // 3, 2 * (distractor_count // 3)):
            dk, dv = make_pair(200 + d_i)
            events.append(MemoryEvent(
                event_id=f"ev_ep{episode_idx:02d}_{step:02d}",
                step_index=step,
                source=EventSource.SELF,
                event_type="distractor",
                content=f"Local verification confirmed secondary status: {dk} -> {dv}.",
                key_bindings={dk: dv},
            ))
            step += 1

        # Late target
        if late_target:
            t = kv_targets[late_target]
            events.append(MemoryEvent(
                event_id=f"ev_ep{episode_idx:02d}_{step:02d}",
                step_index=step,
                source=EventSource(t["source"]),
                event_type="binding_assertion",
                content=format_event_content(t["key"], t["value"], EventSource(t["source"])),
                key_bindings={t["key"]: t["value"]},
            ))
            step += 1

        # Remaining distractors
        for d_i in range(2 * (distractor_count // 3), distractor_count):
            dk, dv = make_pair(300 + d_i)
            events.append(MemoryEvent(
                event_id=f"ev_ep{episode_idx:02d}_{step:02d}",
                step_index=step,
                source=EventSource.ENVIRONMENT,
                event_type="distractor",
                content=f"System pulse checkpoint: {dk} holds value {dv}.",
                key_bindings={dk: dv},
            ))
            step += 1

        # 4. Construct Structured Self-State
        working_bindings = {t["key"]: t["value"] for t in kv_targets.values()}
        source_ledger = {t["key"]: t["source"] for t in kv_targets.values()}
        structured_state = StructuredSelfState(
            working_memory=working_bindings,
            goals=[active_goal, suspended_goal],
            source_ledger=source_ledger,
            unresolved_items=[suspended_goal.goal_id],
            last_updated_step=len(events),
        )

        return EpisodeData(
            episode_id=f"ep_{episode_idx:03d}",
            events=events,
            structured_state=structured_state,
            kv_targets=kv_targets,
            goals=[active_goal, suspended_goal],
        )

    def generate_probe_items(
        self,
        episodes: List[EpisodeData],
        memory_format: MemoryFormat,
        cached_summaries: Optional[Dict[str, str]] = None,
        seed: int = 42,
    ) -> List[MemoryProbeItem]:
        """Generate test probe items for each episode across all three memory probe tasks."""
        rng = random.Random(seed)
        adapter = get_memory_adapter(memory_format)
        items: List[MemoryProbeItem] = []

        cached_summaries = cached_summaries or {}

        for ep_idx, ep in enumerate(episodes):
            context_text = adapter.build_context_prompt(
                events=ep.events,
                structured_state=ep.structured_state,
                cached_summary=cached_summaries.get(ep.episode_id),
            )

            # Probe Type 1: Delayed KV Retrieval (For each target)
            for k, target_info in ep.kv_targets.items():
                correct_val = target_info["value"]
                # 3 distractors
                distractors = [
                    f"val_distractor_{ep_idx}_{rng.randint(100, 999)}"
                    for _ in range(3)
                ]
                options = [correct_val] + distractors
                rng.shuffle(options)
                correct_letter = chr(65 + options.index(correct_val))
                option_map = {chr(65 + i): opt for i, opt in enumerate(options)}

                options_str = "\n".join([f"({letter}) {opt}" for letter, opt in option_map.items()])

                probe_prompt = (
                    f"{context_text}"
                    f"Question: What was the value associated with '{k}' during the episode?\n\n"
                    f"Options:\n{options_str}\n\n"
                    f"Respond strictly with a JSON object in this format with no other text:\n"
                    f'{{"answer": "<Option letter, e.g. A, B, C, or D>"}}'
                )

                item_id = f"e03_{ep.episode_id}_{k}_{memory_format.value}_delayed_kv"
                items.append(MemoryProbeItem(
                    item_id=item_id,
                    probe_type="delayed_kv",
                    prompt=probe_prompt,
                    ground_truth=correct_letter,
                    target_key=k,
                    target_value=correct_val,
                    target_source=target_info["source"],
                    position_stratum=target_info["stratum"],
                    distractor_count=len(ep.events) - 3,
                    memory_format=memory_format,
                    raw_events=ep.events,
                    structured_state=ep.structured_state,
                    cached_summary=cached_summaries.get(ep.episode_id),
                    metadata={
                        "episode_id": ep.episode_id,
                        "target_key": k,
                        "correct_value": correct_val,
                        "option_map": option_map,
                    }
                ))

            # Probe Type 2: Source Memory Attribution (For each target)
            for k, target_info in ep.kv_targets.items():
                true_src = target_info["source"]
                # 3AFC Source options: environment, self, experimenter
                src_options = ["environment", "self", "experimenter"]
                rng.shuffle(src_options)
                correct_letter = chr(65 + src_options.index(true_src))
                option_map = {chr(65 + i): opt for i, opt in enumerate(src_options)}

                options_str = "\n".join([f"({letter}) {opt.capitalize()}" for letter, opt in option_map.items()])

                probe_prompt = (
                    f"{context_text}"
                    f"Question: What was the origin source of the assertion regarding '{k}'?\n\n"
                    f"Options:\n{options_str}\n\n"
                    f"Respond strictly with a JSON object in this format with no other text:\n"
                    f'{{"answer": "<Option letter, e.g. A, B, or C>"}}'
                )

                item_id = f"e03_{ep.episode_id}_{k}_{memory_format.value}_source_attr"
                items.append(MemoryProbeItem(
                    item_id=item_id,
                    probe_type="source_attribution",
                    prompt=probe_prompt,
                    ground_truth=correct_letter,
                    target_key=k,
                    target_value=target_info["value"],
                    target_source=true_src,
                    position_stratum=target_info["stratum"],
                    distractor_count=len(ep.events) - 3,
                    memory_format=memory_format,
                    raw_events=ep.events,
                    structured_state=ep.structured_state,
                    cached_summary=cached_summaries.get(ep.episode_id),
                    metadata={
                        "episode_id": ep.episode_id,
                        "target_key": k,
                        "true_source": true_src,
                        "option_map": option_map,
                    }
                ))

            # Probe Type 3: Interrupted Goal Resumption
            suspended_goal = [g for g in ep.goals if g.status == "suspended"][0]
            distractor_goals = [
                "Deploy security certificate update",
                "Archive obsolete diagnostic logs",
                "Synchronize clock with reference server",
            ]
            goal_options = [suspended_goal.description] + distractor_goals
            rng.shuffle(goal_options)
            correct_letter = chr(65 + goal_options.index(suspended_goal.description))
            option_map = {chr(65 + i): opt for i, opt in enumerate(goal_options)}
            options_str = "\n".join([f"({letter}) {opt}" for letter, opt in option_map.items()])

            probe_prompt = (
                f"{context_text}"
                f"Question: Which task or goal was interrupted and left suspended/unresolved?\n\n"
                f"Options:\n{options_str}\n\n"
                f"Respond strictly with a JSON object in this format with no other text:\n"
                f'{{"answer": "<Option letter, e.g. A, B, C, or D>"}}'
            )

            item_id = f"e03_{ep.episode_id}_goal_resumption_{memory_format.value}"
            items.append(MemoryProbeItem(
                item_id=item_id,
                probe_type="goal_resumption",
                prompt=probe_prompt,
                ground_truth=correct_letter,
                target_key="suspended_goal",
                target_value=suspended_goal.description,
                target_goal_id=suspended_goal.goal_id,
                memory_format=memory_format,
                raw_events=ep.events,
                structured_state=ep.structured_state,
                cached_summary=cached_summaries.get(ep.episode_id),
                metadata={
                    "episode_id": ep.episode_id,
                    "target_goal_id": suspended_goal.goal_id,
                    "correct_description": suspended_goal.description,
                    "option_map": option_map,
                }
            ))

        return items

    def score_response(self, item: TaskItem, response_text: str) -> Dict[str, Any]:
        """Score the response using strict schema validation and answer extraction."""
        score_dict: Dict[str, Any] = {
            "parsed_answer": None,
            "correct": False,
            "schema_valid": False,
            "answer_parse_valid": False,
            "raw_response": response_text,
        }

        # 1. Parse JSON strictly
        try:
            parsed = json.loads(response_text.strip())
            if isinstance(parsed, dict) and "answer" in parsed:
                ans_str = str(parsed["answer"]).strip().upper()
                if len(ans_str) == 1 and ans_str in "ABCD":
                    score_dict["parsed_answer"] = ans_str
                    score_dict["answer_parse_valid"] = True
                    score_dict["correct"] = (ans_str == item.ground_truth)

                # Strict schema validity check (no unexpected extra keys)
                score_dict["schema_valid"] = (
                    score_dict["answer_parse_valid"] and
                    len(parsed.keys()) == 1
                )
        except Exception:
            # Fallback heuristic parser if JSON parsing failed
            text = response_text.strip()
            for line in text.split("\n"):
                clean = line.strip().upper()
                if clean in ["A", "B", "C", "D"]:
                    score_dict["parsed_answer"] = clean
                    score_dict["answer_parse_valid"] = True
                    score_dict["correct"] = (clean == item.ground_truth)
                    break

        return score_dict
