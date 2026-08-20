"""Lineage tracking, cloning, snapshotting, and branching for developmental organisms."""

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional


class LineageTracker:
    """Tracks historical lineage, forks, and event hashes for developmental organisms."""

    def __init__(self, lineage_id: str, parent_id: Optional[str] = None, fork_step: int = 0):
        self.lineage_id = lineage_id
        self.parent_id = parent_id
        self.fork_step = fork_step
        self.event_log: List[Dict[str, Any]] = []
        self._hasher = hashlib.sha256()

    def record_step(
        self,
        step_idx: int,
        observation: Any = None,
        action: Any = None,
        reward: Any = None,
        obs: Any = None,
        act: Any = None,
        rew: Any = None
    ) -> str:
        final_obs = obs if obs is not None else observation
        final_act = act if act is not None else action
        final_rew = rew if rew is not None else reward

        event_dict = {
            "step": step_idx,
            "obs": str(final_obs),
            "act": str(final_act),
            "rew": str(final_rew),
        }
        self.event_log.append(event_dict)
        event_bytes = json.dumps(event_dict, sort_keys=True).encode("utf-8")
        self._hasher.update(event_bytes)
        return self._hasher.hexdigest()

    @property
    def current_event_hash(self) -> str:
        return self._hasher.hexdigest()

    def fork(self, new_lineage_id: str, fork_step: int) -> "LineageTracker":
        forked = LineageTracker(lineage_id=new_lineage_id, parent_id=self.lineage_id, fork_step=fork_step)
        forked.event_log = copy.deepcopy(self.event_log[:fork_step])
        # Rebuild hasher up to fork step
        forked._hasher = hashlib.sha256()
        for ev in forked.event_log:
            ev_bytes = json.dumps(ev, sort_keys=True).encode("utf-8")
            forked._hasher.update(ev_bytes)
        return forked
