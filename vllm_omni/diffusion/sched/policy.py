# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Protocol

from vllm_omni.diffusion.sched.interface import DiffusionExecutionState, DiffusionRequestState


class RequestSelectionPolicy(Protocol):
    """Order waiting requests for one scheduling turn."""

    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        """Return waiting scheduler request ids ordered by selection priority."""


class FCFSSelectionPolicy:
    """Preserve arrival order from the waiting deque."""

    def order_waiting(
        self,
        waiting: list[str],
        request_states: dict[str, DiffusionRequestState],
        execution_states: dict[str, DiffusionExecutionState],
    ) -> list[str]:
        del request_states, execution_states
        return list(waiting)


def build_request_selection_policy(name: str) -> RequestSelectionPolicy:
    if name == "fcfs":
        return FCFSSelectionPolicy()
    raise NotImplementedError(f"Unsupported diffusion step-level selection policy: {name!r}")
