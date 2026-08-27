"""RWKV Agent controller, tools and G1I serving runtime."""

from __future__ import annotations

from typing import Any

__all__ = ["AgentController"]


def __getattr__(name: str) -> Any:
    """Keep lightweight Worker utilities independent of Agent dependencies."""

    if name == "AgentController":
        from .controller import AgentController

        return AgentController
    raise AttributeError(name)
