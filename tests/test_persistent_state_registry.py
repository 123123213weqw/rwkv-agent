from __future__ import annotations

import unittest

from rwkv_agent.persistent_state import (
    PersistentState,
    PersistentStateRegistry,
)


class Scheduler:
    def __init__(self) -> None:
        self.released: list[str] = []

    def release(self, request_id: str) -> None:
        self.released.append(request_id)


class PersistentStateRegistryTests(unittest.TestCase):
    def test_owner_capacity_and_ttl_are_registry_concerns(self) -> None:
        now = [10.0]
        scheduler = Scheduler()
        registry = PersistentStateRegistry(
            scheduler=scheduler,  # type: ignore[arg-type]
            capacity=1,
            ttl_seconds=5,
            clock=lambda: now[0],
        )
        record = PersistentState(
            state_id="state-a",
            owner_id="owner-a",
            parent_state_id=None,
            branch="root",
            created_at=now[0],
            last_used_at=now[0],
        )
        registry.add(record)
        self.assertIs(registry.require("state-a", "owner-a"), record)
        with self.assertRaises(PermissionError):
            registry.require("state-a", "owner-b")
        with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
            registry.ensure_capacity(1)

        now[0] = 15.0
        self.assertEqual(registry.cleanup_expired(), ["state-a"])
        self.assertEqual(scheduler.released, ["state-a"])
        self.assertEqual(registry.allocated, 0)

    def test_explicit_release_removes_only_selected_records(self) -> None:
        scheduler = Scheduler()
        registry = PersistentStateRegistry(
            scheduler=scheduler,  # type: ignore[arg-type]
            capacity=2,
            ttl_seconds=5,
        )
        records = [
            PersistentState(
                state_id=f"state-{suffix}",
                owner_id="owner",
                parent_state_id=None,
                branch=suffix,
                created_at=0,
                last_used_at=0,
            )
            for suffix in ("a", "b")
        ]
        for record in records:
            registry.add(record)
        registry.release_records([records[0]])
        self.assertEqual(scheduler.released, ["state-a"])
        self.assertEqual(registry.allocated, 1)
        self.assertIs(registry.require("state-b", "owner"), records[1])


if __name__ == "__main__":
    unittest.main()
