from __future__ import annotations

import unittest

from rwkv_agent.session_text import SessionTextBuffer


class SessionTextBufferTests(unittest.TestCase):
    def test_text_is_session_scoped_and_replaced(self) -> None:
        buffer = SessionTextBuffer(max_sessions=4, max_chars=100)
        first = buffer.put("alpha", "first document")
        self.assertEqual(buffer.get("alpha"), first)
        self.assertIsNone(buffer.get("beta"))

        second = buffer.put("alpha", "replacement")
        self.assertEqual(buffer.get("alpha"), second)
        self.assertNotEqual(first.sha256, second.sha256)
        self.assertEqual(buffer.health()["sessions"], 1)

    def test_lru_evicts_oldest_session(self) -> None:
        buffer = SessionTextBuffer(max_sessions=2, max_chars=100)
        buffer.put("alpha", "A")
        buffer.put("beta", "B")
        self.assertIsNotNone(buffer.get("alpha"))
        buffer.put("gamma", "C")
        self.assertIsNone(buffer.get("beta"))
        self.assertIsNotNone(buffer.get("alpha"))
        self.assertIsNotNone(buffer.get("gamma"))

    def test_clear_and_close_remove_transient_text(self) -> None:
        buffer = SessionTextBuffer(max_sessions=2, max_chars=100)
        buffer.put("alpha", "temporary")
        self.assertTrue(buffer.clear("alpha"))
        self.assertFalse(buffer.clear("alpha"))
        buffer.put("beta", "temporary")
        buffer.close()
        self.assertEqual(buffer.health()["sessions"], 0)

    def test_empty_session_empty_text_and_oversize_are_rejected(self) -> None:
        buffer = SessionTextBuffer(max_chars=4)
        with self.assertRaises(ValueError):
            buffer.put("", "text")
        with self.assertRaises(ValueError):
            buffer.put("alpha", " ")
        with self.assertRaises(ValueError):
            buffer.put("alpha", "12345")


if __name__ == "__main__":
    unittest.main()
