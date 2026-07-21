import unittest

from rwkv_search.g1i_tool_call import (
    evaluate_web_search_tool_call,
    important_entities,
    render_p4_prompt,
)


class G1IToolCallRuntimeTest(unittest.TestCase):
    def test_accepts_selected_protocol_with_private_think(self):
        result = evaluate_web_search_tool_call(
            '<think>need current data</think>\n'
            '<tool_call>{"name":"web_search","arguments":{"query":"Python latest"}}</tool_call>'
        )
        self.assertTrue(result["strict_success"])
        self.assertEqual(result["query"], "Python latest")

    def test_rejects_extra_visible_text_or_other_tool_format(self):
        extra = evaluate_web_search_tool_call(
            'searching\n<tool_call>{"name":"web_search","arguments":{"query":"RWKV"}}</tool_call>'
        )
        other = evaluate_web_search_tool_call(
            '<tool_code>x</tool_code><tool_call>{"name":"web_search","arguments":{"query":"RWKV"}}</tool_call>'
        )
        self.assertFalse(extra["strict_success"])
        self.assertFalse(other["strict_success"])

    def test_rejects_non_flat_schema(self):
        result = evaluate_web_search_tool_call(
            '<tool_call>{"type":"function","function":{"name":"web_search"}}</tool_call>'
        )
        self.assertFalse(result["parse_success"])

    def test_prompt_and_entity_extraction_are_runtime_owned(self):
        self.assertIn("QUERY_STRING", render_p4_prompt("RWKV 7.2B latest"))
        self.assertEqual(important_entities("RWKV 7.2B latest"), ("RWKV", "7.2B"))


if __name__ == "__main__":
    unittest.main()
