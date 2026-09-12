import unittest
from unittest.mock import patch

from app.llm_polish import LLMPolisher


class LLMStreamingTests(unittest.TestCase):
    def test_openai_stream_calls_on_update_incrementally(self):
        updates = []
        polisher = LLMPolisher(
            {
                "enabled": True,
                "schema": "openai",
                "endpoint": "http://localhost/v1/chat/completions",
                "model": "demo",
            }
        )
        events = [
            {"choices": [{"delta": {"content": "你好"}}]},
            {"choices": [{"delta": {"content": "世界"}}]},
        ]
        with patch.object(polisher, "_http_post_sse", return_value=iter(events)):
            result = polisher.polish_tail_stream("稳定前缀", "你好世界", on_update=updates.append)

        self.assertEqual(result, "你好世界")
        self.assertEqual(updates, ["你好", "你好世界"])

    def test_stream_failure_returns_none(self):
        polisher = LLMPolisher(
            {
                "enabled": True,
                "schema": "anthropic",
                "endpoint": "http://localhost/v1/messages",
                "model": "demo",
            }
        )
        with patch.object(polisher, "_http_post_sse", side_effect=TimeoutError("boom")):
            result = polisher.polish_tail_stream("稳定前缀", "尾巴")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
