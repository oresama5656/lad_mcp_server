import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from lad_mcp_server.serena_bridge import SerenaContext, SerenaLimits, SerenaToolError


class TestSerenaBridge(unittest.TestCase):
    def test_detect_requires_serena_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=1000,
                    max_total_chars=2000,
                    tool_timeout_seconds=1,
                ),
            )
            self.assertIsNone(ctx)

    def test_list_memories_empty_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=1000,
                    max_total_chars=2000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool("list_memories", "{}")
            self.assertIn("memories", out)

    def test_read_memory_requires_name(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena" / "memories").mkdir(parents=True)
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=1000,
                    max_total_chars=2000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            with self.assertRaises(SerenaToolError):
                ctx.call_tool("read_memory", "{\"name\": \"\"}")

    def test_read_file_requires_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=1000,
                    max_total_chars=2000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool("read_file", "{\"path\": \"a.txt\", \"head\": 1}")
            self.assertIn("hello", out)

    def test_search_for_pattern_falls_back_when_rg_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "src").mkdir()
            (repo / "src" / "a.txt").write_text("hello world\nbye\n", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=2000,
                    max_total_chars=4000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")

            with patch("subprocess.run", side_effect=FileNotFoundError()):
                out = ctx.call_tool("search_for_pattern", "{\"pattern\": \"hello\", \"path\": \"src\"}")
            self.assertIn("matches", out)
            self.assertIn("src/a.txt", out)

    def test_read_file_rejects_large_file_without_head_tail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            big = repo / "big.txt"
            big.write_bytes(b"a" * 1_000_001)
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=2000,
                    max_total_chars=4000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")

            with self.assertRaises(SerenaToolError):
                ctx.call_tool("read_file", "{\"path\": \"big.txt\"}")

    def test_read_file_allows_large_file_with_head(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            big = repo / "big.txt"
            big.write_text("first\nsecond\n" + ("x" * 1_000_001), encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=2000,
                    max_total_chars=4000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")

            out = ctx.call_tool("read_file", "{\"path\": \"big.txt\", \"head\": 1}")
            self.assertIn("first", out)

    def test_tool_output_includes_status_and_budget_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=2000,
                    max_total_chars=4000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool("read_file", "{\"path\": \"a.txt\", \"head\": 1}")
            payload = json.loads(out)

            self.assertEqual(payload["tool_status"], "ok")
            self.assertEqual(payload["tool_name"], "read_file")
            self.assertIn("tool_budget", payload)
            self.assertIn("tool_result_json", payload)
            self.assertIn("hello", payload["tool_result_json"])
            self.assertIn("max_total_chars", payload["tool_budget"])
            self.assertIn("used_chars", payload["tool_budget"])
            self.assertIn("remaining_chars", payload["tool_budget"])
            self.assertIn("emitted_chars_this_call", payload["tool_budget"])

    def test_tool_output_marks_truncated_when_capped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "a.txt").write_text("x" * 500, encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=40,
                    max_total_chars=4000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool("read_file", "{\"path\": \"a.txt\"}")
            payload = json.loads(out)

            self.assertEqual(payload["tool_status"], "truncated")
            self.assertIn("note", payload)
            self.assertLessEqual(payload["tool_budget"]["emitted_chars_this_call"], 40)
            self.assertLessEqual(len(payload["tool_result_json"]), 40)

    def test_budget_exhausted_returns_structured_payload_not_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "a.txt").write_text("x" * 500, encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=200,
                    max_total_chars=60,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            # First call burns the remaining budget.
            ctx.call_tool("read_file", "{\"path\": \"a.txt\"}")
            out = ctx.call_tool("list_memories", "{}")

            self.assertNotEqual(out, "")
            payload = json.loads(out)
            self.assertEqual(payload["tool_status"], "budget_exhausted")
            self.assertIn("error", payload)
            self.assertIn("hint", payload)
            self.assertEqual(payload["tool_budget"]["remaining_chars"], 0)


if __name__ == "__main__":
    unittest.main()
