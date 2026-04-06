import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from lad_mcp_server.serena_bridge import SerenaContext, SerenaLimits, SerenaToolError


class TestSerenaBridge(unittest.TestCase):
    def test_tool_schemas_include_read_baseline_memories(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
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
            schemas = ctx.tool_schemas()
            baseline = next(s for s in schemas if (s.get("function") or {}).get("name") == "read_baseline_memories")
            params = (baseline.get("function") or {}).get("parameters") or {}
            self.assertEqual(params.get("required"), [])
            self.assertEqual(params.get("type"), "object")

    def test_tool_schemas_include_read_file_window(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
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
            schemas = ctx.tool_schemas()
            rf_window = next(s for s in schemas if (s.get("function") or {}).get("name") == "read_file_window")
            desc = ((rf_window.get("function") or {}).get("description") or "").lower()
            params = (rf_window.get("function") or {}).get("parameters") or {}
            self.assertEqual(params.get("required"), ["path", "start_line", "num_lines"])
            props = params.get("properties") or {}
            self.assertIn("path", props)
            self.assertIn("start_line", props)
            self.assertIn("num_lines", props)
            self.assertEqual((props["start_line"] or {}).get("type"), "integer")
            self.assertEqual((props["num_lines"] or {}).get("type"), "integer")
            self.assertEqual((props["start_line"] or {}).get("minimum"), 1)
            self.assertEqual((props["num_lines"] or {}).get("minimum"), 0)
            self.assertIn("json", desc)
            self.assertIn("function", desc)

    def test_tool_schemas_include_search_substring_in_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
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
            schemas = ctx.tool_schemas()
            tool = next(s for s in schemas if (s.get("function") or {}).get("name") == "search_substring_in_file")
            params = (tool.get("function") or {}).get("parameters") or {}
            self.assertEqual(params.get("required"), ["path", "substring"])
            props = params.get("properties") or {}
            self.assertIn("path", props)
            self.assertIn("substring", props)

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

    def test_read_file_small_file_returns_full_content(self) -> None:
        """Files <= 10,000 chars return full content, no tail/file_size keys."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            content = "a" * 10_000  # exactly at threshold
            (repo / "a.txt").write_text(content, encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=20_000,
                    max_total_chars=40_000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool("read_file", "{\"path\": \"a.txt\"}")
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["path"], "a.txt")
            self.assertEqual(result["content"], content)
            self.assertNotIn("tail", result)
            self.assertNotIn("file_size", result)

    def test_read_file_truncates_above_threshold(self) -> None:
        """Files > 10,000 chars are auto-truncated to 1000 head + 1000 tail + file_size."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            content = "A" * 5_000 + "B" * 5_001  # 10,001 chars
            (repo / "a.txt").write_text(content, encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=5_000,
                    max_total_chars=10_000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool("read_file", "{\"path\": \"a.txt\"}")
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["path"], "a.txt")
            self.assertEqual(result["file_size"], 10_001)
            self.assertEqual(len(result["content"]), 1_000)
            self.assertEqual(result["content"], "A" * 1_000)
            self.assertEqual(len(result["tail"]), 1_000)
            self.assertEqual(result["tail"], "B" * 1_000)

    def test_read_file_truncated_content_at_exact_threshold_boundary(self) -> None:
        """File of exactly 10,001 chars: first 1000 and last 1000 overlap by 999."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            content = "x" * 10_001
            (repo / "a.txt").write_text(content, encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=5_000,
                    max_total_chars=10_000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool("read_file", "{\"path\": \"a.txt\"}")
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["file_size"], 10_001)
            # Both head and tail are the same char, so content == tail
            self.assertEqual(result["content"], "x" * 1_000)
            self.assertEqual(result["tail"], "x" * 1_000)

    def test_read_file_large_ignores_head_tail_params(self) -> None:
        """For files > 10,000 chars, head/tail line params are ignored."""
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            lines = [f"line-{i:05d}" for i in range(2_000)]  # ~22,000 chars > 10,000
            content = "\n".join(lines)
            (repo / "a.txt").write_text(content, encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=5_000,
                    max_total_chars=10_000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            # head=3 should be ignored — output is always 1000 head chars + 1000 tail chars
            out = ctx.call_tool("read_file", "{\"path\": \"a.txt\", \"head\": 3}")
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertIn("tail", result)
            self.assertIn("file_size", result)
            self.assertEqual(len(result["content"]), 1_000)
            self.assertEqual(len(result["tail"]), 1_000)

    def test_read_file_window_returns_requested_lines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "a.txt").write_text("l1\nl2\nl3\nl4\n", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=4000,
                    max_total_chars=8000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool(
                "read_file_window",
                "{\"path\": \"a.txt\", \"start_line\": 2, \"num_lines\": 2}",
            )
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["path"], "a.txt")
            self.assertEqual(result["start_line"], 2)
            self.assertEqual(result["num_lines"], 2)
            self.assertEqual(result["content"], "l2\nl3\n")

    def test_read_file_window_validates_start_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "a.txt").write_text("l1\n", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=4000,
                    max_total_chars=8000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            with self.assertRaises(SerenaToolError):
                ctx.call_tool("read_file_window", "{\"path\": \"a.txt\", \"start_line\": 0, \"num_lines\": 1}")
            with self.assertRaises(SerenaToolError):
                ctx.call_tool("read_file_window", "{\"path\": \"a.txt\", \"start_line\": 1, \"num_lines\": -1}")

    def test_read_file_window_start_line_beyond_eof_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "a.txt").write_text("l1\nl2\n", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=4000,
                    max_total_chars=8000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool(
                "read_file_window",
                "{\"path\": \"a.txt\", \"start_line\": 100, \"num_lines\": 5}",
            )
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["content"], "")

    def test_read_file_window_num_lines_zero_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "a.txt").write_text("l1\nl2\n", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=4000,
                    max_total_chars=8000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool(
                "read_file_window",
                "{\"path\": \"a.txt\", \"start_line\": 1, \"num_lines\": 0}",
            )
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["content"], "")

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

    def test_search_substring_in_file_returns_count_and_contexts_under_ten(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            content = "xxTARGETyy\nabcTARGETdef\n"
            (repo / "a.txt").write_text(content, encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=10000,
                    max_total_chars=20000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool(
                "search_substring_in_file",
                "{\"path\": \"a.txt\", \"substring\": \"TARGET\"}",
            )
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])

            self.assertEqual(result["path"], "a.txt")
            self.assertEqual(result["file_size"], len(content))
            self.assertEqual(result["count"], 2)
            self.assertIn("occurrences", result)
            self.assertEqual(len(result["occurrences"]), 2)

            indices = [content.find("TARGET"), content.find("TARGET", content.find("TARGET") + len("TARGET"))]
            self.assertEqual([o["index"] for o in result["occurrences"]], indices)
            self.assertEqual([o["line"] for o in result["occurrences"]], [1, 2])

            for occ, idx in zip(result["occurrences"], indices):
                head = content[max(0, idx - 100) : idx]
                tail_start = idx + len("TARGET")
                tail = content[tail_start : tail_start + 100]
                self.assertEqual(occ["context"], head + "TARGET" + tail)

    def test_search_substring_in_file_uses_non_overlapping_matching(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "a.txt").write_text("aaaaa", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=4000,
                    max_total_chars=8000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool(
                "search_substring_in_file",
                "{\"path\": \"a.txt\", \"substring\": \"aa\"}",
            )
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["count"], 2)
            self.assertEqual(result["file_size"], 5)
            self.assertEqual([o["index"] for o in result["occurrences"]], [0, 2])

    def test_search_substring_in_file_six_to_ten_matches_index_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            content = "ab-" * 10
            (repo / "a.txt").write_text(content, encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=4000,
                    max_total_chars=8000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool(
                "search_substring_in_file",
                "{\"path\": \"a.txt\", \"substring\": \"ab\"}",
            )
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["count"], 10)
            self.assertEqual(result["file_size"], len(content))
            self.assertIn("occurrences", result)
            self.assertEqual(len(result["occurrences"]), 10)
            # First 5 have context, 6-10 have index+line only
            for i, occ in enumerate(result["occurrences"]):
                self.assertIn("index", occ)
                self.assertIn("line", occ)
                if i < 5:
                    self.assertIn("context", occ)
                else:
                    self.assertNotIn("context", occ)

    def test_search_substring_in_file_exactly_thirty_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            content = "xy-" * 30  # 30 occurrences of "xy"
            (repo / "a.txt").write_text(content, encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=10000,
                    max_total_chars=20000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool(
                "search_substring_in_file",
                "{\"path\": \"a.txt\", \"substring\": \"xy\"}",
            )
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["count"], 30)
            self.assertEqual(result["file_size"], len(content))
            self.assertIn("occurrences", result)
            self.assertEqual(len(result["occurrences"]), 30)
            # First 5 have context, rest have index+line only
            for i, occ in enumerate(result["occurrences"]):
                self.assertIn("index", occ)
                self.assertIn("line", occ)
                if i < 5:
                    self.assertIn("context", occ)
                else:
                    self.assertNotIn("context", occ)

    def test_search_substring_in_file_over_thirty_matches(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            content = "ab-" * 50  # 50 occurrences of "ab"
            (repo / "a.txt").write_text(content, encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=10000,
                    max_total_chars=20000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool(
                "search_substring_in_file",
                "{\"path\": \"a.txt\", \"substring\": \"ab\"}",
            )
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["count"], 50)
            self.assertEqual(result["file_size"], len(content))
            self.assertIn("occurrences", result)
            # Occurrences list capped at 30
            self.assertEqual(len(result["occurrences"]), 30)

    def test_search_substring_in_file_zero_matches_omits_occurrences(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=4000,
                    max_total_chars=8000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool(
                "search_substring_in_file",
                "{\"path\": \"a.txt\", \"substring\": \"HELLO\"}",
            )
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["count"], 0)
            self.assertEqual(result["file_size"], len("hello\nworld\n"))
            self.assertNotIn("occurrences", result)

    def test_search_substring_in_file_rejects_empty_substring(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            (repo / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=4000,
                    max_total_chars=8000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            with self.assertRaises(SerenaToolError):
                ctx.call_tool(
                    "search_substring_in_file",
                    "{\"path\": \"a.txt\", \"substring\": \"\"}",
                )

    def test_search_substring_in_file_rejects_large_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            big = repo / "big.txt"
            big.write_bytes(b"a" * 100_000_001)
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=4000,
                    max_total_chars=8000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            with self.assertRaises(SerenaToolError):
                ctx.call_tool(
                    "search_substring_in_file",
                    "{\"path\": \"big.txt\", \"substring\": \"a\"}",
                )

    def test_read_file_rejects_large_file_without_head_tail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            big = repo / "big.txt"
            big.write_bytes(b"a" * 100_000_001)
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

    def test_read_baseline_memories_returns_required_present_loaded_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena" / "memories").mkdir(parents=True)
            (repo / ".serena" / "memories" / "project_overview.md").write_text("overview", encoding="utf-8")
            (repo / ".serena" / "memories" / "research_summary.md").write_text("summary", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=6000,
                    max_total_chars=12000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool("read_baseline_memories", "{}")
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["required"], ["project_overview.md", "research_summary.md"])
            self.assertEqual(result["missing"], [])
            self.assertEqual(result["present"], ["project_overview.md", "research_summary.md"])
            loaded_names = [item.get("name") for item in result["loaded"]]
            self.assertEqual(loaded_names, ["project_overview.md", "research_summary.md"])
            self.assertIn("overview", result["loaded"][0]["content"])
            self.assertIn("summary", result["loaded"][1]["content"])

    def test_read_baseline_memories_reports_missing_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena" / "memories").mkdir(parents=True)
            (repo / ".serena" / "memories" / "project_overview.md").write_text("overview", encoding="utf-8")
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=6000,
                    max_total_chars=12000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool("read_baseline_memories", "{}")
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertEqual(result["required"], ["project_overview.md", "research_summary.md"])
            self.assertEqual(result["present"], ["project_overview.md"])
            self.assertEqual(result["missing"], ["research_summary.md"])
            loaded_names = [item.get("name") for item in result["loaded"]]
            self.assertEqual(loaded_names, ["project_overview.md"])

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

    def test_read_file_large_head_warns_with_window_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            big = repo / "big.txt"
            # > 100 MB with many lines (100 bytes per line, ~110 MB total).
            line = b"x" * 99 + b"\n"
            chunk = line * 100_000
            with big.open("wb") as f:
                for _ in range(11):
                    f.write(chunk)
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=60000,
                    max_total_chars=120000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool("read_file", "{\"path\": \"big.txt\", \"head\": 400}")
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertIn("warning", result)
            self.assertIn("search_for_pattern", result["warning"])
            self.assertIn("read_file_window", result["warning"])

    def test_read_file_large_head_below_threshold_does_not_warn(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
            big = repo / "big.txt"
            line = b"x" * 99 + b"\n"
            chunk = line * 100_000
            with big.open("wb") as f:
                for _ in range(11):
                    f.write(chunk)
            ctx = SerenaContext.detect(
                repo,
                SerenaLimits(
                    max_dir_entries=10,
                    max_search_results=10,
                    max_tool_result_chars=60000,
                    max_total_chars=120000,
                    tool_timeout_seconds=1,
                ),
            )
            assert ctx is not None
            ctx.call_tool("activate_project", "{\"project\": \".\"}")
            out = ctx.call_tool("read_file", "{\"path\": \"big.txt\", \"head\": 399}")
            payload = json.loads(out)
            result = json.loads(payload["tool_result_json"])
            self.assertNotIn("warning", result)

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

    def test_tool_params_in_success_response(self) -> None:
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
            self.assertEqual(payload["tool_params"], {"path": "a.txt", "head": 1})

    def test_tool_params_in_empty_args_response(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            (repo / ".serena").mkdir()
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
            out = ctx.call_tool("list_memories", "{}")
            payload = json.loads(out)
            self.assertEqual(payload["tool_params"], {})

    def test_tool_params_in_truncated_response(self) -> None:
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
            self.assertEqual(payload["tool_params"], {"path": "a.txt"})

    def test_tool_params_in_budget_exhausted_response(self) -> None:
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
            ctx.call_tool("read_file", "{\"path\": \"a.txt\"}")
            out = ctx.call_tool("list_memories", "{}")
            payload = json.loads(out)
            self.assertEqual(payload["tool_status"], "budget_exhausted")
            self.assertEqual(payload["tool_params"], {})


if __name__ == "__main__":
    unittest.main()
