from __future__ import annotations

from pathlib import Path

import pytest

from midge.tools.coding import bash, edit, grep, ls, read, write


async def test_read_basic(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("line1\nline2\nline3\n")

    out = await read.invoke({"path": str(f)})
    assert out == "line1\nline2\nline3"


async def test_read_with_offset_and_limit(tmp_path: Path) -> None:
    f = tmp_path / "many.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")

    out = await read.invoke({"path": str(f), "offset": 5, "limit": 2})
    assert out.startswith("line5\nline6")
    assert "offset=7" in out


async def test_read_truncation_hint(tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")

    out = await read.invoke({"path": str(f), "limit": 3})
    assert "truncated" in out
    assert "offset=4" in out


async def test_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await read.invoke({"path": str(tmp_path / "nope.txt")})


async def test_read_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        await read.invoke({"path": str(tmp_path)})


async def test_read_oversized_first_line_raises(tmp_path: Path) -> None:
    f = tmp_path / "huge.txt"
    f.write_text("x" * 60_000)
    with pytest.raises(ValueError, match="exceeds"):
        await read.invoke({"path": str(f)})


async def test_write_creates_file_and_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "file.txt"
    out = await write.invoke({"path": str(target), "content": "hello"})

    assert target.read_text() == "hello"
    assert "5 bytes" in out


async def test_write_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("original")
    await write.invoke({"path": str(target), "content": "new"})
    assert target.read_text() == "new"


async def test_write_byte_count_for_unicode(tmp_path: Path) -> None:
    target = tmp_path / "u.txt"
    out = await write.invoke({"path": str(target), "content": "héllo"})
    # h=1 + é=2 + l=1 + l=1 + o=1 = 6 bytes in utf-8
    assert "6 bytes" in out


async def test_edit_simple(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    return 1\n")

    out = await edit.invoke(
        {
            "path": str(f),
            "edits": [{"old_text": "return 1", "new_text": "return 2"}],
        }
    )

    assert f.read_text() == "def foo():\n    return 2\n"
    assert "first_changed_line: 2" in out
    assert "-    return 1" in out
    assert "+    return 2" in out


async def test_edit_multiple_non_overlapping(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("a = 1\nb = 2\nc = 3\n")

    await edit.invoke(
        {
            "path": str(f),
            "edits": [
                {"old_text": "a = 1", "new_text": "a = 10"},
                {"old_text": "c = 3", "new_text": "c = 30"},
            ],
        }
    )
    assert f.read_text() == "a = 10\nb = 2\nc = 30\n"


async def test_edit_missing_match_raises(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("hello\n")
    with pytest.raises(ValueError, match="not found"):
        await edit.invoke(
            {
                "path": str(f),
                "edits": [{"old_text": "missing", "new_text": "x"}],
            }
        )


async def test_edit_ambiguous_match_raises(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("foo\nfoo\n")
    with pytest.raises(ValueError, match="multiple"):
        await edit.invoke(
            {
                "path": str(f),
                "edits": [{"old_text": "foo", "new_text": "bar"}],
            }
        )


async def test_edit_overlapping_raises(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("abcdef\n")
    with pytest.raises(ValueError, match=r"[Oo]verlapping"):
        await edit.invoke(
            {
                "path": str(f),
                "edits": [
                    {"old_text": "abcd", "new_text": "X"},
                    {"old_text": "cdef", "new_text": "Y"},
                ],
            }
        )


async def test_edit_preserves_crlf(tmp_path: Path) -> None:
    f = tmp_path / "win.txt"
    f.write_bytes(b"foo\r\nbar\r\n")
    await edit.invoke(
        {
            "path": str(f),
            "edits": [{"old_text": "bar", "new_text": "baz"}],
        }
    )
    assert f.read_bytes() == b"foo\r\nbaz\r\n"


async def test_edit_preserves_bom(tmp_path: Path) -> None:
    f = tmp_path / "bom.txt"
    f.write_bytes("﻿hello world".encode())
    await edit.invoke(
        {
            "path": str(f),
            "edits": [{"old_text": "world", "new_text": "there"}],
        }
    )
    assert f.read_bytes() == "﻿hello there".encode()


async def test_bash_simple_command() -> None:
    out = await bash.invoke({"command": "echo hello"})
    assert "hello" in out


async def test_bash_captures_stderr() -> None:
    out = await bash.invoke({"command": "echo to_err 1>&2"})
    assert "to_err" in out


async def test_bash_nonzero_exit_code_in_output() -> None:
    out = await bash.invoke({"command": "exit 7"})
    assert "exit code: 7" in out


async def test_bash_timeout_kills_process() -> None:
    with pytest.raises(TimeoutError):
        await bash.invoke({"command": "sleep 5", "timeout": 1})


async def test_bash_tail_truncates_long_output() -> None:
    out = await bash.invoke(
        {"command": "for i in $(seq 1 3000); do echo line$i; done"}
    )
    assert "truncated" in out
    assert "line3000" in out
    assert "line1\n" not in out


# --- #97: a near miss says what to fix -------------------------------------


async def _edit_error(tmp_path: Path, content: str, old: str) -> str:
    f = tmp_path / "doc.md"
    f.write_text(content)
    with pytest.raises(ValueError) as info:
        await edit.invoke({"path": str(f), "edits": [{"old_text": old, "new_text": "x"}]})
    assert f.read_text() == content, "a near miss must never be applied"
    return str(info.value)


async def test_edit_names_a_whitespace_only_difference(tmp_path: Path) -> None:
    # The first failure #97 recorded: a list item's continuation indent dropped.
    content = "intro\n\n1. Add the field, with prose\n   above it saying *why*.\n2. Next\n"
    msg = await _edit_error(tmp_path, content, "with prose\nabove it saying *why*.")
    assert "not found" in msg
    assert "whitespace" in msg
    assert "lines 3-4" in msg
    assert "   above it saying *why*." in msg


async def test_edit_points_at_the_closest_region(tmp_path: Path) -> None:
    # The second: one word corrupted mid-string.
    content = "a\nround-trips to `Config()` with no\ndiagnostics. A key in the example\nz\n"
    msg = await _edit_error(
        tmp_path, content, "round-trips to `Config()` with no\nostics. A key in the example"
    )
    assert "closest match is lines 2-3" in msg
    assert "first differing at line 3" in msg
    assert "diagnostics. A key" in msg


async def test_edit_finds_a_near_miss_inside_a_longer_line(tmp_path: Path) -> None:
    content = "x = 1\nresult = compute(alpha, beta, gamma)  # the main call\ny = 2\n"
    msg = await _edit_error(tmp_path, content, "result = compute(alpha, betta, gamma)")
    assert "closest match is line 2" in msg


async def test_edit_says_when_nothing_is_close(tmp_path: Path) -> None:
    msg = await _edit_error(tmp_path, "alpha\nbeta\n", "something else entirely here")
    assert "nothing in the file is close" in msg


async def test_edit_ambiguity_says_where(tmp_path: Path) -> None:
    f = tmp_path / "code.py"
    f.write_text("foo\nbar\nfoo\n")
    with pytest.raises(ValueError, match=r"multiple locations \(lines 1, 3\)"):
        await edit.invoke({"path": str(f), "edits": [{"old_text": "foo", "new_text": "x"}]})


# --- ls and grep: the read-only way to look around -------------------------


async def test_ls_sorts_and_marks_directories(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "A").mkdir()
    (tmp_path / ".hidden").write_text("")
    out = await ls.invoke({"path": str(tmp_path)})
    assert out.splitlines() == [".hidden", "A/", "b.txt"]


async def test_ls_says_what_it_left_out(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"f{i}").write_text("")
    out = await ls.invoke({"path": str(tmp_path), "limit": 2})
    assert out.splitlines()[:2] == ["f0", "f1"]
    assert "3 more" in out


async def test_ls_refuses_a_file(tmp_path: Path) -> None:
    f = tmp_path / "x"
    f.write_text("")
    with pytest.raises(NotADirectoryError):
        await ls.invoke({"path": str(f)})


async def test_grep_finds_lines_with_numbers(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("one\ndef slugify(x):\nthree\n")
    (tmp_path / "b.md").write_text("slugify is documented here\n")
    out = await grep.invoke({"pattern": r"def \w+", "path": str(tmp_path)})
    assert out == "a.py:2: def slugify(x):"


async def test_grep_glob_literal_and_case(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = f(1)\n")
    (tmp_path / "b.md").write_text("X = F(1)\n")
    out = await grep.invoke(
        {"pattern": "f(1)", "path": str(tmp_path), "literal": True, "ignore_case": True}
    )
    assert out.splitlines() == ["a.py:1: x = f(1)", "b.md:1: X = F(1)"]
    only_py = await grep.invoke(
        {"pattern": "f(1)", "path": str(tmp_path), "literal": True, "glob": "*.py"}
    )
    assert only_py == "a.py:1: x = f(1)"


async def test_grep_respects_gitignore_in_a_repo(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("build/\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.txt").write_text("needle\n")
    (tmp_path / "src.txt").write_text("needle\n")
    out = await grep.invoke({"pattern": "needle", "path": str(tmp_path)})
    assert out == "src.txt:1: needle"


async def test_grep_skips_binary_and_caps_output(tmp_path: Path) -> None:
    (tmp_path / "bin.dat").write_bytes(b"needle\0\x01")
    (tmp_path / "many.txt").write_text("needle\n" * 10)
    out = await grep.invoke({"pattern": "needle", "path": str(tmp_path), "limit": 3})
    assert "bin.dat" not in out
    assert out.count("many.txt:") == 3
    assert "stopped at 3 matches" in out


async def test_grep_explains_a_bad_regex(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="literal=true"):
        await grep.invoke({"pattern": "f(", "path": str(tmp_path)})


def test_the_lookup_tools_are_read_only() -> None:
    assert read.read_only and ls.read_only and grep.read_only
    assert not (write.read_only or edit.read_only or bash.read_only)
