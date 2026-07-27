from __future__ import annotations

from pathlib import Path

import pytest

from midge.tools.coding import bash, edit, read, write


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
