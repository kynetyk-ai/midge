from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import Field, ValidationError

from midge.tools import Tool, ToolRegistry, tool


async def test_simple_tool_decoration() -> None:
    @tool
    async def echo(text: str) -> str:
        """Return the text unchanged."""
        return text

    assert isinstance(echo, Tool)
    assert echo.name == "echo"
    assert echo.description == "Return the text unchanged."
    assert await echo.invoke({"text": "hi"}) == "hi"


async def test_tool_with_default_arguments() -> None:
    @tool
    async def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone."""
        return f"{greeting}, {name}"

    assert await greet.invoke({"name": "Ada"}) == "Hello, Ada"
    assert await greet.invoke({"name": "Ada", "greeting": "Hi"}) == "Hi, Ada"

    schema = greet.schema()["parameters"]
    assert "name" in schema["required"]
    assert "greeting" not in schema.get("required", [])
    assert schema["properties"]["greeting"]["default"] == "Hello"


async def test_tool_with_explicit_name_and_description() -> None:
    @tool(name="reverse_text", description="Reverse a string.")
    async def _impl(text: str) -> str:
        return text[::-1]

    assert _impl.name == "reverse_text"
    assert _impl.description == "Reverse a string."


async def test_tool_invocation_validates_args() -> None:
    @tool
    async def add(a: int, b: int) -> int:
        return a + b

    assert await add.invoke({"a": 1, "b": 2}) == 3

    with pytest.raises(ValidationError):
        await add.invoke({"a": "not a number", "b": 2})

    with pytest.raises(ValidationError):
        await add.invoke({"a": 1})  # missing b


async def test_extra_args_rejected() -> None:
    @tool
    async def noop(x: int) -> int:
        return x

    with pytest.raises(ValidationError):
        await noop.invoke({"x": 1, "extra": "nope"})


async def test_tool_requires_async_function() -> None:
    def sync_fn(x: int) -> int:
        return x

    with pytest.raises(TypeError, match="async"):
        tool(sync_fn)  # type: ignore[arg-type]


async def test_tool_rejects_var_args() -> None:
    with pytest.raises(TypeError, match="args"):

        @tool
        async def variadic(*args: int) -> int:
            return sum(args)


async def test_schema_includes_required_and_properties() -> None:
    @tool
    async def example(path: str, limit: int = 10) -> str:
        """Read a file."""
        return path

    s = example.schema()
    assert s["name"] == "example"
    assert s["description"] == "Read a file."
    params = s["parameters"]
    assert params["type"] == "object"
    assert set(params["properties"].keys()) == {"path", "limit"}
    assert params["required"] == ["path"]
    assert params["additionalProperties"] is False


async def test_tool_with_annotated_field_description() -> None:
    @tool
    async def query(
        q: Annotated[str, Field(description="search query")],
        limit: Annotated[int, Field(description="max results", ge=1, le=100)] = 10,
    ) -> str:
        return q

    schema = query.schema()["parameters"]
    assert schema["properties"]["q"]["description"] == "search query"
    assert schema["properties"]["limit"]["description"] == "max results"
    assert schema["properties"]["limit"]["minimum"] == 1
    assert schema["properties"]["limit"]["maximum"] == 100


async def test_registry_add_get_iter_len() -> None:
    @tool
    async def a(x: int) -> int:
        return x

    @tool
    async def b(y: int) -> int:
        return y

    reg = ToolRegistry([a, b])
    assert len(reg) == 2
    assert "a" in reg
    assert "b" in reg
    assert reg.get("a") is a
    assert {t.name for t in reg} == {"a", "b"}


async def test_registry_duplicate_name_rejected() -> None:
    @tool
    async def x(n: int) -> int:
        return n

    reg = ToolRegistry([x])
    with pytest.raises(ValueError, match="already registered"):
        reg.add(x)


async def test_registry_invoke() -> None:
    @tool
    async def double(n: int) -> int:
        return n * 2

    reg = ToolRegistry([double])
    assert await reg.invoke("double", {"n": 21}) == 42


async def test_registry_invoke_unknown_tool_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="missing"):
        await reg.invoke("missing", {})


async def test_registry_schemas_match_tools() -> None:
    @tool
    async def t1(a: int) -> int:
        """First tool."""
        return a

    @tool
    async def t2(b: str) -> str:
        """Second tool."""
        return b

    reg = ToolRegistry([t1, t2])
    schemas = reg.schemas()
    assert [s["name"] for s in schemas] == ["t1", "t2"]
    assert schemas[0]["description"] == "First tool."
