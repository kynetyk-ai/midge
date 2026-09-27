from __future__ import annotations

import inspect
import typing
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, overload

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

ToolFn = Callable[..., Awaitable[Any]]


class ToolNotFound(KeyError):
    """The registry has no tool by that name.

    Its own type because "not found" is the registry's answer, and a `KeyError`
    raised *inside* a tool is the tool's — a missing note, a missing key. Both
    were caught as `KeyError`, so a tool that failed was reported as a tool that
    did not exist, and the model stopped calling it (#104).
    """


class InvalidArguments(ValueError):
    """The model's arguments did not validate against the tool's schema.

    Raised only around validation, for the same reason as `ToolNotFound`: a
    pydantic `ValidationError` from a tool's own body is a failure of the tool,
    not a mistake the model can fix by changing its arguments.
    """


class _ParamsBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tool:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        fn: ToolFn,
        params_model: type[BaseModel],
        read_only: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self.params_model = params_model
        self._read_only = read_only

    @property
    def read_only(self) -> bool:
        """Whether the tool only observes. The agent runs these concurrently
        and everything else one at a time, in the order the model asked.

        `False` is the default because it is the safe one: a tool nobody
        classified is serialized, which costs time and never correctness.
        """
        return self._read_only

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.params_model.model_json_schema(),
        }

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            validated = self.params_model.model_validate(arguments)
        except ValidationError as e:
            raise InvalidArguments(str(e)) from e
        return {f: getattr(validated, f) for f in self.params_model.model_fields}

    async def invoke(self, arguments: dict[str, Any], *, call_id: str | None = None) -> Any:
        # `call_id` is the provider's id for this tool call. The base tool has no
        # use for it; a subclass that produces its own artefacts uses it to tie
        # them back to the exact turn that asked for them.
        return await self.fn(**self.validate(arguments))


@overload
def tool(fn: ToolFn, /) -> Tool: ...
@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    read_only: bool = False,
) -> Callable[[ToolFn], Tool]: ...
def tool(
    fn: ToolFn | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    read_only: bool = False,
) -> Tool | Callable[[ToolFn], Tool]:
    def wrap(fn: ToolFn) -> Tool:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(
                f"@tool requires an async function; {fn.__name__} is not `async def`"
            )
        tool_name = name or fn.__name__
        tool_desc = description or (inspect.getdoc(fn) or "").strip()
        params_model = _build_params_model(fn, tool_name)
        return Tool(
            name=tool_name,
            description=tool_desc,
            fn=fn,
            params_model=params_model,
            read_only=read_only,
        )

    if fn is None:
        return wrap
    return wrap(fn)


def _build_params_model(fn: Callable[..., Any], tool_name: str) -> type[BaseModel]:
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"@tool does not support *args/**kwargs (in {fn.__name__}.{pname})"
            )
        annotation = hints.get(pname, Any)
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[pname] = (annotation, default)
    return create_model(
        f"{_pascal(tool_name)}Params",
        __base__=_ParamsBase,
        **fields,
    )


def _pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_"))


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"Tool {t.name!r} already registered")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def remove(self, name: str) -> None:
        """Forget a tool. Silent on a name that is not there, so a validator
        dropping several can do it without checking each one first."""
        self._tools.pop(name, None)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    async def invoke(
        self, name: str, arguments: dict[str, Any], *, call_id: str | None = None
    ) -> Any:
        t = self._tools.get(name)
        if t is None:
            raise ToolNotFound(f"Tool {name!r} not registered")
        return await t.invoke(arguments, call_id=call_id)
