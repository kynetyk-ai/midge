from __future__ import annotations

import inspect
import typing
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, overload

from pydantic import BaseModel, ConfigDict, create_model

ToolFn = Callable[..., Awaitable[Any]]


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
    ) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self.params_model = params_model

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.params_model.model_json_schema(),
        }

    async def invoke(self, arguments: dict[str, Any], *, call_id: str | None = None) -> Any:
        # `call_id` is the provider's id for this tool call. The base tool has no
        # use for it; a subclass that produces its own artefacts uses it to tie
        # them back to the exact turn that asked for them.
        validated = self.params_model.model_validate(arguments)
        kwargs = {f: getattr(validated, f) for f in self.params_model.model_fields}
        return await self.fn(**kwargs)


@overload
def tool(fn: ToolFn, /) -> Tool: ...
@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[ToolFn], Tool]: ...
def tool(
    fn: ToolFn | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
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
            raise KeyError(f"Tool {name!r} not registered")
        return await t.invoke(arguments, call_id=call_id)
