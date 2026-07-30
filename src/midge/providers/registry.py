"""The model registry: which service a given model id lives on.

Two different things in this package are called a registry, so to be plain about
it: `base.py` registers **adapters** — one per wire format, keyed by a `kind`
like `openai` or `openai-compatible`. This module maps a **model id** to a live
`Provider` built from one of those adapters, keyed by whatever the user called
the service in `[providers.*]`.

    "gpt-4o-mini"  ->  provider "openai"  ->  adapter kind "openai"

The rule this exists to enforce is that **the model determines the provider**.
`Client.stream` already takes a model per call — `Agent.model` is mutable and
re-read every iteration, a hook can override it through `BeforeProviderRequest`,
and a sub-agent runs its own model against the parent's `Client` — so resolution
belongs at that call rather than at construction.

**Empty means permissive** — but that policy lives in `Client`, which falls back
to the single provider it built when this registry holds nothing. An empty
registry is every install predating these tables, and is why the registry is
additive rather than a migration. Writing a `[models]` table is what turns
enforcement on.

**midge never populates this.** Vendors add and retire models continuously, so
any list checked into this repo is wrong within weeks. A user lists the models
they want available; those are the models that can be used. Model ids are passed
through to the provider unaltered — a typo fails at the API with the vendor's own
error, which is more accurate than anything midge could say. Do not add a
known-models check or a "did you mean".

What *is* validated is the wiring: a model naming a provider that was never
defined, or a provider naming an adapter that does not exist, is dropped with a
diagnostic. The user asked for something unreachable, and saying so at startup
beats a confusing failure on the first turn.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from midge.config import Diagnostic, ProviderConfig
from midge.providers.base import Capabilities, Provider, get, names


class UnknownModel(LookupError):
    """A model that no entry in a non-empty registry claims."""

    def __init__(self, model: str, registered: list[str]) -> None:
        known = ", ".join(registered) or "none"
        super().__init__(f"Unknown model {model!r}. Registered: {known}")
        self.model = model
        self.registered = registered


class ModelRegistry:
    def __init__(
        self,
        *,
        models: Mapping[str, str],
        providers: Mapping[str, ProviderConfig],
    ) -> None:
        self._instances: dict[str, Provider] = {}
        self._providers: dict[str, ProviderConfig] = {}
        self._models: dict[str, str] = {}
        self.diagnostics: list[Diagnostic] = []

        for name, config in providers.items():
            # Checked now because it is free — resolving the adapter is a dict
            # lookup. Building the provider is not, so that waits for first use.
            try:
                get(config.kind)
            except KeyError:
                self.diagnostics.append(
                    Diagnostic(
                        "provider_kind_unknown",
                        {"provider": name, "kind": config.kind, "known": ",".join(names())},
                    )
                )
                continue
            self._providers[name] = config

        for model, provider in models.items():
            if provider not in self._providers:
                self.diagnostics.append(
                    Diagnostic(
                        "model_provider_undefined", {"model": model, "provider": provider}
                    )
                )
                continue
            self._models[model] = provider

        if self._providers and not self._models:
            # Services defined and nothing routed to them. Harmless — the
            # registry is empty, so everything falls back — but it is never what
            # someone meant to write.
            self.diagnostics.append(
                Diagnostic("providers_unused", {"providers": ",".join(sorted(self._providers))})
            )

    def __bool__(self) -> bool:
        return bool(self._models)

    def __contains__(self, model: str) -> bool:
        return model in self._models

    def names(self) -> list[str]:
        return sorted(self._models)

    def provider_for(self, model: str) -> Provider:
        """The provider for `model`, built on first use.

        Raises rather than falling back. What an empty registry means is
        `Client`'s policy, not this map's — mixing the two here would make the
        registry's answer depend on how its caller wanted to be treated.
        """
        name = self._models.get(model)
        if name is None:
            raise UnknownModel(model, self.names())
        instance = self._instances.get(name)
        if instance is None:
            instance = self._instances[name] = self._build(name)
        return instance

    def _build(self, name: str) -> Provider:
        config = self._providers[name]
        return get(config.kind)(
            # The credential is fetched by the *name* the config gave, and is
            # never in the config itself. A file gets committed; the environment
            # does not.
            api_key=os.getenv(config.api_key_env) if config.api_key_env else None,
            base_url=config.base_url,
            capabilities=(
                None
                if config.include_usage is None
                else Capabilities(stream_usage=config.include_usage)
            ),
        )


__all__ = ["ModelRegistry", "UnknownModel"]
