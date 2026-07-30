"""The model registry: model id -> a live provider.

Resolves a `probe` adapter in place of the real OpenAI one, so that "was a
provider built?" and "what was it built with?" are directly observable and no
test depends on the SDK's constructor.
"""

from __future__ import annotations

from typing import Any

import pytest

from midge import providers
from midge.config import ProviderConfig
from midge.providers import ModelRegistry, UnknownModel

_BUILT: list[dict[str, Any]] = []


class _Probe:
    """Records how it was constructed. Never opened, so it needs nothing else."""

    def __init__(self, **kw: Any) -> None:
        self.name = "probe"
        self.kwargs = kw
        _BUILT.append(kw)


@pytest.fixture(autouse=True)
def _probe_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve `kind = "probe"` for this module only.

    `providers.register` is global and has no inverse, so registering here would
    leak into every other test — `test_both_names_are_registered` asserts the
    exact set of adapters. Patching the lookup keeps the fixture local while
    still exercising the real code path, including the KeyError an unknown kind
    raises.
    """
    real = providers.get

    def fake_get(name: str) -> Any:
        return (lambda **kw: _Probe(**kw)) if name == "probe" else real(name)

    monkeypatch.setattr("midge.providers.registry.get", fake_get)
    _BUILT.clear()


def _registry(models: dict[str, str], configs: dict[str, ProviderConfig]) -> ModelRegistry:
    return ModelRegistry(models=models, providers=configs)


def _probe(**kw: Any) -> ProviderConfig:
    return ProviderConfig(kind="probe", **kw)


def _built(registry: ModelRegistry, model: str) -> _Probe:
    """Resolve, and narrow to the probe so its construction args are readable."""
    provider = registry.provider_for(model)
    assert isinstance(provider, _Probe)
    return provider


# --- the empty case, which everything else rests on -----------------------


def test_an_empty_registry_is_falsey_and_claims_nothing() -> None:
    """`Client` reads this as "permissive" and uses its own provider.

    Every install predating these tables is this case, so it is the
    compatibility guarantee for the whole feature.
    """
    registry = _registry({}, {})
    assert not registry
    assert "anything" not in registry
    assert registry.names() == []
    assert registry.diagnostics == []


def test_a_client_given_no_registry_builds_an_empty_one() -> None:
    # So `stream` has something to ask, and asks it the same way either way.
    from midge.client import Client

    client = Client()
    assert not client.registry
    assert client.registry.names() == []


# --- resolution -----------------------------------------------------------


def test_a_registered_model_resolves_to_its_provider() -> None:
    registry = _registry({"m1": "svc"}, {"svc": _probe(base_url="http://x/v1")})
    provider = _built(registry, "m1")
    assert provider.name == "probe"
    assert provider.kwargs["base_url"] == "http://x/v1"


def test_two_models_on_two_providers_get_two_instances() -> None:
    registry = _registry(
        {"m1": "a", "m2": "b"},
        {"a": _probe(base_url="http://a/v1"), "b": _probe(base_url="http://b/v1")},
    )
    assert registry.provider_for("m1") is not registry.provider_for("m2")
    assert len(_BUILT) == 2


def test_two_models_on_one_provider_share_an_instance() -> None:
    # Caching matters: a provider owns an HTTP client, and building one per
    # request would leak connections for the length of a session.
    registry = _registry({"m1": "svc", "m2": "svc"}, {"svc": _probe()})
    assert registry.provider_for("m1") is registry.provider_for("m2")
    assert len(_BUILT) == 1


def test_a_provider_is_built_only_when_a_model_reaches_it() -> None:
    # Listing a service you never use should cost nothing — no client, no
    # connection pool, and no credential lookup.
    registry = _registry({"m1": "used"}, {"used": _probe(), "spare": _probe()})
    assert _BUILT == []
    registry.provider_for("m1")
    assert len(_BUILT) == 1


def test_membership_and_names_report_the_registered_models() -> None:
    registry = _registry({"z": "svc", "a": "svc"}, {"svc": _probe()})
    assert registry
    assert "z" in registry and "nope" not in registry
    assert registry.names() == ["a", "z"]  # sorted, so a picker is stable


# --- refusal --------------------------------------------------------------


def test_an_unregistered_model_raises_and_names_the_alternatives() -> None:
    registry = _registry({"m1": "svc"}, {"svc": _probe()})
    with pytest.raises(UnknownModel) as excinfo:
        registry.provider_for("m2")
    assert excinfo.value.model == "m2"
    assert excinfo.value.registered == ["m1"]
    assert "m1" in str(excinfo.value)


# --- validating the wiring, not the vocabulary ----------------------------


def test_a_model_naming_an_undefined_provider_is_dropped() -> None:
    """The rest of the registry still loads.

    One typo should cost the user that one model, not the whole file.
    """
    registry = _registry({"good": "svc", "orphan": "nowhere"}, {"svc": _probe()})
    assert registry.names() == ["good"]
    [d] = registry.diagnostics
    assert d.event == "model_provider_undefined"
    assert d.fields == {"model": "orphan", "provider": "nowhere"}


def test_a_provider_naming_an_unknown_adapter_is_dropped() -> None:
    registry = _registry({"m": "svc"}, {"svc": ProviderConfig(kind="anthropic")})
    assert registry.names() == []
    events = [d.event for d in registry.diagnostics]
    # The provider goes, and the model that depended on it goes with it.
    assert events == ["provider_kind_unknown", "model_provider_undefined"]


def test_the_unknown_adapter_diagnostic_names_what_is_registered() -> None:
    registry = _registry({}, {"svc": ProviderConfig(kind="nope")})
    # One diagnostic, not two: a provider that was dropped is not then also
    # reported as unused.
    [kind_unknown] = registry.diagnostics
    assert kind_unknown.event == "provider_kind_unknown"
    assert "openai" in kind_unknown.fields["known"]


def test_providers_with_no_models_are_reported() -> None:
    # Harmless — the registry is empty, so everything falls back — but defining
    # services and routing nothing to them is never what someone meant.
    registry = _registry({}, {"svc": _probe()})
    assert [d.event for d in registry.diagnostics] == ["providers_unused"]


def test_a_registered_model_id_is_passed_through_unaltered() -> None:
    # midge ships no model list and validates no model id. A name that does not
    # exist at the vendor is the vendor's error to give.
    registry = _registry({"totally-made-up-9000": "svc"}, {"svc": _probe()})
    assert registry.provider_for("totally-made-up-9000").name == "probe"
    assert registry.diagnostics == []


# --- credentials ----------------------------------------------------------


def test_the_credential_is_fetched_by_the_name_the_config_gave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOME_OTHER_KEY", "sk-from-env")
    registry = _registry({"m": "svc"}, {"svc": _probe(api_key_env="SOME_OTHER_KEY")})
    assert _built(registry, "m").kwargs["api_key"] == "sk-from-env"


def test_no_api_key_env_means_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # A local server needs none; the adapter supplies its own placeholder.
    registry = _registry({"m": "svc"}, {"svc": _probe()})
    assert _built(registry, "m").kwargs["api_key"] is None


def test_an_unset_variable_yields_none_rather_than_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEVER_SET_ANYWHERE", raising=False)
    registry = _registry({"m": "svc"}, {"svc": _probe(api_key_env="NEVER_SET_ANYWHERE")})
    assert _built(registry, "m").kwargs["api_key"] is None


# --- capabilities ---------------------------------------------------------


def test_include_usage_reaches_the_provider_as_a_capability() -> None:
    registry = _registry({"m": "svc"}, {"svc": _probe(include_usage=False)})
    caps = _built(registry, "m").kwargs["capabilities"]
    assert caps is not None and caps.stream_usage is False


def test_no_include_usage_leaves_the_adapter_to_declare_it() -> None:
    # None is not False: it means "whatever the adapter says".
    registry = _registry({"m": "svc"}, {"svc": _probe()})
    assert _built(registry, "m").kwargs["capabilities"] is None
