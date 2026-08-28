"""MODEL_REGISTRY (Task 10, Part A).

The registry is what makes "the architecture is general" checkable rather than asserted:
three models with three different normalizations, so a demo cannot pass by working on
Qwen2's RMSNorm alone. These tests are cheap schema guards — no network, no downloads —
and they exist because the registry is consumed by `audit_model`, the CLI's `--all`
sweep, the dashboard's model picker and the transfer-readiness table, all of which
index it by key and read the same fields.
"""

from __future__ import annotations

import pytest

from kernelsmith.config import DEFAULT_AUDIT_MODEL, MODEL_REGISTRY, SERVED_MODEL
from kernelsmith.tools.profiler_tool import family_from_name, resolve_model_id

REQUIRED_KEYS = frozenset(
    {"hf_id", "family", "norm_type", "activation", "hidden_size", "description"}
)

ALL_MODELS = sorted(MODEL_REGISTRY)


def test_the_registry_is_not_empty():
    assert MODEL_REGISTRY


@pytest.mark.parametrize("key", ALL_MODELS)
def test_every_entry_has_every_required_key(key):
    assert set(MODEL_REGISTRY[key]) >= REQUIRED_KEYS, (
        f"{key} is missing {REQUIRED_KEYS - set(MODEL_REGISTRY[key])}"
    )


@pytest.mark.parametrize("key", ALL_MODELS)
def test_hf_ids_are_non_empty_org_slash_name_strings(key):
    hf_id = MODEL_REGISTRY[key]["hf_id"]
    assert isinstance(hf_id, str)
    assert hf_id.strip() == hf_id and hf_id
    assert "/" in hf_id, f"{key}: {hf_id!r} is not an org/name HuggingFace id"


@pytest.mark.parametrize("key", ALL_MODELS)
def test_hidden_size_is_a_positive_int(key):
    hidden = MODEL_REGISTRY[key]["hidden_size"]
    assert isinstance(hidden, int)
    assert not isinstance(hidden, bool)
    assert hidden > 0


@pytest.mark.parametrize("key", ALL_MODELS)
def test_the_text_fields_are_non_empty_strings(key):
    entry = MODEL_REGISTRY[key]
    for field in ("family", "norm_type", "activation", "description"):
        value = entry[field]
        assert isinstance(value, str) and value.strip(), f"{key}.{field} is {value!r}"


@pytest.mark.parametrize("key", ALL_MODELS)
def test_keys_are_lowercase_so_resolve_model_id_can_normalize(key):
    """`resolve_model_id` lowercases before looking up; a mixed-case key would never hit."""
    assert key == key.lower()
    assert resolve_model_id(key) == MODEL_REGISTRY[key]["hf_id"]
    assert resolve_model_id(key.upper()) == MODEL_REGISTRY[key]["hf_id"]


def test_hf_ids_are_unique():
    ids = [entry["hf_id"] for entry in MODEL_REGISTRY.values()]
    assert len(ids) == len(set(ids))


def test_the_default_audit_model_is_registered_and_is_the_served_model():
    """`make audit` with no --model must audit the model the server actually runs."""
    assert DEFAULT_AUDIT_MODEL in MODEL_REGISTRY
    assert MODEL_REGISTRY[DEFAULT_AUDIT_MODEL]["hf_id"] == SERVED_MODEL


def test_the_registry_covers_more_than_one_normalization():
    """The point of the registry: it is not three variations on the same architecture."""
    norms = {entry["norm_type"] for entry in MODEL_REGISTRY.values()}
    assert len(norms) >= 3, f"only {norms} represented — the transfer claim is untestable"


def test_the_registry_covers_more_than_one_model_family():
    families = {entry["family"] for entry in MODEL_REGISTRY.values()}
    assert len(families) >= 2, f"only {families} represented"


def test_every_norm_type_classifies_into_the_retrieval_family_norm():
    """RMSNorm, LayerNorm and BatchNorm must all land in `op_family="norm"`.

    This is the mechanism behind cross-model transfer: retrieval pre-filters on
    `op_family`, so if one of these classified elsewhere its architecture could never
    retrieve a skill learned on the others.
    """
    for key, entry in MODEL_REGISTRY.items():
        assert family_from_name(entry["norm_type"]) == "norm", (
            f"{key}: {entry['norm_type']} does not classify as norm"
        )


def test_the_default_run_demo_audit_model_matches_the_config_default():
    from kernelsmith.run_demo import DEFAULT_AUDIT_MODEL as cli_default

    assert cli_default == DEFAULT_AUDIT_MODEL
