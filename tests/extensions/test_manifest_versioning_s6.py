from __future__ import annotations

import math
import sys

import pytest

from solverpilot.extensions import (
    ExtensionDependency,
    ExtensionKind,
    ExtensionLifecycle,
    ExtensionManifest,
    ExtensionManifestError,
    ExtensionResolutionError,
    SemanticVersion,
    resolve_entrypoint,
    resolve_manifest_entrypoint,
    version_satisfies,
)


def manifest(**overrides):
    data = dict(
        extension_id="reporter.demo",
        name="Demo",
        version="1.2.3",
        kind="reporter",
        description="demo",
        capabilities=(" Markdown ", "markdown", "json"),
        dependencies=("core.helper>=1.0.0,<2.0.0",),
        lifecycle="active",
        metadata={"nested": {"x": [1, 2, 3]}},
    )
    data.update(overrides)
    return ExtensionManifest(**data)


def test_manifest_is_normalized_immutable_and_content_addressed():
    item = manifest()
    assert item.extension_id == "reporter.demo"
    assert item.kind is ExtensionKind.REPORTER
    assert item.lifecycle is ExtensionLifecycle.ACTIVE
    assert item.capabilities == ("markdown", "json")
    assert len(item.sha256) == 64
    assert item.sha256 == manifest().sha256
    with pytest.raises(TypeError):
        item.metadata["x"] = 1


def test_manifest_hash_changes_when_semantics_change():
    assert manifest().sha256 != manifest(description="different").sha256


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_manifest_rejects_nonfinite_metadata(value):
    with pytest.raises(ExtensionManifestError):
        manifest(metadata={"bad": value})


def test_manifest_rejects_invalid_identifier_and_self_dependency():
    with pytest.raises(ExtensionManifestError):
        manifest(extension_id="Bad ID")
    with pytest.raises(ExtensionManifestError):
        manifest(dependencies=("reporter.demo",))


def test_dependency_parser_supports_version_constraints():
    dep = ExtensionDependency.parse("core.helper>=1.0.0,<2.0.0")
    assert dep.extension_id == "core.helper"
    assert dep.version == ">=1.0.0,<2.0.0"


def test_semver_precedence_and_build_metadata():
    assert SemanticVersion.parse("1.0.0-alpha") < SemanticVersion.parse("1.0.0")
    assert SemanticVersion.parse("1.0.0+build.1") == SemanticVersion.parse("1.0.0+build.2")


def test_version_constraint_language():
    assert version_satisfies("1.4.2", ">=1.2.0,<2.0.0")
    assert version_satisfies("1.4.2", "^1.2.0")
    assert version_satisfies("0.2.9", "^0.2.1")
    assert not version_satisfies("0.3.0", "^0.2.1")
    assert version_satisfies("1.2.9", "~1.2.0")
    assert not version_satisfies("1.3.0", "~1.2.0")


def test_invalid_semver_or_constraint_fails_closed():
    with pytest.raises(ExtensionManifestError):
        SemanticVersion.parse("1.2")
    with pytest.raises(ExtensionManifestError):
        version_satisfies("1.0.0", ">=")


def test_entrypoint_resolution_is_explicit():
    item = manifest(entrypoint="json:loads")
    assert resolve_manifest_entrypoint(item) is not None
    assert resolve_entrypoint("json:loads") is not None


def test_missing_or_bad_entrypoint_fails_without_discovery():
    with pytest.raises(ExtensionResolutionError):
        resolve_manifest_entrypoint(manifest(entrypoint=None))
    with pytest.raises(ExtensionResolutionError):
        resolve_entrypoint("definitely_missing_solverpilot_module_xyz:Thing")


def test_importing_extensions_does_not_import_declared_entrypoint_module():
    sys.modules.pop("definitely_missing_solverpilot_module_xyz", None)
    manifest(entrypoint="definitely_missing_solverpilot_module_xyz:Thing")
    assert "definitely_missing_solverpilot_module_xyz" not in sys.modules


def test_dependency_parser_preserves_semver_identifier_case():
    dep = ExtensionDependency.parse("core.helper==1.0.0-RC.1")
    assert dep.extension_id == "core.helper"
    assert dep.version == "==1.0.0-RC.1"
    assert version_satisfies("1.0.0-RC.1", dep.version)
    assert not version_satisfies("1.0.0-rc.1", dep.version)


def test_entrypoint_rejects_empty_attribute_segments():
    with pytest.raises(ExtensionManifestError):
        manifest(entrypoint="json:decoder..JSONDecoder")
    with pytest.raises(ExtensionManifestError):
        manifest(entrypoint="json:decoder.")


def test_manifest_rejects_cyclic_metadata():
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ExtensionManifestError, match="cyclic"):
        manifest(metadata=cyclic)


def test_version_constraint_rejects_empty_clauses():
    with pytest.raises(ExtensionManifestError):
        version_satisfies("1.2.3", ">=1.0.0,,<2.0.0")
    with pytest.raises(ExtensionManifestError):
        version_satisfies("1.2.3", ">=1.0.0,")


def test_self_check_contract_rejects_non_boolean_ok():
    from solverpilot.extensions import ExtensionSelfCheck

    with pytest.raises(ExtensionManifestError):
        ExtensionSelfCheck(ok=1)
