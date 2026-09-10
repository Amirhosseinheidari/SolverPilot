from __future__ import annotations

import importlib
from typing import Any

from .errors import ExtensionResolutionError
from .manifest import ExtensionManifest


def resolve_entrypoint(entrypoint: str) -> Any:
    """Resolve ``package.module:Attribute`` only when explicitly requested by the caller."""

    raw = str(entrypoint).strip()
    if ":" not in raw:
        raise ExtensionResolutionError("entrypoint must use the form 'package.module:Attribute'")
    module_name, attribute_path = raw.split(":", 1)
    try:
        value: Any = importlib.import_module(module_name)
    except Exception as exc:
        raise ExtensionResolutionError(f"could not import extension module {module_name!r}: {exc}") from exc
    for attribute in attribute_path.split("."):
        try:
            value = getattr(value, attribute)
        except AttributeError as exc:
            raise ExtensionResolutionError(
                f"entrypoint attribute {attribute_path!r} was not found in {module_name!r}"
            ) from exc
    return value


def resolve_manifest_entrypoint(manifest: ExtensionManifest) -> Any:
    if manifest.entrypoint is None:
        raise ExtensionResolutionError(f"extension {manifest.extension_id!r} declares no entrypoint")
    return resolve_entrypoint(manifest.entrypoint)
