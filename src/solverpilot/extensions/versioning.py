from __future__ import annotations

from dataclasses import dataclass
import re

from .errors import ExtensionManifestError

_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_COMPARATOR_RE = re.compile(r"^(==|!=|>=|<=|>|<|\^|~)?\s*(.+)$")


def _identifier_key(identifier: str) -> tuple[int, int | str]:
    if identifier.isdigit():
        return (0, int(identifier))
    return (1, identifier)


@dataclass(frozen=True, slots=True)
class SemanticVersion:
    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()
    build: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: str) -> "SemanticVersion":
        raw = str(value).strip()
        match = _SEMVER_RE.fullmatch(raw)
        if match is None:
            raise ExtensionManifestError(
                f"version {value!r} must be valid SemVer MAJOR.MINOR.PATCH with optional prerelease/build suffix"
            )
        pre = tuple(match.group(4).split(".")) if match.group(4) else ()
        build = tuple(match.group(5).split(".")) if match.group(5) else ()
        for ident in pre:
            if ident.isdigit() and len(ident) > 1 and ident.startswith("0"):
                raise ExtensionManifestError(f"invalid numeric prerelease identifier {ident!r}")
        return cls(int(match.group(1)), int(match.group(2)), int(match.group(3)), pre, build)

    def _precedence(self) -> tuple[object, ...]:
        base: tuple[object, ...] = (self.major, self.minor, self.patch)
        if not self.prerelease:
            return base + ((2, ""),)
        return base + ((1, tuple(_identifier_key(x) for x in self.prerelease)),)

    def __lt__(self, other: "SemanticVersion") -> bool:
        return self._precedence() < other._precedence()

    def __le__(self, other: "SemanticVersion") -> bool:
        return self == other or self < other

    def __gt__(self, other: "SemanticVersion") -> bool:
        return not self <= other

    def __ge__(self, other: "SemanticVersion") -> bool:
        return not self < other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (
            self.major,
            self.minor,
            self.patch,
            self.prerelease,
        ) == (
            other.major,
            other.minor,
            other.patch,
            other.prerelease,
        )

    def __str__(self) -> str:
        text = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            text += "-" + ".".join(self.prerelease)
        if self.build:
            text += "+" + ".".join(self.build)
        return text


def _caret_upper(v: SemanticVersion) -> SemanticVersion:
    if v.major > 0:
        return SemanticVersion(v.major + 1, 0, 0)
    if v.minor > 0:
        return SemanticVersion(0, v.minor + 1, 0)
    return SemanticVersion(0, 0, v.patch + 1)


def _tilde_upper(v: SemanticVersion) -> SemanticVersion:
    return SemanticVersion(v.major, v.minor + 1, 0)


def version_satisfies(version: str, specifier: str) -> bool:
    """Return whether *version* satisfies a small deterministic SemVer constraint language.

    Supported clauses are comma-separated ``== != >= <= > < ^ ~`` comparisons. An empty
    specifier accepts every valid version. Build metadata does not affect precedence.
    """

    current = SemanticVersion.parse(version)
    raw = str(specifier).strip()
    if not raw or raw == "*":
        return True
    parts = raw.split(",")
    if any(not part.strip() for part in parts):
        raise ExtensionManifestError(f"invalid version constraint {specifier!r}")
    clauses = [part.strip() for part in parts]
    for clause in clauses:
        match = _COMPARATOR_RE.fullmatch(clause)
        if match is None:
            raise ExtensionManifestError(f"invalid version constraint {clause!r}")
        op = match.group(1) or "=="
        target = SemanticVersion.parse(match.group(2))
        if op == "==" and not current == target:
            return False
        if op == "!=" and current == target:
            return False
        if op == ">=" and not current >= target:
            return False
        if op == "<=" and not current <= target:
            return False
        if op == ">" and not current > target:
            return False
        if op == "<" and not current < target:
            return False
        if op == "^" and not (current >= target and current < _caret_upper(target)):
            return False
        if op == "~" and not (current >= target and current < _tilde_upper(target)):
            return False
    return True
