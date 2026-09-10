# M31 publication audit

## Packaging standard

Current PyPA metadata uses SPDX license expressions (`license`) together with `license-files`; author/maintainer and project URLs are independent metadata fields. M31 upgrades the setuptools build-system floor to a PEP-639-capable version (`setuptools>=77.0.3`) but deliberately does not invent owner metadata.

## Unresolved owner decisions

1. license/SPDX expression and matching license text;
2. final public distribution/project name;
3. author/maintainer names and emails if desired;
4. source/documentation/issue-tracker/homepage URLs.

These are publication blockers, not solver defects.

## Naming risk

The public-facing name **OptiMind** is already used by a Microsoft/OptiGuide optimization-reasoning research project. This does not by itself establish a legal conflict, but it is enough to require an explicit owner naming/branding decision before 1.0 publication. M31 therefore retains the technical distribution name `optimind-core-codename` and does not claim that it is the final public brand.

## Dependency availability vs compatibility

PyPI currently exposes wheels for the pinned/bounded optional integrations across CPython 3.10-3.13 and the major desktop OS families. This only reduces expected installation risk; it does not replace OptiMind CI execution. See `M31-COMPATIBILITY-MATRIX.json` for the claim boundary.
