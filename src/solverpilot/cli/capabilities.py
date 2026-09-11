from __future__ import annotations

import argparse
import json

from solverpilot.capabilities import conform_backends, project_legacy_manifest
from solverpilot.runtime import builtin_backend_candidates
from solverpilot.runtime.catalog import backend_catalog, capability_manifest, verify_specialized_backend


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect SolverPilot backend capability protocol v2 manifests")
    p.add_argument("--verify", action="store_true", help="run native problem-class conformance smokes")
    p.add_argument("--p0-schema", action="store_true", help="emit the P0 frozen capability schema projection")
    p.add_argument("--backend", action="append", default=[], help="restrict output to one or more backend names")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    catalog = backend_catalog()
    selected = set(args.backend) if args.backend else set(catalog)
    missing = sorted(selected-set(catalog))
    if missing:
        raise SystemExit(f"unknown backend(s): {', '.join(missing)}")
    specialized = [(name, b) for name, b in catalog.items() if name in selected and not hasattr(b, 'manifest')]
    backends = [b for name, b in catalog.items() if name in selected and hasattr(b, 'manifest')]
    if args.backend:
        wanted = set(args.backend)
        backends = [b for b in backends if b.manifest.name in wanted]
        missing = sorted(wanted - {b.manifest.name for b in backends} - {name for name, _ in specialized})
        if missing:
            raise SystemExit(f"unknown backend(s): {', '.join(missing)}")

    if args.verify:
        reports = conform_backends(backends)
        payload = []
        for report in reports:
            manifest = report.manifest
            payload.append({
                "backend": report.backend,
                "available": report.available,
                "conformance_passed": report.passed,
                "checks": [
                    {
                        "capability": c.capability.value,
                        "passed": c.passed,
                        "evidence_id": c.evidence_id,
                        "detail": c.detail,
                    }
                    for c in report.checks
                ],
                "manifest": manifest.to_p0_schema_dict() if args.p0_schema else manifest.to_canonical_dict(),
                "signature": manifest.signature,
            })
    else:
        payload = []
        for backend in backends:
            import solverpilot
            manifest = project_legacy_manifest(backend.manifest, adapter_version=solverpilot.__version__)
            payload.append({
                "backend": backend.manifest.name,
                "available": backend.is_available(),
                "manifest": manifest.to_p0_schema_dict() if args.p0_schema else manifest.to_canonical_dict(),
                "signature": manifest.signature,
            })
    for name, backend in specialized:
        manifest = capability_manifest(backend)
        item = {'backend': name, 'available': backend.is_available(),
                'manifest': manifest.to_p0_schema_dict() if args.p0_schema else manifest.to_canonical_dict(),
                'signature': manifest.signature}
        if args.verify:
            item.update(conformance_passed=verify_specialized_backend(backend), checks=[])
        payload.append(item)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
