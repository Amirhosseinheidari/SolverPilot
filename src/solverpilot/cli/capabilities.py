from __future__ import annotations

import argparse
import json

from solverpilot.capabilities import conform_backends, project_legacy_manifest
from solverpilot.runtime import builtin_backend_candidates


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect SolverPilot backend capability protocol v2 manifests")
    p.add_argument("--verify", action="store_true", help="run native problem-class conformance smokes")
    p.add_argument("--p0-schema", action="store_true", help="emit the P0 frozen capability schema projection")
    p.add_argument("--backend", action="append", default=[], help="restrict output to one or more backend names")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    backends = list(builtin_backend_candidates())
    if args.backend:
        wanted = set(args.backend)
        backends = [b for b in backends if b.manifest.name in wanted]
        missing = sorted(wanted - {b.manifest.name for b in backends})
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
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
