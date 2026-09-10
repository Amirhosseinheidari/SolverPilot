# M31 cross-platform source audit

M31 performs a source-level portability audit in addition to the runtime matrix definition.

## Finding fixed in M31

`optimind.benchmark.runner._clean_pythonpath()` previously identified hook-only paths with Unix-specific string markers such as `/python-hooks`. A Windows path using backslashes could therefore bypass the hook filter. This matters because clean benchmark workers use `python -S` plus a reconstructed `PYTHONPATH`, and the project previously identified startup hooks as a source of benchmark distortion.

M31 changes the check to normalize both `\\` and `/` separators before comparing path components. A regression test supplies both POSIX- and Windows-style paths and requires both hook directories to be rejected.

## Other portability observations

- runtime code does not directly use `fork`, `fcntl`, `resource`, or Unix signals;
- benchmark subprocess isolation uses `subprocess.run`, `sys.executable`, `os.pathsep`, and pathlib;
- Linux `/proc/cpuinfo` and `/proc/meminfo` are optional probes with platform fallbacks / `None`, not hard requirements;
- SLURM script generation is intentionally an HPC-specific feature and is not interpreted as Windows/macOS runtime support.

This source audit reduces known portability risk but does not replace execution on Windows/macOS.
