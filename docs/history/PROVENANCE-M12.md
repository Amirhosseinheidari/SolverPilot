# M12 Provenance Note

The original M11 working source tree was not retained in `/mnt/data/m11_work`; only the built 0.0.12 wheel, reports, and raw artifacts were available. M12 did **not** pretend that missing source was present.

The package implementation baseline was audited against the actual 0.0.12 wheel. Relative to that wheel, M12 contains one new package module (`benchmark/splits.py`) and intentional changes to 16 existing package modules covering benchmark trust, environment/thread accounting, solver semantics, policy evaluation, CLI, and bridge provenance. No package module present in the 0.0.12 wheel disappeared.

This means M12 source is a reconstructed, audited successor of the M11 wheel; it is not claimed to be a byte-for-byte recovery of the lost M11 working tree/tests. M12 itself is fully packaged from the current source tree and tested as version 0.0.13.
