# Provenance M33

M33 starts from the final-verified M32 source/release lineage and changes release policy/CI/security packaging only; it does not add solver algorithms or learned routing.

Protocol was frozen before implementation at SHA-256:

`041dbc772f54a60d2dab451c4b34083e4d61c0e3bec6ee68c43cd54e0799921c`

Key lineage decisions:

- M32 exact-artifact/supply-chain hardening is preserved.
- M33 support policy is intentionally narrowed/advanced to CPython 3.12–3.14 based on current Scientific Python/Python ecosystem policy before external outcomes were observed.
- M33 runtime dependency floors become NumPy >=2.2 / SciPy >=1.15, with an explicit exact-minimum execution job.
- M30/M31/M32 API/backend snapshots remain historical and unchanged; M33 has its own snapshots.
- learned LP performance routing remains disabled.
- all unexecuted external compatibility/security/publication states remain pending rather than inferred from upstream wheel availability.
