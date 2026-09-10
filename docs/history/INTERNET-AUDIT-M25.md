# Internet / Methodology Audit — M25

M25 uses the SBS→VBS opportunity framing standard in algorithm-selection research.

Relevant methodology references reviewed on 2026-09-06:

- ASlib specification: https://github.com/coseal/aslib-spec/blob/master/format.md
  - scenario data separates algorithm runs, features, feature costs, and evaluation splits;
  - ASlib explicitly recommends feature-cost reporting and notes that runtime improvement against SBS cannot be analyzed correctly without feature costs;
  - repeated cross-validation is supported for small/unstable scenarios.
- ASlib scenario repository: https://github.com/coseal/aslib_data
- Algorithm selection literature commonly treats SBS as the deployment baseline and VBS as the oracle upper-bound opportunity; selector quality is measured by how much of this gap can be closed.

M25 therefore charges current inspection cost to the oracle opportunity before deciding whether selector research is justified. This is intentionally stricter than looking only at raw DS/IPM winner counts.
