# NATIVE WHEEL ATTEMPT — M9

## highspy

- downloaded: `False`
- return code: `6`
- stderr: `curl: (6) Could not resolve host: files.pythonhosted.org`
- expected SHA-256: `238b2ee88b974b21c7e9ef198139502a7d87451939cae143dce789bbda121182`

## osqp

- downloaded: `False`
- return code: `6`
- stderr: `curl: (6) Could not resolve host: files.pythonhosted.org`
- expected SHA-256: `2d3ee63e8c65ef89fce979c068d05bfc3ed92b1bbc4246fbacc663f86cbe02b2`

## pyscipopt

- downloaded: `False`
- return code: `6`
- stderr: `curl: (6) Could not resolve host: files.pythonhosted.org`
- expected SHA-256: `725503ea90fd0962f1111a5b46ee3e271b7375e1b0dfda708ce223dbebaeff5e`

NLopt is different in M9: version 2.11.0 was already installed in the execution environment, its wheel tag matches CPython 3.13 manylinux x86-64, and its public adapter was actually executed. The official PyPI wheel SHA-256 is recorded in `benchmarks/native-wheel-lock.json`.
