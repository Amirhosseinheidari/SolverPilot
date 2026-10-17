# Installed artifact test identity

The ordinary nine-cell OS/Python regression matrix builds a wheel, installs that
exact file, and enables `tools/release_import_guard.py` in the disposable Python
environment. Release qualification enables the same guard after installing its
frozen wheel, including the dedicated HiGHS and OSQP integration tests.

The guard compares package files to SHA-256 values read directly from the wheel.
A `.pth` bootstrap runs in the parent interpreter and children using the same
environment, including `python -I`. An import finder checks module origins and
contents again at import time, so a late `sys.path` change cannot silently select
the checkout. Failures terminate with exit code 86; Python's usual nonfatal
handling of `.pth` errors cannot turn a failed guard into a passing test.

Native subprocess tests inherit the location of the package selected by their
parent, rather than unconditionally inserting the checkout's `src` directory.
Explicit source testing remains possible with `PYTHONPATH=src` in an unguarded
development environment. Tests may still read repository fixtures and source
contracts; that is different from importing the implementation under test.

Each process writes a separate receipt. The aggregate records the wheel digest,
interpreter, package origin, failures, and how many processes actually imported
SolverPilot. Raw receipts are uploaded even when tests fail. Startup alone is
not evidence of a package import. Test coverage and numerical correctness are
established by the tests, not by these identity receipts.

The guard is test tooling, not part of the published runtime or a security
sandbox. A different interpreter/new virtual environment requires its own
installation and guard; deliberate `-S`, custom loaders, or hostile code can
bypass Python startup machinery. The integration fixtures separately check
normal and isolated children, early and late source shadowing, and changed
installed Python/data files.
