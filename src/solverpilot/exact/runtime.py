"""Explicit subprocess execution. Native status alone never establishes a proof."""
from dataclasses import dataclass, replace, field
from solverpilot._identity import new_execution_id
from fractions import Fraction
from hashlib import sha256
import math
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory, mkdtemp
from time import monotonic, sleep

from .certificate import MAX_CERTIFICATE_BYTES, bind_certificate, close_objective_bound
from .problem import ExactModel


@dataclass(frozen=True)
class ExactSolveResult:
    status: str
    independently_verified: bool
    reason: str
    x: tuple[Fraction, ...] | None = None
    objective: Fraction | None = None
    bound: Fraction | None = None
    absolute_gap: Fraction | None = None
    elapsed_s: float = 0.0
    problem_hash: str = ""
    certificate_sha256: str = ""
    input_certificate_sha256: str = ""
    checker_sha256: str = ""
    solver_sha256: str = ""
    evidence_directory: str | None = None
    execution_id: str = field(default_factory=new_execution_id, kw_only=True, compare=False)


def _executable(path):
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.is_file():
        raise ValueError("supply an absolute path to a trusted native executable")
    if candidate.suffix.lower() in (".bat", ".cmd"):
        raise ValueError("shell scripts are not accepted as native executables")
    return str(candidate.resolve())


def _digest(path):
    with open(path, "rb") as stream:
        h = sha256()
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _budget(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("time_limit must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("time_limit must be a finite positive number")
    return float(value)


def _output_exceeds_limit(directory):
    # SCIP combines/removes certificate fragments while it is running. A file
    # disappearing between enumeration and stat is normal, not a failed solve.
    for path in directory.iterdir():
        try:
            if path.is_file() and path.stat().st_size > MAX_CERTIFICATE_BYTES:
                return True
        except FileNotFoundError:
            continue
    return False


def _run(command, directory, log_name, deadline):
    """Bound runtime and output, reap children on timeout and on caller interruption."""
    if monotonic() >= deadline:
        raise TimeoutError("exact execution budget exhausted")
    flags = {"start_new_session": True} if os.name != "nt" else {}
    with (directory / log_name).open("wb") as log:
        proc = subprocess.Popen(command, cwd=directory, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, shell=False, **flags)
        try:
            while proc.poll() is None:
                if monotonic() >= deadline:
                    raise TimeoutError("exact execution budget exhausted")
                if _output_exceeds_limit(directory):
                    raise ValueError("native output exceeds size limit")
                sleep(min(0.02, max(0.0, deadline - monotonic())))
            if proc.returncode:
                raise RuntimeError(f"{log_name} process failed (exit {proc.returncode})")
        finally:
            if proc.poll() is None:
                if os.name != "nt":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    # Windows venv executables may be launchers with a child Python.
                    # Kill our process tree before reaping the launcher.
                    taskkill = str(Path(os.environ["SystemRoot"]) / "System32" / "taskkill.exe")
                    subprocess.run([taskkill, "/PID", str(proc.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=5, check=False)
                    if proc.poll() is None:
                        proc.kill()
                proc.wait()
    if monotonic() >= deadline:
        raise TimeoutError("exact execution budget exhausted")
    if (directory / log_name).stat().st_size > MAX_CERTIFICATE_BYTES:
        raise ValueError("native log exceeds size limit")


def _read_certificate(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_CERTIFICATE_BYTES + 1)
    if len(payload) > MAX_CERTIFICATE_BYTES:
        raise ValueError("certificate exceeds size limit")
    return payload


def _verify(model, payload, checker, directory, deadline):
    bound = bind_certificate(model, payload)
    path = directory / "checked.vipr"
    path.write_bytes(payload)
    _run([checker, "checked.vipr"], directory, "vipr.log", deadline)
    output = (directory / "vipr.log").read_text(encoding="utf-8", errors="replace")
    lines = output.splitlines()
    success = ("Successfully verified infeasibility." in lines if bound.relation == "infeas"
               else any(line.startswith("Successfully verified optimal value range ") for line in lines)
               if bound.lower is not None else "Successfully verified." in lines)
    if not success or "Verification failed." in output:
        raise ValueError("checker did not positively confirm verification")
    if _read_certificate(path) != payload:
        raise ValueError("certificate changed during checking")
    if bound.relation == "infeas":
        return ExactSolveResult("infeasible", True, "VIPR checked original-model infeasibility")
    x = min(bound.solutions, key=lambda v: model.sign * model.objective(v), default=None)
    objective = None if x is None else model.objective(x)
    lower = None if bound.lower is None else model.sign * bound.lower + model.offset
    gap = None if objective is None or lower is None else model.sign * (objective - lower)
    if gap is not None and gap < 0:
        raise ValueError("verified bound conflicts with original feasible objective")
    if gap == 0:
        return ExactSolveResult("optimal", True, "exact original-model objective equals verified bound",
                                x, objective, lower, gap)
    if lower is not None:
        return ExactSolveResult("bound_verified", True, "finite bound verified; optimality not proved",
                                x, objective, lower, gap)
    return ExactSolveResult("unverified", False, "certificate has no finite optimality bound",
                            x, objective)


def _execute(problem, *, checker_executable, time_limit, evidence_directory,
             scip_executable=None, certificate=None):
    start = monotonic()
    deadline = start + _budget(time_limit)
    model = ExactModel.from_problem(problem)
    checker = _executable(checker_executable)
    solver = None if scip_executable is None else _executable(scip_executable)
    checker_hash = _digest(checker)
    solver_hash = "" if solver is None else _digest(solver)
    saved = None
    with TemporaryDirectory(prefix="solverpilot-exact-") as temporary:
        if evidence_directory is None:
            directory = Path(temporary)
        else:
            parent = Path(evidence_directory).resolve()
            parent.mkdir(parents=True, exist_ok=True)
            directory = Path(mkdtemp(prefix="exact-", dir=parent))
            saved = str(directory)
        payload = b""
        input_hash = ""
        try:
            if solver is not None:
                (directory / "problem.lp").write_text(model.lp_text(), encoding="ascii")
                remaining = max(0.001, deadline - monotonic())
                (directory / "exact.set").write_text(
                    'exact/enable = TRUE\ncertificate/filename = "proof.vipr"\n'
                    'certificate/maxfilesize = 32\npresolving/maxrounds = 0\n'
                    'presolving/maxrestarts = 0\nmisc/scaleobj = FALSE\n'
                    'separating/maxrounds = 0\nseparating/maxroundsroot = 0\n'
                    'conflict/enable = FALSE\n'
                    f'limits/time = {remaining:.9f}\nlimits/memory = 1024\n', encoding="ascii")
                _run([solver, "-c", "set load exact.set", "-c", "set separating emphasis off",
                      "-c", "read problem.lp", "-c", "optimize", "-c", "quit"],
                     directory, "scip.log", deadline)
                payload = _read_certificate(directory / "proof.vipr")
            else:
                payload = _read_certificate(certificate)
            input_hash = sha256(payload).hexdigest()
            payload = close_objective_bound(bind_certificate(model, payload), payload)
            result = _verify(model, payload, checker, directory, deadline)
            if _digest(checker) != checker_hash or (solver and _digest(solver) != solver_hash):
                raise ValueError("native executable changed during execution")
            if monotonic() >= deadline:
                raise TimeoutError("exact execution budget exhausted")
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            result = ExactSolveResult("unverified", False, str(exc))
        return replace(result, elapsed_s=monotonic() - start, problem_hash=model.data_hash,
                       certificate_sha256=sha256(payload).hexdigest() if payload else "",
                       input_certificate_sha256=input_hash,
                       checker_sha256=checker_hash, solver_sha256=solver_hash,
                       evidence_directory=saved)


def solve_exact(problem, *, scip_executable, checker_executable, time_limit=60.0,
                evidence_directory=None):
    """Solve LP/MILP exactly and verify a VIPR 1.0 proof against original assumptions.

    Coefficients are the exact binary64 values already stored in LinearProblem,
    not the decimal text from which those floats may have been created. Executables
    are caller-trusted native programs. Unsupported presolve mappings fail closed.
    This explicit API does not affect solve()/solve_production() routing.
    """
    return _execute(problem, scip_executable=scip_executable, checker_executable=checker_executable,
                    time_limit=time_limit, evidence_directory=evidence_directory)


def verify_exact_certificate(problem, certificate, *, checker_executable, time_limit=60.0,
                             evidence_directory=None):
    """Recheck certificate bytes and their model binding; never trust stored status."""
    return _execute(problem, certificate=certificate, checker_executable=checker_executable,
                    time_limit=time_limit, evidence_directory=evidence_directory)
