from __future__ import annotations

from pathlib import Path
import shlex


def render_slurm_array(
    *,
    spec_path: str,
    dataset_dir: str,
    manifest: str,
    reference: str | None,
    output_dir: str,
    backends: tuple[str, ...],
    shards: int,
    repetitions: int,
    time_limit_s: float | None,
    job_name: str = "solverpilot-bench",
    cpus_per_task: int = 1,
    memory: str = "8G",
    walltime: str = "02:00:00",
    thread_env_limit: int | None = 1,
    solver_threads: int | None = None,
    worker_python_mode: str = "no_site",
) -> str:
    if shards < 1:
        raise ValueError("shards must be >= 1")
    backend_args = " ".join(f"--backend {shlex.quote(b)}" for b in backends)
    ref_arg = "" if reference is None else f" --reference {shlex.quote(reference)}"
    limit_arg = "" if time_limit_s is None else f" --time-limit-s {time_limit_s}"
    thread_env_arg = " --thread-env-uncontrolled" if thread_env_limit is None else f" --thread-env-limit {thread_env_limit}"
    solver_threads_arg = "" if solver_threads is None else f" --solver-threads {solver_threads}"
    worker_mode_arg = f" --worker-python-mode {shlex.quote(worker_python_mode)}"
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{shards - 1}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
#SBATCH --time={walltime}
set -euo pipefail
mkdir -p {shlex.quote(output_dir)}
solverpilot-bench run \\
  --dataset-dir {shlex.quote(dataset_dir)} \\
  --manifest {shlex.quote(manifest)}{ref_arg} \\
  {backend_args} \\
  --repetitions {repetitions}{limit_arg}{thread_env_arg}{solver_threads_arg} \\
  --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count {shards} \\
  --output {shlex.quote(output_dir)}/shard-${{SLURM_ARRAY_TASK_ID}}.jsonl
"""


def write_slurm_array(path: str | Path, **kwargs) -> str:
    text = render_slurm_array(**kwargs)
    Path(path).write_text(text, encoding="utf-8")
    return text
