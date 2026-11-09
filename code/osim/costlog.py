"""Per-step compute ledger: wall, CPU and peak memory of each pipeline step, joined to sacct billing."""

import contextlib
import io
import os
import resource
import socket
import subprocess
import time

import pandas as pd

CREDIT = 3_100_032   # billing-minutes per credit


@contextlib.contextmanager
def step(path, name, **meta):
    """Append one row to the CSV at path when the block finishes, with the replicate and item count from `meta`
    (which the block may update) in fixed columns."""
    t0, c0 = time.time(), time.process_time()
    yield meta
    env = os.environ.get
    row = dict(step=name, rep=meta.get("rep", ""), n=meta.get("n", ""), wall_s=round(time.time() - t0, 1), cpu_s=round(time.process_time() - c0, 1),
               maxrss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
               job=env("SLURM_ARRAY_JOB_ID") or env("SLURM_JOB_ID", ""), task=env("SLURM_ARRAY_TASK_ID", ""),
               account=env("SLURM_JOB_ACCOUNT", ""), partition=env("SLURM_JOB_PARTITION", ""),
               cpus=env("SLURM_CPUS_PER_TASK", ""), mem_mb=env("SLURM_MEM_PER_NODE", ""), host=socket.gethostname(),
               finished=pd.Timestamp.now().isoformat(timespec="seconds"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def sacct(jobs):
    """Elapsed, CPU, peak RSS and billing of every task of the given jobs (one row per task)."""
    out = subprocess.run(["sacct", "-j", ",".join(map(str, jobs)), "-n", "-P", "--units=G", "-o",
                          "JobID,JobName,State,ElapsedRaw,TotalCPU,MaxRSS,AllocTRES"],
                         capture_output=True, text=True, check=True).stdout
    d = pd.read_csv(io.StringIO(out), sep="|", header=None,
                    names=["id", "name", "state", "elapsed_s", "cpu", "maxrss", "tres"])
    rss = d[d.id.str.endswith(".batch")].assign(id=lambda x: x.id.str[:-6]).set_index("id").maxrss
    d = d[~d.id.str.contains(r"\.")].set_index("id")
    d["maxrss_gb"] = rss.reindex(d.index).str.rstrip("G").astype(float)
    d["billing"] = d.tres.str.extract(r"billing=(\d+)", expand=False).astype(float)
    d["billing_min"] = d.billing * d.elapsed_s / 60
    return d.drop(columns="tres").reset_index()


def ledger(path, md=None):
    """Per-step table (tasks, mean/max wall, CPU-h, peak RSS) plus a total row; written as markdown if md."""
    d = pd.read_csv(path)
    agg = dict(tasks=("wall_s", "size"), wall_mean_min=("wall_s", lambda x: x.mean() / 60),
               wall_max_min=("wall_s", lambda x: x.max() / 60), cpu_h=("cpu_s", lambda x: x.sum() / 3600),
               maxrss_gb=("maxrss_gb", "max"))
    t = pd.concat([d.groupby("step").agg(**agg), d.agg(**agg).T.set_axis(["total"])]).round(2)
    if md:
        with open(md, "w") as f:
            f.write(t.to_markdown() + "\n")
    return t
