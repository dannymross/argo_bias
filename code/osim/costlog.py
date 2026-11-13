"""Per-step compute ledger: wall, CPU and peak memory of each pipeline step, joined to sacct billing."""

import contextlib
import io
import os
import resource
import socket
import subprocess
import time

import numpy as np
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
                          "JobID,JobName,State,ElapsedRaw,TotalCPU,MaxRSS,AllocTRES,NodeList"],
                         capture_output=True, text=True, check=True).stdout
    d = pd.read_csv(io.StringIO(out), sep="|", header=None,
                    names=["id", "name", "state", "elapsed_s", "cpu", "maxrss", "tres", "node"])
    rss = d[d.id.str.endswith(".batch")].assign(id=lambda x: x.id.str[:-6]).set_index("id").maxrss
    d = d[~d.id.str.contains(r"\.")].set_index("id")
    d["maxrss_gb"] = rss.reindex(d.index).str.rstrip("G").astype(float)
    d["cpus"] = d.tres.str.extract(r"cpu=(\d+)", expand=False).astype(float)
    d["billing"] = d.tres.str.extract(r"billing=(\d+)", expand=False).astype(float)
    d["billing_min"] = d.billing * d.elapsed_s / 60
    d["cpu_arch"] = d.node.map({n: node_arch(n) for n in d.node.dropna().unique()})
    return d.drop(columns="tres").reset_index()


def node_arch(node):
    """CPU generation feature of a node (e.g. icelake), from its Slurm features."""
    out = subprocess.run(["scontrol", "show", "node", node], capture_output=True, text=True).stdout
    f = next((w.split("=", 1)[1] for w in out.split() if w.startswith("AvailableFeatures=")), "")
    return ",".join(x for x in f.split(",") if x not in ("sc", "bc", "standard", "basic", "open")) or None


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


CREDIT_BILLING_MIN = 3_100_032   # Roar Collab: billing-minutes per credit


def _secs(x):
    """sacct TotalCPU ([D-]HH:MM:SS[.mmm] or MM:SS.mmm) in seconds."""
    d, _, x = x.rpartition("-") if "-" in x else ("0", "", x)
    p = [float(v) for v in x.split(":")]
    return int(d) * 86400 + sum(v * 60 ** i for i, v in enumerate(reversed(p)))


def job_ledger(jobs_csv, md=None):
    """Per-stage compute from Slurm accounting for the jobs registered in jobs_csv (job, stage, purpose, status,
    account taken from sacct): tasks by state, allocated core-h, CPU-h, peak RSS, billing-minutes and credits (credits
    only for the `open` account; allocation accounts are not charged); production and pilot/superseded kept apart."""
    reg = pd.read_csv(jobs_csv, dtype={"job": str})
    d = sacct(reg.job)
    acct = subprocess.run(["sacct", "-j", ",".join(reg.job), "-X", "-n", "-P", "-o", "JobID,Account"],
                          capture_output=True, text=True, check=True).stdout
    acct = pd.read_csv(io.StringIO(acct), sep="|", header=None, names=["id", "account"]).set_index("id").account
    d = d.assign(job=d.id.str.split("_").str[0], account=d.id.map(acct), cpu_s=d.cpu.map(_secs),
                 core_h=d.elapsed_s * d.cpus / 3600, done=d.state.eq("COMPLETED"),
                 failed=d.state.str.startswith(("FAILED", "TIMEOUT", "OUT_OF")), cancelled=d.state.str.startswith("CANCELLED"))
    d = d.merge(reg, on="job")
    d["credits"] = np.where(d.account.eq("open"), d.billing_min / CREDIT_BILLING_MIN, 0.0)
    agg = dict(jobs=("job", "nunique"), tasks=("id", "size"), done=("done", "sum"), failed=("failed", "sum"),
               cancelled=("cancelled", "sum"), core_h=("core_h", "sum"), cpu_h=("cpu_s", lambda x: x.sum() / 3600),
               maxrss_gb=("maxrss_gb", "max"), billing_min=("billing_min", "sum"), credits=("credits", "sum"))
    t = pd.concat([d.groupby(["status", "stage", "account"]).agg(**agg),
                   d.assign(stage="all", account="all").groupby(["status", "stage", "account"]).agg(**agg),
                   d.assign(status="total", stage="", account="").groupby(["status", "stage", "account"]).agg(**agg)]
                  ).round(3)
    if md:
        with open(md, "w") as f:
            f.write(f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} by costlog.job_ledger from {os.path.basename(jobs_csv)}"
                    " and sacct. core_h = elapsed x allocated CPUs; credits only for `open`.\n\n"
                    + t.to_markdown() + "\n")
    return t, d


BASIC_RATE = dict(cpu=71.76, mem_gb=17.94)   # billing weights per minute on the basic partition


def task_costs(jobs_csv, status=("production",)):
    """One row per COMPLETED task of the registered jobs with the given status: account, wall, allocated CPUs and
    memory, CPU time, peak RSS, and credits: billed for `open`, and for other accounts the `open`/basic equivalent
    max(CPU x 71.76, GB x 17.94) billing-min per minute (Roar Collab billing is the max of the two)."""
    reg = pd.read_csv(jobs_csv, dtype={"job": str})
    reg = reg[reg.status.isin(status)]
    out = subprocess.run(["sacct", "-j", ",".join(reg.job), "-n", "-P", "--units=G", "-o",
                          "JobID,Account,Partition,State,ElapsedRaw,TotalCPU,MaxRSS,AllocTRES"],
                         capture_output=True, text=True, check=True).stdout
    d = pd.read_csv(io.StringIO(out), sep="|", header=None,
                    names=["id", "account", "partition", "state", "elapsed_s", "cpu", "maxrss", "tres"], dtype=str)
    rss = d[d.id.str.endswith(".batch")].assign(id=lambda x: x.id.str[:-6]).set_index("id").maxrss
    d = d[~d.id.str.contains(r"\.")].query("state == 'COMPLETED'").copy()
    mem = d.tres.str.extract(r"mem=([\d.]+)([MGT])")
    d = d.assign(job=d.id.str.split("_").str[0], elapsed_s=d.elapsed_s.astype(float),
                 cpus=d.tres.str.extract(r"cpu=(\d+)", expand=False).astype(float),
                 mem_gb=mem[0].astype(float) * mem[1].map(dict(M=1 / 1024, G=1, T=1024)),
                 billing=d.tres.str.extract(r"billing=(\d+)", expand=False).astype(float),
                 cpu_h=d.cpu.map(_secs) / 3600, maxrss_gb=rss.reindex(d.id).str.rstrip("G").astype(float).values)
    d["core_h"] = d.elapsed_s * d.cpus / 3600
    rate = np.maximum(d.cpus * BASIC_RATE["cpu"], d.mem_gb * BASIC_RATE["mem_gb"])
    d["credits"] = np.where(d.account.eq("open"), d.billing * d.elapsed_s / 60, rate * d.elapsed_s / 60) / CREDIT_BILLING_MIN
    return d.merge(reg, on="job").drop(columns=["tres", "cpu", "maxrss"])
