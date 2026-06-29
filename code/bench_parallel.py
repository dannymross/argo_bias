"""Compare 1-core (sequential) vs 2-core (parallel) runs of the same simulation.

Runs run_trajsim.py twice from the same config, varying only ntasks/local-cores.
Reads the per-task log files and prints a summary table.

Usage:
    python bench_parallel.py [--config configs/bench_parallel.json]
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

# ---- config --------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="code/configs/bench_parallel.json")
    return p.parse_args()


# ---- run -----------------------------------------------------------------

def run_case(python, config_path, ntasks, cores, prefix):
    """Run run_trajsim.py and return (wall_seconds, log_data_list)."""
    cmd = [
        python, os.path.join(SCRIPT_DIR, "run_trajsim.py"),
        "--config", config_path,
        "--ntasks", str(ntasks),
        "--local-cores", str(cores),
        "--prefix", prefix,
    ]
    print(f"\n{'─' * 60}")
    print(f"  cores={cores}  ntasks={ntasks}  prefix={prefix}")
    print(f"{'─' * 60}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT,
                            capture_output=False, text=True)
    wall = time.perf_counter() - t0

    if result.returncode != 0:
        sys.exit(f"run failed (exit {result.returncode})")

    # Collect the per-task logs written by run_trajsim.
    log_pattern = os.path.join(PROJECT_ROOT,
                               "data/virtualfleet", f"{prefix}_task*.log.json")
    logs = []
    for path in sorted(glob.glob(log_pattern)):
        with open(path) as f:
            logs.append(json.load(f))

    return wall, logs


# ---- report --------------------------------------------------------------

def summarise(label, wall, logs):
    task_walls = [d["wall_seconds"] for d in logs]
    peak_mbs   = [d["peak_memory_mb"] for d in logs]
    nfloats    = [d["nfloats_batch"] for d in logs]
    return {
        "label":            label,
        "cores":            len(logs),
        "ntasks":           logs[0]["ntasks"] if logs else 0,
        "nfloats_per_task": nfloats,
        "wall_total_s":     round(wall, 2),
        "task_wall_s":      [round(w, 2) for w in task_walls],
        "task_wall_min_s":  round(min(task_walls), 2),
        "task_wall_max_s":  round(max(task_walls), 2),
        "peak_mem_per_task_mb": [round(m, 1) for m in peak_mbs],
        "peak_mem_max_mb":  round(max(peak_mbs), 1),
        "peak_mem_total_mb": round(sum(peak_mbs), 1),
    }


def print_comparison(s1, s2):
    speedup = s1["wall_total_s"] / s2["wall_total_s"]
    mem_overhead = s2["peak_mem_total_mb"] - s1["peak_mem_total_mb"]

    rows = [
        ("Cores",                  s1["cores"],                   s2["cores"]),
        ("Batches (ntasks)",        s1["ntasks"],                  s2["ntasks"]),
        ("Floats per batch",        s1["nfloats_per_task"],        s2["nfloats_per_task"]),
        ("Wall time (total)",       f"{s1['wall_total_s']} s",     f"{s2['wall_total_s']} s"),
        ("Task wall (min/max)",
            f"{s1['task_wall_min_s']} / {s1['task_wall_max_s']} s",
            f"{s2['task_wall_min_s']} / {s2['task_wall_max_s']} s"),
        ("Peak RAM per task (max)", f"{s1['peak_mem_max_mb']} MB", f"{s2['peak_mem_max_mb']} MB"),
        ("Peak RAM total (sum)",    f"{s1['peak_mem_total_mb']} MB", f"{s2['peak_mem_total_mb']} MB"),
    ]

    col_w = [30, 22, 22]
    sep   = "─" * (sum(col_w) + 6)

    print(f"\n{'═' * len(sep)}")
    print(f"  Parallelism benchmark")
    print(f"{'═' * len(sep)}")
    header = f"  {'Metric':<{col_w[0]}}  {s1['label']:>{col_w[1]}}  {s2['label']:>{col_w[2]}}"
    print(header)
    print(f"  {sep}")
    for label, v1, v2 in rows:
        print(f"  {label:<{col_w[0]}}  {str(v1):>{col_w[1]}}  {str(v2):>{col_w[2]}}")
    print(f"  {sep}")
    print(f"  {'Speedup (wall time)':<{col_w[0]}}  {'':>{col_w[1]}}  {speedup:>{col_w[2]-2}.2f}x")
    print(f"  {'Extra RAM (2-core total)':<{col_w[0]}}  {'':>{col_w[1]}}  {mem_overhead:>+{col_w[2]-3}.0f} MB")
    print(f"{'═' * len(sep)}\n")


# ---- main ----------------------------------------------------------------

def main():
    args = parse_args()
    config_path = os.path.join(PROJECT_ROOT, args.config)

    with open(config_path) as f:
        cfg = json.load(f)

    python   = cfg.get("python", "python")
    outdir   = cfg.get("outdir", "data/virtualfleet")
    base_pfx = cfg.get("prefix", "bench")

    # Clean up any previous bench logs so glob picks up only this run's files.
    for pfx in (f"{base_pfx}_1core", f"{base_pfx}_2core"):
        for old in glob.glob(os.path.join(PROJECT_ROOT, outdir, f"{pfx}_task*.log.json")):
            os.remove(old)

    wall1, logs1 = run_case(python, config_path, ntasks=1, cores=1,
                            prefix=f"{base_pfx}_1core")
    wall2, logs2 = run_case(python, config_path, ntasks=2, cores=2,
                            prefix=f"{base_pfx}_2core")

    s1 = summarise("1 core  (sequential)", wall1, logs1)
    s2 = summarise("2 cores (parallel)",   wall2, logs2)
    print_comparison(s1, s2)

    # Write the comparison as JSON too.
    result_path = os.path.join(PROJECT_ROOT, outdir, f"{base_pfx}_comparison.json")
    with open(result_path, "w") as f:
        json.dump({"1core": s1, "2core": s2,
                   "speedup": round(s1["wall_total_s"] / s2["wall_total_s"], 3),
                   "extra_ram_mb": round(s2["peak_mem_total_mb"] - s1["peak_mem_total_mb"], 1)},
                  f, indent=2)
    print(f"Comparison saved -> {result_path}")


if __name__ == "__main__":
    main()
