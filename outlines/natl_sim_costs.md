# North Atlantic case study: compute record

Measured, not projected. Sources: `data/osim/natl_sim/costs.csv` (per-step wall, CPU, peak RSS of the Python
process, written by `costlog.step`) joined to Slurm accounting (`costlog.sacct`: elapsed, allocation, billing,
node, CPU generation). Credits on Roar Collab `open`: 1 credit = 3,100,032 billing-minutes; a 1-CPU, <= 4 GB
task on `basic` bills 72 per minute (about 720 CPU-hours per credit).

Software: Python 3.11.15, Parcels 3.1.4, NumPy 2.4.6, SciPy 1.17.1, xarray 2026.4.0.

## Advection (step 3)

Setup per replicate:
- one continuous Parcels run, 2010-01-01 to 2025-12-31, RK4, dt = 1 h;
- 1000 m GLORYS12 velocity (daily, 1/12 deg), floats pinned at 1000 m;
- daily position output to node-local disk;
- 1,985 nominal deployments (2010-2025 launches plus the pre-2010 stock), advected without mortality until
  release + 12 y or the end of 2025.

Resources requested: 1 CPU, 3 GB, `open`/`basic`, 6 h limit. Array job 57184865, replicates 0-99.

Pilot (2 replicates, 2,208 deployments, `sc_default`, Ice Lake): 2.5 h wall per replicate, 2.3 h CPU, peak RSS
1.4 GB (Slurm MaxRSS reports the full 4 GB cgroup, page cache included).

| Quantity | Value |
|---|---|
| Replicates | 100 (all completed, no retries) |
| Wall per replicate (Slurm elapsed), median (range) | 1.45 h (1.24-1.88 h) |
| Advection step alone, median | 1.44 h; candidate table + selection < 1 min |
| CPU-hours, total (Python process) | 137.8 |
| Allocated core-hours (Slurm elapsed x 1 CPU) | 146.5 |
| Peak RSS (Python), max | 1.30 GB (Slurm MaxRSS 3.0 GB = the full cgroup incl. page cache) |
| Node CPU generation | Intel Sapphire Rapids (all 100 tasks), `open`/`basic` |
| Billing | 71 per minute; 624,143 billing-minutes = 0.201 credits |
| Array wall time, first start to last end | 2 h 16 min (all 100 tasks concurrent; 2026-11-10 09:30-11:46) |
