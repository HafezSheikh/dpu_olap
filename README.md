# Exploring In-Memory Processing with UPMEM Hardware

This repository contains the improved fork of [upmem/dpu_olap](https://github.com/upmem/dpu_olap).
It extends the original evaluation of SQL primitives on UPMEM’s Processing-In-Memory (PIM) hardware with new
profiling kernels, richer benchmarks, and data-generation fixes that make mismatch-driven analyses reliable.
The project accompanies the BSc thesis “Exploring In-Memory Processing with UPMEM Hardware: Performance Evaluation
in Executing Core Database Functions”.

---
## Table of Contents
1. [Overview](#overview)
2. [What’s New Compared to upstream](#whats-new-compared-to-upstream)
3. [Prerequisites](#prerequisites)
4. [Repository Layout](#repository-layout)
5. [Building](#building)
6. [Running Benchmarks](#running-benchmarks)
7. [Configuring Experiments](#configuring-experiments)
8. [Interpreting Outputs](#interpreting-outputs)
9. [Tips & Troubleshooting](#tips--troubleshooting)
10. [Acknowledgements](#acknowledgements)

---
## Overview
The code evaluates core database operators (filter, take, join, aggregation) on UPMEM DPUs. Each operator has
an Apache Arrow baseline and a DPU implementation. In this fork, special attention is given to the join operator
and Bloom filters: we added a Bloom profiling kernel, expanded the *paper_join* benchmark to emit meaningful
statistics, and fixed generators so requested mismatch rates match the actual data.

The harness is designed to run on the UPMEM functional simulator as well as real hardware (the simulator is used
in the thesis work). Because simulator timings are not trustworthy in absolute terms, we focus on instruction
counts and analytic models derived from the Bloom counters.

---
## What’s New Compared to upstream
- **Bloom profiling kernel** (`KernelBloomProfile`): measures total probes, Bloom skips, false positives, and matches on the DPUs without running the full join.
- **Paper join rewritten**: the host pipeline now partitions both sides, launches the profiling kernel, and reports counters (`total_probes`, `matches`, `mismatches`, `bloom_skipped`, `bloom_false_positives`).
- **Generator fix**: foreign keys marked “outside” are guaranteed to lie outside the primary-key domain, so the actual mismatch rate equals the requested ratio.
- **Hash-table miss bug fix**: DPU `ht_get` now stops at the first empty slot, avoiding O(capacity) scans on misses.
- **Benchmark sweeps expanded**: Paper join benchmark scans batch sizes, mismatch ratios, Bloom thresholds, Bloom bits per key, and DPU counts; invalid combinations and DPU errors are handled gracefully.
- **JSON result dump**: `paper_join_bench` mirrors the join benchmark and writes `paper_join_benchmark_results.json` for easy ingestion.
- Numerous quality-of-life updates (parameterized environment variables, improved logging, simulator-friendly guards).

---
## Prerequisites
1. **UPMEM SDK 2023.2** — newer SDK releases change runtime behaviour and cause build/run failures. Install the SDK and ensure `UPMEM_HOME` is set.
2. **CMake ≥ 3.24** and **Ninja** (or GNU Make if you prefer the provided targets).
3. **Host toolchain**: GCC/Clang with C++17 support.
4. **Python 3** (only if you plan to use the helper scripts).
5. CPM downloads third-party libraries (Google Benchmark, GoogleTest, Apache Arrow). To disable downloading and use local copies set `CPM_USE_LOCAL_PACKAGES=ON` with the appropriate environment variables.

> **Note:** The build tree caches CPM packages under `build/` and `~/.cache/CPM/`. If you switch SDK versions, wipe those caches before rebuilding.

---
## Repository Layout
```
.
├── baseline/                 # Python reference implementations
├── dpu/                      # DPU kernels (join, filter, take, aggr, partition)
├── host/                     # Host-side orchestrators and benchmarks
│   ├── join/                 # Join host code
│   ├── paper_join/           # Bloom profiling harness (this project’s focus)
│   └── ...
├── shared/                   # Common headers (UMQ, timers, logging)
├── scripts/                  # Helper scripts for larger evaluation campaigns
├── CMakeLists.txt / Makefile # Build entry points
└── README.md                 # This document
```

Important files for this fork:
- `dpu/shared/kernels/bloom_profile.{c,h}`: new profiling kernel
- `host/paper_join/paper_join_dpu.{cc,h}`: host pipeline executing the profiling kernel
- `host/paper_join/paper_join_benchmark.cc`: expanded benchmark with JSON output
- `host/generator/generator.cc`: corrected FK generator

---
## Building
```bash
cmake -S . -B build -G "Ninja Multi-Config"
cmake --build build --config Release --target all
```
Alternatively:
```bash
make build        # wraps the cmake invocation above
```
The build produces binaries under `build/bin/<Config>/` and DPU binaries under `build/dpu/<Config>/`.

If you modify DPU kernels or host code, rebuild the corresponding targets:
```bash
cmake --build build --config Release --target paper_join_bench
cmake --build build --config Release --target kernel-join
```

---
## Running Benchmarks
### Quick Start
1. **Paper join benchmark (Bloom profiling)**
   ```bash
   cd build/bin/Release
   ./paper_join_bench --benchmark_list_tests           # list scenarios
   ./paper_join_bench                                    # run full sweep
   ```
   Results appear in `paper_join_benchmark_results.json`.

2. **Join benchmark (for reference)**
   ```bash
   ./upmem-query-host --benchmark_filter=BM_JoinDpu
   ```
   Outputs `join_benchmark_results.json` via the wrapper in `host/main_benchmark.cc`.

### Customizing on the command line
You can filter or repeat benchmarks using standard Google Benchmark flags, e.g.:
- `--benchmark_filter=65536` to only run cases with 64K-row batches.
- `--benchmark_repetitions=3` to repeat each case.
- `--benchmark_min_time=0.5` to force minimum runtime per benchmark.

### Environment overrides
Many parameters can be overridden via environment variables at runtime:
- `NR_DPUS`, `SF`                — default number of DPUs / scale factor (join benchmark)
- `BLOOM_MIN_MISMATCH_RATE`     — threshold for enabling Bloom filters
- `BLOOM_BITS_PER_KEY`          — bits per key used when Bloom is active
- `FK_OUTSIDE_RATIO`            — generator override for mismatch ratio

`paper_join_bench` sets `BLOOM_MIN_MISMATCH_RATE` and `BLOOM_BITS_PER_KEY` automatically for each sweep element.
If you invoke `paper_join` CLI (`host/paper_join/main.cc`) directly, set them manually:
```bash
BLOOM_MIN_MISMATCH_RATE=0.2 BLOOM_BITS_PER_KEY=6 ./paper_join/main --batches=4 --batch_rows=65536 --fk_outside_ratio=0.1
```

---
## Configuring Experiments
1. **Modify sweep dimensions** in `host/paper_join/paper_join_benchmark.cc`:
   - `batches_list`
   - `batch_rows_list`
   - `fk_ratio_milli_list`
   - `bloom_thresh_milli_list`
   - `bloom_bits_per_key_list`
   - `dpus_list`
   All values are `int64_t`. `fk_ratio_milli_list` and `bloom_thresh_milli_list` are milli-units (e.g. 300 ≙ 0.3).

2. **Adjust data generation** in `host/paper_join/paper_join_dpu.cc` or `host/generator/generator.cc` if you need different schemas or mismatch patterns.

3. **Change MRAM limits** by editing `dpu/join/main.c` (e.g. `BUFFER_LENGTH`, `MAX_BLOOM_BITS`). Rebuild the DPU binaries afterwards.

4. **Add new counters** to `bloom_profile_counters_t` (shared/umq/kernels.h) if you need additional telemetry. Update the kernel and host accordingly.

---
## Interpreting Outputs
Each JSON record in `paper_join_benchmark_results.json` contains Google Benchmark fields plus the custom counters we set (averaged over iterations):
- `total_probes`
- `matches`
- `mismatches`
- `bloom_skipped`
- `bloom_false_positive`
- `bytes_per_second`, `items_per_second` (from Google Benchmark)

Use these to compute the analytic Bloom speedup model described in the thesis:
```
Time_without_bloom = matches * C_probe_hit + mismatches * C_probe_miss
Time_with_bloom    = matches * (C_bloom + C_probe_hit)
                    + (mismatches - bloom_skipped) * (C_bloom + C_probe_miss)
                    + bloom_skipped * C_bloom
Speedup = Time_without_bloom / Time_with_bloom
```
Derive `C_bloom`, `C_probe_hit`, `C_probe_miss` from instruction counts in dedicated micro-benchmarks (see `paper_join_dpu.cc`).

`join_benchmark_results.json` follows the same schema but for the original join pipeline.

---
## Tips & Troubleshooting
- **Use SDK 2023.2**: newer SDKs change runtime APIs and break scatter/gather transfers used in this project.
- **Functional simulator lag**: expect each large benchmark to take minutes. Reduce sweep sizes or use `--benchmark_filter` when iterating quickly.
- **“scatter/gather transfer prepare exceed max number of blocks”**: too many partitions vs DPUs. Reduce `batches` or align with `nr_dpus` in the benchmark.
- **“#partitions must be divisible by #DPUs”**: update the `dpus_list` or `batches_list` arrays to satisfy the constraint.
- **Stale CPM downloads**: delete `~/.cache/CPM` and `build/cpm-package-lock.cmake` if dependency builds fail after toolchain/SDK changes.
- **Overflow in MT/dpu buffers**: adjust `BUFFER_LENGTH`/`MAX_BLOOM_BITS` in DPU code, rebuild DPU binaries, and make sure MRAM consumption stays < 64 MiB.
- **JSON file overwritten**: both `join` and `paper_join` benchmarks write to their respective JSON files in the working directory. Copy them elsewhere if you want to keep historical runs.

---
## Acknowledgements
This work builds on the efforts of the original UPMEM evaluation team and adapts their code for deeper Bloom filter analysis.
It was extended as part of the BSc thesis “Exploring In-Memory Processing with UPMEM Hardware: Performance Evaluation in Executing Core Database Functions”.
