#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

#include "dpuext/api.h"
#include "generator/generator.h"
#include "paper_join_dpu.h"

namespace {

struct BenchmarkParams {
  int64_t batches;
  int64_t batch_rows;
  double fk_outside_ratio;
  double bloom_threshold;
  int nr_dpus;
};

BenchmarkParams ParseState(const benchmark::State& state) {
  BenchmarkParams params;
  params.batches = state.range(0);
  params.batch_rows = state.range(1);
  params.fk_outside_ratio = static_cast<double>(state.range(2)) / 1000.0;
  params.bloom_threshold = static_cast<double>(state.range(3)) / 1000.0;
  params.nr_dpus = static_cast<int>(state.range(4));
  return params;
}

void BM_PaperJoin(benchmark::State& state) {
  auto params = ParseState(state);
  params.fk_outside_ratio = std::clamp(params.fk_outside_ratio, 0.0, 1.0);
  if (params.batches <= 0 || params.batch_rows <= 0 || params.nr_dpus <= 0) {
    state.SkipWithError("invalid benchmark parameters");
    return;
  }

  setenv("BLOOM_MIN_MISMATCH_RATE",
         std::to_string(params.bloom_threshold).c_str(), 1);

  uint64_t total_sum = 0;
  uint64_t matches_sum = 0;
  uint64_t mismatches_sum = 0;
  uint64_t bloom_sum = 0;
  uint64_t bloom_fp_sum = 0;
  bool first_iteration = true;

  for (auto _ : state) {
    state.PauseTiming();

    arrow::random::RandomArrayGenerator rng_right(state.iterations() + 42);
    arrow::random::RandomArrayGenerator rng_left(state.iterations() + 1337);

    auto right_schema =
        arrow::schema({arrow::field("payload", arrow::uint32(), false)});
    auto right_batches_base = upmemeval::generator::MakeRandomRecordBatches(
        rng_right, right_schema, static_cast<int>(params.batches),
        static_cast<int>(params.batch_rows));
    auto right_pk_column = upmemeval::generator::MakeIndexColumn(
                                   static_cast<int>(params.batches),
                                   static_cast<int>(params.batch_rows))
                               .ValueOrDie();
    auto right_batches = upmemeval::generator::AddColumn(
        "pk", right_batches_base, right_pk_column);
    auto right_schema_with_pk = right_batches[0]->schema();

    auto left_schema =
        arrow::schema({arrow::field("payload", arrow::uint32(), false)});
    auto left_batches_base = upmemeval::generator::MakeRandomRecordBatches(
        rng_left, left_schema, static_cast<int>(params.batches),
        static_cast<int>(params.batch_rows));
    std::pair<uint64_t, uint64_t> counts{0, 0};
    auto left_fk_column = upmemeval::generator::MakeForeignKeyColumn(
                                   rng_left, static_cast<uint32_t>(params.batch_rows),
                                   static_cast<int32_t>(params.batches),
                                   static_cast<int32_t>(params.batch_rows),
                                   params.fk_outside_ratio, &counts)
                               .ValueOrDie();
    auto left_batches = upmemeval::generator::AddColumn(
        "fk", left_batches_base, left_fk_column);
    auto left_schema_with_fk = left_batches[0]->schema();

    double actual_ratio = 0.0;
    uint64_t total_pairs = counts.first + counts.second;
    if (total_pairs > 0) {
      actual_ratio = static_cast<double>(counts.second) / static_cast<double>(total_pairs);
    }

    if (first_iteration && state.thread_index() == 0) {
      std::cout << "[Generator] batches=" << params.batches
                << " batch_rows=" << params.batch_rows
                << " fk_outside_ratio_requested=" << params.fk_outside_ratio
                << " actual=" << actual_ratio << std::endl;
    }

    auto system = dpu::DpuSet::allocate(
        params.nr_dpus, "backend=simulator,nrJobsPerRank=256,sgXferEnable=true");
    if (first_iteration && state.thread_index() == 0) {
      std::cout << "Allocated DPUs: " << system.dpus().size() << std::endl;
    }

    if (first_iteration && state.thread_index() == 0) {
      first_iteration = false;
    }
    upmemeval::paper_join::PaperJoinDpu join(system, left_schema_with_fk,
                                             right_schema_with_pk, left_batches,
                                             right_batches);

    state.ResumeTiming();

    try {
      auto status = join.Prepare();
      if (!status.ok()) {
        state.SkipWithError(status.ToString().c_str());
        return;
      }

      auto result = join.Run();
      if (!result.ok()) {
        state.SkipWithError(result.status().ToString().c_str());
        return;
      }

      auto metrics = result.ValueOrDie();
      total_sum += metrics.total_probes;
      matches_sum += metrics.matches;
      mismatches_sum += metrics.mismatches;
      bloom_sum += metrics.bloom_skipped;
      bloom_fp_sum += metrics.bloom_false_positives;
    } catch (const dpu::DpuError& e) {
      state.SkipWithError(e.what());
      return;
    }
  }

  uint64_t rows_per_iter = static_cast<uint64_t>(params.batches) *
                           static_cast<uint64_t>(params.batch_rows);
  state.SetItemsProcessed(rows_per_iter * state.iterations());
  state.SetBytesProcessed(state.items_processed() * sizeof(uint32_t));
  state.counters["total_probes"] =
      benchmark::Counter(static_cast<double>(total_sum),
                         benchmark::Counter::kAvgIterations);
  state.counters["matches"] =
      benchmark::Counter(static_cast<double>(matches_sum),
                         benchmark::Counter::kAvgIterations);
  state.counters["mismatches"] =
      benchmark::Counter(static_cast<double>(mismatches_sum),
                         benchmark::Counter::kAvgIterations);
  state.counters["bloom_skipped"] =
      benchmark::Counter(static_cast<double>(bloom_sum),
                         benchmark::Counter::kAvgIterations);
  state.counters["bloom_false_positive"] =
      benchmark::Counter(static_cast<double>(bloom_fp_sum),
                         benchmark::Counter::kAvgIterations);
}

static void ApplyPaperJoinArgs(benchmark::internal::Benchmark* bench) {
  const std::array<int64_t, 1> batches_list = {2};
  const std::array<int64_t, 3> batch_rows_list = {65536, 131072, 262144};
  const std::array<int64_t, 4> fk_ratio_milli_list = {100, 300, 600, 900};
  const std::array<int64_t, 2> bloom_thresh_milli_list = {0};
  const std::array<int64_t, 1> dpus_list = {2};

  for (auto batches : batches_list) {
    for (auto batch_rows : batch_rows_list) {
      for (auto fk_ratio : fk_ratio_milli_list) {
        for (auto bloom_thresh : bloom_thresh_milli_list) {
          for (auto nr_dpus : dpus_list) {
            if (batches % nr_dpus != 0) {
              continue;
            }
            bench->Args({batches, batch_rows, fk_ratio, bloom_thresh, nr_dpus});
          }
        }
      }
    }
  }
}

static void RegisterPaperJoinBenchmarks() {
  benchmark::RegisterBenchmark(
      "BM_PaperJoin",
      [](benchmark::State& state) { BM_PaperJoin(state); })
      ->ArgNames({"batches", "batch_rows", "fk_ratio_milli",
                  "bloom_thresh_milli", "nr_dpus"})
      ->Apply(ApplyPaperJoinArgs)
      ->UseRealTime()
      ->Iterations(1)
      ->Unit(benchmark::kMillisecond);
}

static const bool kBenchRegistered = []() {
  RegisterPaperJoinBenchmarks();
  return true;
}();

}  // namespace

BENCHMARK_MAIN();
