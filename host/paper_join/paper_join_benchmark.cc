#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "dpuext/api.h"
#include "generator/generator.h"
#include "join/join_profiles.h"
#include "paper_join_dpu.h"

namespace {

struct BenchmarkParams {
  int64_t batches;
  int64_t batch_rows;
  double fk_outside_ratio;
  double bloom_threshold;
  int64_t bloom_bits_per_key;
  int nr_dpus;
};

BenchmarkParams ParseState(const benchmark::State& state) {
  BenchmarkParams params;
  params.batches = state.range(0);
  params.batch_rows = state.range(1);
  params.fk_outside_ratio = static_cast<double>(state.range(2)) / 1000.0;
  params.bloom_threshold = static_cast<double>(state.range(3)) / 1000.0;
  params.bloom_bits_per_key = state.range(4);
  params.nr_dpus = static_cast<int>(state.range(5));
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
  setenv("BLOOM_BITS_PER_KEY",
         std::to_string(params.bloom_bits_per_key).c_str(), 1);

  uint64_t total_sum = 0;
  uint64_t matches_sum = 0;
  uint64_t mismatches_sum = 0;
  uint64_t bloom_sum = 0;
  uint64_t bloom_fp_sum = 0;
  bool first_iteration = true;

  for (auto _ : state) {
    state.PauseTiming();

    auto scenario = upmemeval::join::ResolveScenarioConfig(
        std::optional<double>(params.fk_outside_ratio));
    upmemeval::join::GeneratedJoinData generated;
    try {
      generated = upmemeval::join::GenerateJoinData(
          scenario, params.batches, params.batch_rows, params.batch_rows);
    } catch (const std::exception& e) {
      state.SkipWithError(e.what());
      return;
    }

    auto left_schema_with_fk = generated.left_schema;
    auto right_schema_with_pk = generated.right_schema;
    auto& left_batches = generated.left_batches;
    auto& right_batches = generated.right_batches;

    double actual_ratio = 0.0;
    double null_ratio = 0.0;
    double hot_ratio = 0.0;
    double hot_miss_ratio = 0.0;
    double considered = static_cast<double>(generated.stats.inside + generated.stats.outside);
    double total = static_cast<double>(generated.stats.total());
    if (considered > 0.0) {
      actual_ratio = static_cast<double>(generated.stats.outside) / considered;
    }
    if (total > 0.0) {
      null_ratio = static_cast<double>(generated.stats.nulls) / total;
    }
    if (generated.stats.inside > 0) {
      hot_ratio = static_cast<double>(generated.stats.hot) /
                  static_cast<double>(generated.stats.inside);
    }
    if (generated.stats.outside > 0) {
      hot_miss_ratio = static_cast<double>(generated.stats.miss_hot) /
                       static_cast<double>(generated.stats.outside);
    }

    if (first_iteration && state.thread_index() == 0) {
      std::cout << "[Generator(" << "paper," << upmemeval::join::ProfileTag(scenario.profile)
                << ")] batches=" << params.batches
                << " batch_rows=" << params.batch_rows
                << " fk_outside_ratio_requested=" << params.fk_outside_ratio
                << " actual=" << actual_ratio
                << " null_ratio=" << null_ratio
                << " hot_share=" << hot_ratio
                << " hot_miss_share=" << hot_miss_ratio << std::endl;
      std::cout << "[Generator] profile_desc=" << upmemeval::join::ProfileDescription(scenario.profile)
                << std::endl;
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
  const std::array<int64_t, 1> bloom_thresh_milli_list = {0};
  const std::array<int64_t, 3> bloom_bits_per_key_list = {3, 6, 9};
  const std::array<int64_t, 1> dpus_list = {2};

  for (auto batches : batches_list) {
    for (auto batch_rows : batch_rows_list) {
      for (auto fk_ratio : fk_ratio_milli_list) {
        for (auto bloom_thresh : bloom_thresh_milli_list) {
          for (auto bloom_bits_per_key : bloom_bits_per_key_list) {
            for (auto nr_dpus : dpus_list) {
              if (batches % nr_dpus != 0) {
                continue;
              }
              bench->Args({batches, batch_rows, fk_ratio, bloom_thresh,
                           bloom_bits_per_key, nr_dpus});
            }
          }
        }
      }
    }
  }
}

static auto* const kPaperJoinBenchmarkRegistration =
    benchmark::RegisterBenchmark("BM_PaperJoin",
                                 [](benchmark::State& state) { BM_PaperJoin(state); })
        ->ArgNames({"batches", "batch_rows", "fk_ratio_milli",
                    "bloom_thresh_milli", "bloom_bits_per_key", "nr_dpus"})
        ->Apply(ApplyPaperJoinArgs)
        ->UseRealTime()
        ->Iterations(1)
        ->Unit(benchmark::kMillisecond);

}  // namespace

int main(int argc, char** argv) {
  (void)kPaperJoinBenchmarkRegistration;
  std::vector<std::string> forced_flags = {
      "--benchmark_out=paper_join_benchmark_results.json",
      "--benchmark_out_format=json"};
  std::vector<char*> argv_extended;
  argv_extended.reserve(argc + forced_flags.size());
  for (int i = 0; i < argc; ++i) {
    argv_extended.push_back(argv[i]);
  }
  for (auto& flag : forced_flags) {
    argv_extended.push_back(const_cast<char*>(flag.c_str()));
  }
  int argc_extended = static_cast<int>(argv_extended.size());
  benchmark::Initialize(&argc_extended, argv_extended.data());
  if (::benchmark::ReportUnrecognizedArguments(argc_extended, argv_extended.data())) {
    return 1;
  }

  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
