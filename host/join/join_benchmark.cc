#include <arrow/api.h>
#include <benchmark/benchmark.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

#include "dpuext/api.h"
#include "generator/generator.h"
#include "timer/timer.h"

#include "umq/cflags.h"
#include "umq/kernels.h"

#include "system/system.h"

#include "join_dpu.h"
#include "join_native.h"

using namespace dpu;

namespace upmemeval {

using namespace generator;
using namespace timer;

namespace join {

namespace {

struct ScopedEnv {
  struct Entry {
    std::string name;
    std::optional<std::string> previous;
  };

  explicit ScopedEnv(std::vector<std::pair<std::string, std::string>> vars) {
    entries_.reserve(vars.size());
    for (auto& kv : vars) {
      Entry entry{kv.first, std::nullopt};
      if (const char* existing = std::getenv(kv.first.c_str())) {
        entry.previous = existing;
      }
      setenv(kv.first.c_str(), kv.second.c_str(), 1);
      entries_.push_back(std::move(entry));
    }
  }

  ~ScopedEnv() {
    for (const auto& entry : entries_) {
      if (entry.previous.has_value()) {
        setenv(entry.name.c_str(), entry.previous->c_str(), 1);
      } else {
        unsetenv(entry.name.c_str());
      }
    }
  }

 private:
  std::vector<Entry> entries_;
};

bool IsPowerOfTwo(int64_t value) { return value > 0 && (value & (value - 1)) == 0; }

int64_t RoundUpToPowerOfTwo(int64_t value) {
  if (value <= 1) {
    return 2;
  }
  int64_t power = 1;
  while (power < value) {
    power <<= 1;
  }
  return power;
}

int64_t FloorPowerOfTwo(int64_t value) {
  if (value <= 1) {
    return 2;
  }
  int64_t power = 2;
  while ((power << 1) <= value) {
    power <<= 1;
  }
  return power;
}

int64_t NormalizeDpuCount(int64_t requested, int64_t max_dpus) {
  requested = std::max<int64_t>(2, requested);
  requested = RoundUpToPowerOfTwo(requested);
  int64_t max_cap = FloorPowerOfTwo(std::max<int64_t>(2, max_dpus));
  return std::min<int64_t>(requested, max_cap);
}

struct JoinBenchmarkConfig {
  int64_t batches;
  int64_t left_rows;
  int64_t right_rows;
  int64_t dpus;
  double bloom_threshold;
};

int64_t EncodeBloomThreshold(double value) {
  return static_cast<int64_t>(std::llround(value * 1000.0));
}

double DecodeBloomThreshold(int64_t encoded_value) {
  return static_cast<double>(encoded_value) / 1000.0;
}

std::vector<JoinBenchmarkConfig> BuildJoinDpuBenchmarkConfigs() {
  std::vector<JoinBenchmarkConfig> configs;

  const int64_t hardware_limit = 64;
  const int64_t max_dpus = hardware_limit;
  const std::array<int64_t, 3> dpus_values = {4, 8, 16};
  const std::array<int64_t, 2> scale_multipliers = {1, 2};
  const std::array<int64_t, 2> batch_rows = {64LL << 10, 128LL << 10};
  const std::array<double, 4> bloom_thresholds = {0.30, 0.60, 0.90, 1.10};

  for (int64_t dpus : dpus_values) {
    int64_t normalized_dpus = NormalizeDpuCount(dpus, max_dpus);
    if (normalized_dpus != dpus) {
      continue;
    }
    for (int64_t multiplier : scale_multipliers) {
      int64_t target_batches = dpus * multiplier;
      int64_t batches = RoundUpToPowerOfTwo(std::max<int64_t>(target_batches, dpus));
      if (!IsPowerOfTwo(batches) || batches % dpus != 0) {
        continue;
      }
      for (int64_t rows : batch_rows) {
        for (double threshold : bloom_thresholds) {
          JoinBenchmarkConfig cfg{batches, rows, rows, dpus, threshold};
          if (std::find_if(configs.begin(), configs.end(),
                           [&](const JoinBenchmarkConfig& existing) {
                             return existing.batches == cfg.batches &&
                                    existing.left_rows == cfg.left_rows &&
                                    existing.right_rows == cfg.right_rows &&
                                    existing.dpus == cfg.dpus &&
                                    std::abs(existing.bloom_threshold - cfg.bloom_threshold) <
                                        1e-6;
                           }) == configs.end()) {
            configs.push_back(cfg);
          }
        }
      }
    }
  }

  std::sort(configs.begin(), configs.end(), [](const JoinBenchmarkConfig& a,
                                               const JoinBenchmarkConfig& b) {
    if (a.dpus != b.dpus) return a.dpus < b.dpus;
    if (a.batches != b.batches) return a.batches < b.batches;
    if (a.left_rows != b.left_rows) return a.left_rows < b.left_rows;
    return a.bloom_threshold < b.bloom_threshold;
  });

  return configs;
}

}  // namespace

template <class T>
void BM_Join(benchmark::State& state, T& joiner) {
  uint64_t total_rows = 0;
  auto nr_dpus = state.range(3);
  auto batches = state.range(0);
  auto left_rows = state.range(1);
  double bloom_threshold = DecodeBloomThreshold(state.range(4));
  std::ostringstream threshold_stream;
  threshold_stream << std::fixed << std::setprecision(2) << bloom_threshold;
  ScopedEnv scoped_env({
      {"NR_DPUS", std::to_string(nr_dpus)},
      {"SF", std::to_string(batches)},
      {"BLOOM_MIN_MISMATCH_RATE", threshold_stream.str()},
  });
  std::ostringstream label;
  label << "dpus=" << nr_dpus << ",sf=" << batches << ",batch_rows=" << left_rows
        << ",thr=" << threshold_stream.str();
  state.SetLabel(label.str());
  double nr_ranks = (nr_dpus + 63) / 64.0;
  for (auto _ : state) {
    auto prepared = joiner.Prepare();
    if (!prepared.ok()) {
      auto err = std::string("Prepare failed with error: ") + prepared.ToString();
      state.SkipWithError(err.c_str());
      break;
    }

    arrow::Result<std::shared_ptr<arrow::Table>> result = joiner.Run();
    if (!result.ok()) {
      auto err = std::string("Run failed with error: ") + result.status().ToString();
      state.SkipWithError(err.c_str());
      break;
    }

    total_rows += result->get()->num_rows();

    auto timers = joiner.Timers();
    if (timers != nullptr) {
      for (auto& r : timers->get()) {
        auto name = r.first;
        auto timer = r.second;
        auto& counter = state.counters[name];
        counter.value +=
            std::chrono::duration_cast<std::chrono::milliseconds>(timer->Result())
                .count() /
            nr_ranks;
      }
    }
  }

  std::cout << "Total Rows: " << total_rows << " Iterations: " << state.iterations()
            << std::endl;
}

class PartitionedBatchGeneratorFixture : public benchmark::Fixture {
 public:
  PartitionedBatchGeneratorFixture() : rng_(42) {}

  void SetUp(::benchmark::State& state) override {
    auto num_batches = state.range(0);
    auto left_batch_size = state.range(1);
    auto right_batch_size = state.range(2);
    assert(num_batches > 0);
    assert(left_batch_size > 0);
    assert(right_batch_size > 0);

    if (right_batches_.size() != static_cast<size_t>(num_batches) ||
        right_batches_[0]->num_rows() != right_batch_size) {
      auto schema =
          arrow::schema({arrow::field("x", arrow::uint32(), /*nullable=*/false)});
      auto right_batches =
          generator::MakeRandomRecordBatches(rng_, schema, num_batches, right_batch_size);
      auto right_pk_column = generator::MakeIndexColumn(num_batches, right_batch_size);
      right_batches_ =
          generator::AddColumn("pk", right_batches, right_pk_column.ValueOrDie());
      right_schema_ = right_batches_[0]->schema();
    }

    if (left_batches_.size() != static_cast<size_t>(num_batches) ||
        left_batches_[0]->num_rows() != left_batch_size) {
      auto schema =
          arrow::schema({arrow::field("y", arrow::uint32(), /*nullable=*/false)});
      auto left_batches =
          generator::MakeRandomRecordBatches(rng_, schema, num_batches, left_batch_size);
      auto left_fk_column = generator::MakeForeignKeyColumn(rng_, right_batch_size,
                                                            num_batches, left_batch_size);
      left_batches_ =
          generator::AddColumn("fk", left_batches, left_fk_column.ValueOrDie());
      left_schema_ = left_batches_[0]->schema();
    }
  }

  void TearDown(::benchmark::State&) override {
    left_batches_.clear();
    right_batches_.clear();
  }

  int64_t total_items() {
    auto left_batch_total_size = left_batches_.size() * left_batches_[0]->num_rows() *
                                 left_batches_[0]->num_columns();
    auto right_batch_total_size = right_batches_.size() * right_batches_[0]->num_rows() *
                                  right_batches_[0]->num_columns();
    return left_batch_total_size + right_batch_total_size;
  }

  int64_t total_bytes() { return total_items() * sizeof(uint32_t); }

 protected:
  arrow::random::RandomArrayGenerator rng_;
  std::shared_ptr<arrow::Schema> left_schema_;
  std::shared_ptr<arrow::Schema> right_schema_;
  arrow::RecordBatchVector left_batches_;
  arrow::RecordBatchVector right_batches_;
};

BENCHMARK_DEFINE_F(PartitionedBatchGeneratorFixture, BM_JoinNative)
(benchmark::State& state) {
  bool partitioned = state.range(3);
  auto batches = state.range(0);
  auto left_rows = state.range(1);
  double bloom_threshold = DecodeBloomThreshold(state.range(4));
  std::ostringstream threshold_stream;
  threshold_stream << std::fixed << std::setprecision(2) << bloom_threshold;
  ScopedEnv scoped_env({
      {"SF", std::to_string(batches)},
      {"BLOOM_MIN_MISMATCH_RATE", threshold_stream.str()},
  });
  std::ostringstream label;
  label << "sf=" << batches << ",batch_rows=" << left_rows << ",thr="
        << threshold_stream.str() << (partitioned ? ",partitioned" : ",single");
  state.SetLabel(label.str());
  JoinNative joiner{left_schema_, right_schema_, left_batches_, right_batches_, partitioned};
  BM_Join<>(state, joiner);
  state.SetItemsProcessed(total_items());
  state.SetBytesProcessed(total_bytes());
}

BENCHMARK_DEFINE_F(PartitionedBatchGeneratorFixture, BM_JoinDpu)
(benchmark::State& state) {
  auto nr_dpus = state.range(3);
  auto batches = state.range(0);
  auto left_rows = state.range(1);
  double bloom_threshold = DecodeBloomThreshold(state.range(4));
  std::ostringstream threshold_stream;
  threshold_stream << std::fixed << std::setprecision(2) << bloom_threshold;
  ScopedEnv scoped_env({
      {"NR_DPUS", std::to_string(nr_dpus)},
      {"SF", std::to_string(batches)},
      {"BLOOM_MIN_MISMATCH_RATE", threshold_stream.str()},
  });
  std::ostringstream label;
  label << "dpus=" << nr_dpus << ",sf=" << batches << ",batch_rows=" << left_rows
        << ",thr=" << threshold_stream.str();
  state.SetLabel(label.str());
  auto system_ = DpuSet::allocate(nr_dpus, "nrJobsPerRank=256,sgXferEnable=true");
  JoinDpu joiner{system_, left_schema_, right_schema_, left_batches_, right_batches_};
  BM_Join<>(state, joiner);
  state.SetItemsProcessed(total_items());
  state.SetBytesProcessed(total_bytes());
#if ENABLE_PERF
  state.counters["cycles"] = cycles_count(system_);
#endif
}

constexpr std::array<double, 4> kNativeBloomThresholds = {0.30, 0.60, 0.90, 1.10};

void RegisterJoinNativeArgs(benchmark::internal::Benchmark* bench) {
  for (double threshold : kNativeBloomThresholds) {
    bench->Args({4, 64 << 10, 64 << 10, true, EncodeBloomThreshold(threshold)});
    bench->Args({4, 64 << 10, 64 << 10, false, EncodeBloomThreshold(threshold)});
  }
}

static const bool kJoinNativeBenchmarksRegistered = []() {
  BENCHMARK_REGISTER_F(PartitionedBatchGeneratorFixture, BM_JoinNative)
      ->ArgNames({"Batches", "L-Batch-Size", "R-Batch-Size", "Partitioned",
                  "Bloom-Threshold"})
      ->Apply(RegisterJoinNativeArgs)
      ->MeasureProcessCPUTime()
      ->UseRealTime()
      ->Unit(benchmark::kMillisecond);
  return true;
}();

const auto kJoinDpuConfigs = BuildJoinDpuBenchmarkConfigs();

void RegisterJoinDpuArgs(benchmark::internal::Benchmark* bench) {
  for (const auto& config : kJoinDpuConfigs) {
    bench->Args({config.batches, config.left_rows, config.right_rows, config.dpus,
                 EncodeBloomThreshold(config.bloom_threshold)});
  }
}

static const bool kJoinDpuBenchmarksRegistered = []() {
  BENCHMARK_REGISTER_F(PartitionedBatchGeneratorFixture, BM_JoinDpu)
      ->ArgNames({"Batches", "L-Batch-Size", "R-Batch-Size", "DPUs", "Bloom-Threshold"})
      ->Apply(RegisterJoinDpuArgs)
      ->MeasureProcessCPUTime()
      ->UseRealTime()
      ->Unit(benchmark::kMillisecond);
  return true;
}();

}  // namespace join
}  // namespace upmemeval
