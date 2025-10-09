#include <arrow/api.h>
#include <benchmark/benchmark.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <tuple>
#include <vector>

#include <filesystem>

#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
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

struct CacheKey {
  int64_t batches;
  int64_t left_rows;
  int64_t right_rows;
  int64_t ratio_milli;

  bool operator<(const CacheKey& other) const {
    return std::tie(batches, left_rows, right_rows, ratio_milli) <
           std::tie(other.batches, other.left_rows, other.right_rows, other.ratio_milli);
  }
};

struct CachedJoinData {
  std::shared_ptr<arrow::Schema> left_schema;
  std::shared_ptr<arrow::Schema> right_schema;
  arrow::RecordBatchVector left_batches;
  arrow::RecordBatchVector right_batches;
  uint64_t inside = 0;
  uint64_t outside = 0;
};

static std::mutex g_join_cache_mutex;
static std::map<CacheKey, CachedJoinData> g_join_cache;

std::filesystem::path CacheDirectory() {
  static std::filesystem::path cache_dir = [] {
    const char* env = std::getenv("JOIN_CACHE_DIR");
    std::filesystem::path path = env != nullptr ? std::filesystem::path(env) : std::filesystem::path("join_cache");
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
      std::cerr << "[Generator] Warning: failed to create cache directory '" << path.string()
                << "': " << ec.message() << std::endl;
    }
    return path;
  }();
  return cache_dir;
}

std::string CacheFileBase(const CacheKey& key) {
  std::ostringstream oss;
  oss << "b" << key.batches << "_l" << key.left_rows << "_r" << key.right_rows << "_ratio"
      << key.ratio_milli;
  return oss.str();
}

std::filesystem::path CachePath(const CacheKey& key, const std::string& suffix) {
  return CacheDirectory() / (CacheFileBase(key) + suffix);
}

arrow::Status WriteBatchesToFile(const std::filesystem::path& path,
                                 const arrow::RecordBatchVector& batches,
                                 const std::shared_ptr<arrow::Schema>& schema) {
  ARROW_ASSIGN_OR_RAISE(auto sink, arrow::io::FileOutputStream::Open(path.string()));
  ARROW_ASSIGN_OR_RAISE(auto writer, arrow::ipc::MakeStreamWriter(sink.get(), schema));
  for (const auto& batch : batches) {
    ARROW_RETURN_NOT_OK(writer->WriteRecordBatch(*batch));
  }
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(sink->Close());
  return arrow::Status::OK();
}

arrow::Result<std::pair<std::shared_ptr<arrow::Schema>, arrow::RecordBatchVector>>
ReadBatchesFromFile(const std::filesystem::path& path) {
  ARROW_ASSIGN_OR_RAISE(auto input, arrow::io::ReadableFile::Open(path.string()));
  ARROW_ASSIGN_OR_RAISE(auto reader, arrow::ipc::RecordBatchStreamReader::Open(input));
  auto schema = reader->schema();
  arrow::RecordBatchVector batches;
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, reader->Next());
    if (batch == nullptr) break;
    batches.push_back(batch);
  }
  ARROW_RETURN_NOT_OK(input->Close());
  return std::make_pair(schema, std::move(batches));
}

arrow::Status WriteJoinDataToDisk(const CacheKey& key, const CachedJoinData& data) {
  auto left_path = CachePath(key, "_left.arrow");
  auto right_path = CachePath(key, "_right.arrow");
  auto meta_path = CachePath(key, ".meta");

  ARROW_RETURN_NOT_OK(WriteBatchesToFile(left_path, data.left_batches, data.left_schema));
  ARROW_RETURN_NOT_OK(WriteBatchesToFile(right_path, data.right_batches, data.right_schema));

  std::ofstream meta(meta_path);
  if (meta.is_open()) {
    meta << data.inside << " " << data.outside;
  }
  return arrow::Status::OK();
}

arrow::Result<CachedJoinData> LoadJoinDataFromDisk(const CacheKey& key) {
  auto left_path = CachePath(key, "_left.arrow");
  auto right_path = CachePath(key, "_right.arrow");
  auto meta_path = CachePath(key, ".meta");

  if (!std::filesystem::exists(left_path) || !std::filesystem::exists(right_path)) {
    return arrow::Status::IOError("cache files not found");
  }

  ARROW_ASSIGN_OR_RAISE(auto left_pair, ReadBatchesFromFile(left_path));
  ARROW_ASSIGN_OR_RAISE(auto right_pair, ReadBatchesFromFile(right_path));

  CachedJoinData data;
  data.left_schema = left_pair.first;
  data.left_batches = std::move(left_pair.second);
  data.right_schema = right_pair.first;
  data.right_batches = std::move(right_pair.second);

  std::ifstream meta(meta_path);
  if (meta.is_open()) {
    meta >> data.inside >> data.outside;
  }

  return data;
}

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
const std::array<int64_t, 1> dpus_values = {2};
const std::array<int64_t, 1> scale_multipliers = {1};
const std::array<int64_t, 3> batch_rows = {56LL << 10, 128LL << 10, 256LL << 10};
const std::array<double, 3> bloom_thresholds = {0.0};

  for (int64_t dpus : dpus_values) {
    int64_t normalized_dpus = NormalizeDpuCount(dpus, max_dpus);
    if (normalized_dpus != dpus) {
      continue;
    }
    for (int64_t multiplier : scale_multipliers) {
      int64_t target_batches = dpus * multiplier;
      int64_t batches = RoundUpToPowerOfTwo(std::max<int64_t>(target_batches, dpus));
      if (batches > dpus) {
        batches = dpus;
      }
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

template <typename T>
auto GetBloomSkipped(const T& joiner, int) -> decltype(joiner.BloomSkipped(), uint64_t()) {
  return joiner.BloomSkipped();
}

template <typename T>
uint64_t GetBloomSkipped(const T&, ...) {
  return 0;
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

  uint64_t skipped = GetBloomSkipped(joiner, 0);
  state.counters["bloom_skipped"] =
      benchmark::Counter(static_cast<double>(skipped), benchmark::Counter::kAvgIterations);
}

class PartitionedBatchGeneratorFixture : public benchmark::Fixture {
 public:
  PartitionedBatchGeneratorFixture() = default;

  void SetUp(::benchmark::State& state) override {
    auto num_batches = state.range(0);
    auto left_batch_size = state.range(1);
    auto right_batch_size = state.range(2);
    assert(num_batches > 0);
    assert(left_batch_size > 0);
    assert(right_batch_size > 0);

    double outside_ratio = 0.0;
    if (const char* env = std::getenv("FK_OUTSIDE_RATIO")) {
      outside_ratio = std::clamp(std::strtod(env, nullptr), 0.0, 1.0);
    }
    int64_t ratio_milli = static_cast<int64_t>(std::llround(outside_ratio * 1000.0));

    CacheKey key{num_batches, left_batch_size, right_batch_size, ratio_milli};

    {
      std::lock_guard<std::mutex> lock(g_join_cache_mutex);
      auto it = g_join_cache.find(key);
      if (it != g_join_cache.end()) {
        left_batches_ = it->second.left_batches;
        right_batches_ = it->second.right_batches;
        left_schema_ = it->second.left_schema;
        right_schema_ = it->second.right_schema;
        if (state.thread_index() == 0) {
          uint64_t total = it->second.inside + it->second.outside;
          double actual = total ? static_cast<double>(it->second.outside) / static_cast<double>(total) : 0.0;
          std::cout << "[Generator(cache)] batches=" << num_batches
                    << " left_rows=" << left_batch_size << " right_rows=" << right_batch_size
                    << " outside_ratio_requested=" << outside_ratio << " actual=" << actual << std::endl;
        }
        return;
      }
    }

    if (auto maybe_disk = LoadJoinDataFromDisk(key); maybe_disk.ok()) {
      CachedJoinData data = std::move(maybe_disk).ValueOrDie();
      if (state.thread_index() == 0) {
        uint64_t total = data.inside + data.outside;
        double actual = total ? static_cast<double>(data.outside) / static_cast<double>(total) : 0.0;
        std::cout << "[Generator(disk)] batches=" << num_batches << " left_rows=" << left_batch_size
                  << " right_rows=" << right_batch_size << " outside_ratio_requested=" << outside_ratio
                  << " actual=" << actual << std::endl;
      }
      {
        std::lock_guard<std::mutex> lock(g_join_cache_mutex);
        auto [it, inserted] = g_join_cache.emplace(key, data);
        left_batches_ = it->second.left_batches;
        right_batches_ = it->second.right_batches;
        left_schema_ = it->second.left_schema;
        right_schema_ = it->second.right_schema;
      }
      return;
    }

    arrow::random::RandomArrayGenerator rng_right(static_cast<int64_t>(num_batches * 1315423911ULL ^
                                                                      left_batch_size * 2654435761ULL ^
                                                                      right_batch_size * 97531ULL ^
                                                                      ratio_milli));
    arrow::random::RandomArrayGenerator rng_left(static_cast<int64_t>(num_batches * 89 +
                                                                     left_batch_size * 53 +
                                                                     right_batch_size * 97 +
                                                                     ratio_milli));

    auto right_schema = arrow::schema({arrow::field("x", arrow::uint32(), /*nullable=*/false)});
    auto right_batches_base =
        generator::MakeRandomRecordBatches(rng_right, right_schema, num_batches, right_batch_size);
    auto right_pk_column = generator::MakeIndexColumn(num_batches, right_batch_size);
    auto right_batches =
        generator::AddColumn("pk", right_batches_base, right_pk_column.ValueOrDie());
    auto right_schema_with_pk = right_batches[0]->schema();

    auto left_schema = arrow::schema({arrow::field("y", arrow::uint32(), /*nullable=*/false)});
    auto left_batches_base =
        generator::MakeRandomRecordBatches(rng_left, left_schema, num_batches, left_batch_size);
    std::pair<uint64_t, uint64_t> counts{0, 0};
    auto left_fk_result = generator::MakeForeignKeyColumn(rng_left, right_batch_size, num_batches,
                                                          left_batch_size, outside_ratio, &counts);
    if (!left_fk_result.ok()) {
      state.SkipWithError(left_fk_result.status().ToString().c_str());
      return;
    }
    auto left_fk_column = left_fk_result.ValueOrDie();
    auto left_batches =
        generator::AddColumn("fk", left_batches_base, left_fk_column);
    auto left_schema_with_fk = left_batches[0]->schema();

    uint64_t total = counts.first + counts.second;
    double actual_ratio = total ? static_cast<double>(counts.second) / static_cast<double>(total) : 0.0;
    std::cout << "[Generator] batches=" << num_batches << " left_rows=" << left_batch_size
              << " right_rows=" << right_batch_size << " outside_ratio_requested=" << outside_ratio
              << " actual=" << actual_ratio << " (outside=" << counts.second
              << ", inside=" << counts.first << ")" << std::endl;

    CachedJoinData data;
    data.left_schema = left_schema_with_fk;
    data.right_schema = right_schema_with_pk;
    data.left_batches = left_batches;
    data.right_batches = right_batches;
    data.inside = counts.first;
    data.outside = counts.second;

    auto write_status = WriteJoinDataToDisk(key, data);
    if (!write_status.ok()) {
      std::cerr << "[Generator] Warning: failed to write cache files: " << write_status.ToString()
                << std::endl;
    }

    {
      std::lock_guard<std::mutex> lock(g_join_cache_mutex);
      auto [it, inserted] = g_join_cache.emplace(key, data);
      left_batches_ = it->second.left_batches;
      right_batches_ = it->second.right_batches;
      left_schema_ = it->second.left_schema;
      right_schema_ = it->second.right_schema;
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

constexpr std::array<double, 3> kNativeBloomThresholds = {0, 0.5, 1};

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
