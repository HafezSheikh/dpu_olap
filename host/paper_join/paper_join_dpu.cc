#include "paper_join_dpu.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

#include "dpuext/api.h"
#include "generator/generator.h"
#include "system/system.h"

#include "umq/bitops.h"
#include "umq/kernels.h"
#include "umq/log.h"

using namespace dpu;

namespace upmemeval {
namespace paper_join {

namespace {

/** Compute buffer lengths when data is still organized in batches. */
std::vector<std::vector<uint32_t>> GetBatchBufferLengths(
    size_t n_partitions, const arrow::RecordBatchVector& batches,
    const std::vector<std::vector<uint32_t>>& metadata) {
  std::vector<std::vector<uint32_t>> buffer_lengths(n_partitions);
  for (size_t partition = 0; partition < n_partitions; ++partition) {
    uint32_t total = 0;
    for (size_t batch = 0; batch < batches.size(); ++batch) {
      total += metadata[batch][partition] -
               (partition == 0 ? 0 : metadata[batch][partition - 1]);
    }
    buffer_lengths[partition] = {total};
  }
  return buffer_lengths;
}

/** Compute buffer lengths once data has already been partitioned. */
std::vector<std::vector<uint32_t>> GetPartitionBufferLengths(
    int key_column_index, arrow::RecordBatchVector& partitions) {
  std::vector<std::vector<uint32_t>> buffer_lengths(partitions.size());
  for (size_t partition = 0; partition < partitions.size(); ++partition) {
    auto length = arrow_data_buffer_length(partitions, partition, key_column_index).ValueOrDie();
    buffer_lengths[partition] = {length};
  }
  return buffer_lengths;
}

struct BloomConfig {
  uint32_t bits_per_key;
  uint32_t min_bits;
  uint32_t max_bits;
  uint32_t min_partition_rows;
  uint32_t max_partition_rows;
  double min_mismatch_rate;
};

BloomConfig GetBloomConfig() {
  BloomConfig cfg{};
  cfg.bits_per_key = static_cast<uint32_t>(std::max(1, variables::bloom_bits_per_key()));
  cfg.min_bits = static_cast<uint32_t>(std::max(1, variables::bloom_min_bits()));
  cfg.max_bits = std::max<uint32_t>(cfg.min_bits,
                                    static_cast<uint32_t>(std::max(1, variables::bloom_max_bits())));
  cfg.min_partition_rows =
      static_cast<uint32_t>(std::max(0, variables::bloom_min_partition_rows()));
  cfg.max_partition_rows = std::max<uint32_t>(cfg.min_partition_rows,
                                              static_cast<uint32_t>(std::max(0, variables::bloom_max_partition_rows())));
  cfg.min_mismatch_rate = std::clamp(variables::bloom_min_mismatch_rate(), 0.0, 1.0);
  return cfg;
}

template <typename ArrowType>
double EstimateMismatchRateTyped(const arrow::NumericArray<ArrowType>& left,
                                 const arrow::NumericArray<ArrowType>& right) {
  using CType = typename ArrowType::c_type;

  const int64_t right_length = right.length();
  if (right_length == 0) {
    return 0.0;
  }

  const CType* right_raw = right.raw_values();
  std::unordered_set<CType> right_values;
  right_values.reserve(static_cast<size_t>(right_length));
  for (int64_t idx = 0; idx < right_length; ++idx) {
    if (!right.IsNull(idx)) {
      right_values.insert(right_raw[idx]);
    }
  }
  if (right_values.empty()) {
    return 1.0;
  }

  const int64_t left_length = left.length();
  if (left_length == 0) {
    return 0.0;
  }

  const CType* left_raw = left.raw_values();
  size_t considered = 0;
  size_t misses = 0;
  for (int64_t idx = 0; idx < left_length; ++idx) {
    if (left.IsNull(idx)) {
      continue;
    }
    ++considered;
    if (right_values.find(left_raw[idx]) == right_values.end()) {
      ++misses;
    }
  }

  if (considered == 0) {
    return 0.0;
  }

  return static_cast<double>(misses) / static_cast<double>(considered);
}

double EstimateMismatchRate(const std::shared_ptr<arrow::Array>& left,
                            const std::shared_ptr<arrow::Array>& right) {
  if (left->type_id() != right->type_id()) {
    return 1.0;
  }
  switch (left->type_id()) {
    case arrow::Type::UINT32:
      return EstimateMismatchRateTyped<arrow::UInt32Type>(
          static_cast<const arrow::UInt32Array&>(*left),
          static_cast<const arrow::UInt32Array&>(*right));
    case arrow::Type::INT32:
      return EstimateMismatchRateTyped<arrow::Int32Type>(
          static_cast<const arrow::Int32Array&>(*left),
          static_cast<const arrow::Int32Array&>(*right));
    default:
      return 1.0;
  }
}

std::vector<std::vector<uint32_t>> ComputeBloomBitAllocation(
    const arrow::RecordBatchVector& left_partitions,
    const arrow::RecordBatchVector& right_partitions,
    int fk_column_index, int pk_column_index,
    const std::vector<std::vector<uint32_t>>& right_lengths) {
  const bool verbose = std::getenv("PAPER_JOIN_VERBOSE") != nullptr;
  const BloomConfig cfg = GetBloomConfig();
  if (verbose) {
    std::cout << "[Bloom] config bits_per_key=" << cfg.bits_per_key
              << " min_bits=" << cfg.min_bits << " max_bits=" << cfg.max_bits
              << " min_partition_rows=" << cfg.min_partition_rows
              << " max_partition_rows=" << cfg.max_partition_rows
              << " min_mismatch_rate=" << cfg.min_mismatch_rate << std::endl;
  }

  std::vector<std::vector<uint32_t>> bloom_bits(
      right_lengths.size(), std::vector<uint32_t>(1, 0));

  for (size_t i = 0; i < right_lengths.size(); ++i) {
    const uint32_t right_length = right_lengths[i].empty() ? 0 : right_lengths[i][0];
    if (right_length < cfg.min_partition_rows) {
      if (verbose) {
        std::cout << "[Bloom] partition " << i << " skipped (right_length=" << right_length
                  << " < " << cfg.min_partition_rows << ")" << std::endl;
      }
      continue;
    }

    if (cfg.max_partition_rows > 0 && right_length > cfg.max_partition_rows) {
      if (verbose) {
        std::cout << "[Bloom] partition " << i << " skipped (right_length=" << right_length
                  << " > " << cfg.max_partition_rows << ")" << std::endl;
      }
      continue;
    }

    const auto& right_column = right_partitions[i]->column(pk_column_index);
    const auto& left_column = left_partitions[i]->column(fk_column_index);

    const double mismatch_rate = EstimateMismatchRate(left_column, right_column);
    if (mismatch_rate < cfg.min_mismatch_rate) {
      if (verbose) {
        std::cout << std::fixed << std::setprecision(3)
                  << "[Bloom] partition " << i << " skipped (mismatch_rate=" << mismatch_rate
                  << " < " << cfg.min_mismatch_rate << ") right_length=" << right_length
                  << std::endl;
      }
      continue;
    }

    uint64_t bits = static_cast<uint64_t>(right_length) * cfg.bits_per_key;
    bits = std::max<uint64_t>(bits, cfg.min_bits);
    bits = std::min<uint64_t>(bits, cfg.max_bits);
    if (bits > (1u << 20)) {
      if (verbose) {
        std::cout << "[Bloom] partition " << i << " requested " << bits
                  << " bits but clamped to device limit " << (1u << 20) << std::endl;
      }
      bits = 1u << 20;
    }
    bloom_bits[i][0] = static_cast<uint32_t>(bits);
    if (verbose) {
      std::cout << std::fixed << std::setprecision(3)
                << "[Bloom] partition " << i << " enabled (mismatch_rate=" << mismatch_rate
                << ", right_length=" << right_length << ", bloom_bits=" << bloom_bits[i][0]
                << ")" << std::endl;
    }
  }

  return bloom_bits;
}

}  // namespace

arrow::Status PaperJoinDpu::Prepare() {
  try {
    system_.load(JOIN_DPU_BINARY);
  } catch (const dpu::DpuError& e) {
    return arrow::Status::UnknownError(e.what());
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<partition::Partitioner>> PaperJoinDpu::do_partition(
    int key_column_index, const std::vector<uint32_t>& dpu_offset,
    std::vector<std::vector<uint32_t>>& metadata, uint32_t buffer_length, bool is_left) {
  using namespace partition;

  auto pool = arrow::internal::GetCpuThreadPool();
  auto schema = is_left ? left_schema_ : right_schema_;
  auto partitioner = std::make_shared<partition::Partitioner>(pool, schema);

  auto& batches = is_left ? left_batches_ : right_batches_;
  auto n_partitions = batches.size();
  auto nr_dpus = system_.dpus().size();
  auto batch_rows = batches[0]->num_rows();

  ARROW_RETURN_NOT_OK(partitioner->AllocatePartitions(
      n_partitions, static_cast<int64_t>(batch_rows) * 2));
  ARROW_RETURN_NOT_OK(partitioner->AllocateOffsets(n_partitions, nr_dpus));
  ARROW_RETURN_NOT_OK(partitioner->GenerateRandomShifts(n_partitions, nr_dpus));

  for (size_t i = 0; i < batches.size(); i += nr_dpus) {
    ARROW_RETURN_NOT_OK(partitioner->PartitionKernel(system_, batches, i, key_column_index,
                                                     n_partitions, dpu_offset, metadata));
    ARROW_RETURN_NOT_OK(partitioner->GetOffsets(system_, metadata, i, dpu_offset));
    ARROW_RETURN_NOT_OK(partitioner->LoadPartitions(system_, metadata, i, key_column_index,
                                                    buffer_length));
  }

  return partitioner;
}

arrow::Result<PaperJoinResult> PaperJoinDpu::Run() {
  auto result = Run_internal();
  system_.async().sync();
  return result;
}

arrow::Result<PaperJoinResult> PaperJoinDpu::Run_internal() {
  const bool verbose = std::getenv("PAPER_JOIN_VERBOSE") != nullptr;
  auto time_now = []() { return std::chrono::steady_clock::now(); };
  auto to_ms = [](auto delta) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(delta).count();
  };
  auto run_start = time_now();

  const auto n_partitions = left_batches_.size();
  const auto pk_column_index = right_schema_->GetFieldIndex("pk");
  const auto fk_column_index = left_schema_->GetFieldIndex("fk");
  if (pk_column_index < 0 || fk_column_index < 0) {
    return arrow::Status::Invalid("pk/fk columns not found");
  }

  ARROW_ASSIGN_OR_RAISE(uint32_t right_buffer_length,
                        arrow_data_buffer_length(right_batches_, 0UL, pk_column_index));
  ARROW_ASSIGN_OR_RAISE(uint32_t left_buffer_length,
                        arrow_data_buffer_length(left_batches_, 0UL, fk_column_index));

  if (verbose) {
    std::cout << "[PaperJoin] partition start (partitions=" << n_partitions
              << ", fk_rows_per_batch=" << left_buffer_length
              << ", pk_rows_per_batch=" << right_buffer_length << ")" << std::endl;
  }
  auto partition_start = time_now();

  auto nr_dpus = system_.dpus().size();
  if (n_partitions % nr_dpus != 0) {
    return arrow::Status::Invalid("#partitions must be divisible by #DPUs");
  }

  std::vector<uint32_t> dpu_offset(system_.ranks().size(), 0);
  for (size_t i = 0; i + 1 < system_.ranks().size(); ++i) {
    dpu_offset[i + 1] = dpu_offset[i] + system_.ranks()[i]->dpus().size();
  }

  std::vector<std::vector<uint32_t>> metadata(n_partitions,
                                              std::vector<uint32_t>(n_partitions, 0));

  auto [left_partitions, right_partitions] = [&]() {
    auto left_partitioner =
        do_partition(fk_column_index, dpu_offset, metadata, left_buffer_length, true)
            .ValueOrDie();
    auto right_partitioner =
        do_partition(pk_column_index, dpu_offset, metadata, right_buffer_length, false)
            .ValueOrDie();

    system_.async().sync();

    auto left = partition::ToRecordBatches(std::move(left_partitioner)->partitions()).ValueOrDie();
    auto right = partition::ToRecordBatches(std::move(right_partitioner)->partitions()).ValueOrDie();
    return std::make_pair(std::move(left), std::move(right));
  }();

  if (verbose) {
    std::cout << "[PaperJoin] partition finished in "
              << to_ms(time_now() - partition_start) << " ms" << std::endl;
  }

  auto buffer_partitioned_length_param_right =
      GetPartitionBufferLengths(pk_column_index, right_partitions);
  auto buffer_partitioned_length_param_left =
      GetPartitionBufferLengths(fk_column_index, left_partitions);

  auto bloom_bits_per_partition = ComputeBloomBitAllocation(
      left_partitions, right_partitions, fk_column_index, pk_column_index,
      buffer_partitioned_length_param_right);

  std::vector<int32_t> kernel_param_build{KernelHashBuild};
  std::vector<int32_t> kernel_param_bloom{KernelBloomProfile};

  PaperJoinResult accum{};

  for (size_t batches_offset = 0; batches_offset < n_partitions; batches_offset += nr_dpus) {
    const size_t batch_index = batches_offset / nr_dpus;
    auto batch_start = time_now();
    if (verbose) {
      std::cout << "[PaperJoin] batch " << batch_index << " build start" << std::endl;
    }
    system_.async().copy("kernel", 0, kernel_param_build, sizeof(int32_t));
    system_.async().copy_from("bloom_n_bits", bloom_bits_per_partition, batches_offset);
    system_.async().copy_from("buffer_length", buffer_partitioned_length_param_right,
                              batches_offset);
    ARROW_RETURN_NOT_OK(arrow_copy_to_dpus(system_, "buffer", right_partitions, batches_offset,
                                           pk_column_index, true));
    system_.async().exec();
    if (verbose) {
      std::cout << "[PaperJoin] batch " << batch_index
                << " build exec issued" << std::endl;
    }
    auto build_end = time_now();
    if (verbose) {
      std::cout << "[PaperJoin] batch " << batch_index
                << " build finished in " << to_ms(build_end - batch_start)
                << " ms" << std::endl;
      std::cout << "[PaperJoin] batch " << batch_index << " bloom profile start" << std::endl;
    }
    system_.async().copy("kernel", 0, kernel_param_bloom, sizeof(int32_t));
    system_.async().copy_from("buffer_length", buffer_partitioned_length_param_left,
                              batches_offset);
    ARROW_RETURN_NOT_OK(arrow_copy_to_dpus(system_, "buffer", left_partitions, batches_offset,
                                           fk_column_index, true));
    system_.async().exec();
    system_.async().sync();
    auto profile_end = time_now();

    std::vector<std::vector<bloom_profile_counters_t>> bloom_profiles(system_.dpus().size());
    for (auto& counters : bloom_profiles) {
      counters.resize(1);
    }
    system_.copy(bloom_profiles, "bloom_profile_counters");

    uint64_t batch_total = 0;
    uint64_t batch_skipped = 0;
    uint64_t batch_false_positive = 0;
    uint64_t batch_matches = 0;
    for (const auto& per_dpu : bloom_profiles) {
      const auto& counters = per_dpu.front();
      batch_total += counters.total_probes;
      batch_skipped += counters.bloom_skipped;
      batch_false_positive += counters.bloom_false_positives;
      batch_matches += counters.matches;
    }

    accum.total_probes += batch_total;
    accum.bloom_skipped += batch_skipped;
    accum.bloom_false_positives += batch_false_positive;
    accum.matches += batch_matches;

    if (verbose) {
      const auto mismatches = batch_total - batch_matches;
      std::cout << "[PaperJoin] batch " << batch_index
                << " bloom profile finished in " << to_ms(profile_end - build_end)
                << " ms (total=" << batch_total
                << ", matches=" << batch_matches
                << ", mismatches=" << mismatches
                << ", bloom_skipped=" << batch_skipped
                << ", bloom_false_positive=" << batch_false_positive << ")" << std::endl;
      std::cout << "[PaperJoin] batch " << batch_index
                << " total batch time " << to_ms(profile_end - batch_start)
                << " ms" << std::endl;
    }
  }

  system_.async().sync();

  PaperJoinResult result{};
  result.total_probes = accum.total_probes;
  result.matches = accum.matches;
  result.mismatches = accum.total_probes - accum.matches;
  result.bloom_skipped = accum.bloom_skipped;
  result.bloom_false_positives = accum.bloom_false_positives;
  if (verbose) {
    std::cout << "[PaperJoin] run finished in "
              << to_ms(time_now() - run_start) << " ms"
              << " (total=" << result.total_probes
              << ", matches=" << result.matches
              << ", mismatches=" << result.mismatches
              << ", bloom_skipped=" << result.bloom_skipped
              << ", bloom_false_positive=" << result.bloom_false_positives << ")" << std::endl;
  }
  return result;
}

}  // namespace paper_join
}  // namespace upmemeval
