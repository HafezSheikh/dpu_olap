#include "generator.h"

#include <arrow/testing/random.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>

namespace upmemeval {

namespace generator {

using namespace arrow;
using namespace arrow::compute;

template <typename Fn, typename From,
          typename To = decltype(std::declval<Fn>()(std::declval<From>()))>
std::vector<To> MapVector(Fn&& map, const std::vector<From>& source) {
  std::vector<To> out;
  out.reserve(source.size());
  std::transform(source.begin(), source.end(), std::back_inserter(out),
                 std::forward<Fn>(map));
  return out;
}

arrow::RecordBatchVector MakeRandomRecordBatches(
    ::arrow::random::RandomArrayGenerator& g,
    const std::shared_ptr<arrow::Schema>& schema, int num_batches, int batch_size) {
  RecordBatchVector out(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    out[i] = g.BatchOf(schema->fields(), batch_size);
  }
  return out;
}

arrow::RecordBatchVector AddColumn(const std::string& name,
                                   const arrow::RecordBatchVector& batches,
                                   arrow::ArrayVector indexColumn) {
  arrow::RecordBatchVector out;
  out.reserve(batches.size());

  assert(batches.size() == indexColumn.size());
  for (size_t i = 0; i < batches.size(); ++i) {
    assert(batches[i]->num_rows() == indexColumn[i]->length());
    out.push_back(batches[i]->AddColumn(0, name, indexColumn[i]).ValueOrDie());
  }
  return out;
}

arrow::Result<arrow::ArrayVector> MakeForeignKeyColumn(
    ::arrow::random::RandomArrayGenerator& g, uint32_t pk_batch_size, int32_t num_batches,
    int32_t batch_size, const ForeignKeyConfig& config, ForeignKeyStats* stats_ptr) {
  arrow::ArrayVector out(num_batches);

  ForeignKeyStats local_stats;
  ForeignKeyStats* stats = stats_ptr != nullptr ? stats_ptr : &local_stats;
  if (stats_ptr != nullptr) {
    *stats_ptr = ForeignKeyStats{};
  }

  std::mt19937_64 rng(config.seed);
  const double outside_ratio = std::clamp(config.outside_ratio, 0.0, 1.0);
  const double null_ratio = std::clamp(config.null_ratio, 0.0, 1.0);
  const double hot_probability = std::clamp(config.hot_probability, 0.0, 1.0);
  double hot_key_fraction = std::clamp(config.hot_key_fraction, 0.0, 1.0);
  const double miss_hot_probability =
      std::clamp(config.miss_hot_probability, 0.0, 1.0);
  double miss_hot_key_fraction = std::clamp(config.miss_hot_key_fraction, 0.0, 1.0);
  if (hot_probability == 0.0 || pk_batch_size == 0) {
    hot_key_fraction = 0.0;
  }
  if (miss_hot_probability == 0.0 || pk_batch_size == 0) {
    miss_hot_key_fraction = 0.0;
  }

  const uint64_t global_span = static_cast<uint64_t>(num_batches) * pk_batch_size;
  const bool use_high_bit_escape = global_span < (1ull << 31);
  std::uniform_int_distribution<uint32_t> outside_dist_global(
      0, pk_batch_size > 0 ? pk_batch_size - 1 : 0);

  uint32_t hot_span = 0;
  if (hot_key_fraction > 0.0) {
    double scaled = static_cast<double>(pk_batch_size) * hot_key_fraction;
    hot_span = static_cast<uint32_t>(std::max<double>(1.0, std::floor(scaled + 0.5)));
    hot_span = std::min<uint32_t>(hot_span, pk_batch_size == 0 ? 0 : pk_batch_size);
  }

  uint32_t miss_span = 0;
  uint32_t miss_pool_base = 0;
  std::uniform_int_distribution<uint32_t> miss_hot_dist;
  bool use_hot_miss_pool = false;
  if (miss_hot_probability > 0.0 && miss_hot_key_fraction > 0.0) {
    double scaled = static_cast<double>(pk_batch_size) * miss_hot_key_fraction;
    miss_span = static_cast<uint32_t>(std::max<double>(1.0, std::floor(scaled + 0.5)));
    if (pk_batch_size > 0) {
      miss_span = std::min<uint32_t>(miss_span, pk_batch_size);
    }
    miss_span = std::max<uint32_t>(1U, miss_span);
    if (use_high_bit_escape) {
      miss_pool_base = 1u << 31;
    } else {
      uint64_t tentative_base = global_span;
      uint64_t max_value = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
      if (tentative_base + miss_span > max_value) {
        tentative_base = max_value - miss_span;
      }
      miss_pool_base = static_cast<uint32_t>(tentative_base);
    }
    miss_hot_dist = std::uniform_int_distribution<uint32_t>(0, miss_span - 1);
    use_hot_miss_pool = true;
  }

  for (int32_t batch = 0; batch < num_batches; ++batch) {
    arrow::UInt32Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(batch_size));

    uint32_t pk_base = static_cast<uint32_t>(batch) * pk_batch_size;
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_int_distribution<uint32_t> inside_dist;
    std::uniform_int_distribution<uint32_t> hot_dist;
    if (pk_batch_size > 0) {
      inside_dist = std::uniform_int_distribution<uint32_t>(0, pk_batch_size - 1);
    }
    if (hot_span > 0) {
      hot_dist = std::uniform_int_distribution<uint32_t>(0, hot_span - 1);
    }

    for (int32_t row = 0; row < batch_size; ++row) {
      if (null_ratio > 0.0 && prob_dist(rng) < null_ratio) {
        builder.UnsafeAppendNull();
        stats->nulls++;
        continue;
      }

      bool choose_outside = pk_batch_size == 0;
      if (!choose_outside && outside_ratio > 0.0) {
        choose_outside = prob_dist(rng) < outside_ratio;
      }

      if (choose_outside) {
        uint32_t value;
        bool take_hot_miss = use_hot_miss_pool && (prob_dist(rng) < miss_hot_probability);
        if (take_hot_miss) {
          uint32_t miss_offset = miss_hot_dist(rng);
          value = use_high_bit_escape ? (miss_pool_base | miss_offset)
                                      : static_cast<uint32_t>(miss_pool_base + miss_offset);
          stats->miss_hot++;
        } else {
          if (use_high_bit_escape) {
            value = (1u << 31) | outside_dist_global(rng);
          } else {
            uint64_t candidate = global_span + outside_dist_global(rng);
            if (candidate > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
              candidate = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
            }
            value = static_cast<uint32_t>(candidate);
          }
        }
        builder.UnsafeAppend(value);
        stats->outside++;
        continue;
      }

      bool counted_hot = false;
      uint32_t offset = 0;
      if (hot_span > 0 && prob_dist(rng) < hot_probability) {
        offset = hot_dist(rng);
        counted_hot = true;
      } else if (pk_batch_size > 0) {
        offset = inside_dist(rng);
      }

      uint32_t value = pk_base + offset;
      builder.UnsafeAppend(value);
      stats->inside++;
      if (counted_hot) {
        stats->hot++;
      }
    }

    ARROW_ASSIGN_OR_RAISE(out[batch], builder.Finish());
  }

  return out;
}

arrow::Result<arrow::ArrayVector> MakeIndexColumn(int num_batches, int batch_size) {
  uint32_t value = 0;
  arrow::ArrayVector out(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    arrow::UInt32Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(batch_size));
    for (int j = 0; j < batch_size; ++j) {
      builder.UnsafeAppend(value++);
    }
    ARROW_ASSIGN_OR_RAISE(out[i], builder.Finish());
  }
  return out;
}

std::vector<arrow::compute::ExecBatch> ToExecBatches(arrow::RecordBatchVector batches) {
  return MapVector([](std::shared_ptr<RecordBatch> batch) { return ExecBatch(*batch); },
                   batches);
}

arrow::AsyncGenerator<arrow::util::optional<arrow::compute::ExecBatch>>
MakeAsyncGenerator(std::vector<arrow::compute::ExecBatch> batches, bool parallel) {
  auto opt_batches = MapVector(
      [](arrow::compute::ExecBatch batch) {
        return util::make_optional(std::move(batch));
      },
      batches);

  AsyncGenerator<util::optional<ExecBatch>> gen;

  if (parallel) {
    // emulate batches completing initial decode-after-scan on a cpu thread
    gen = MakeBackgroundGenerator(MakeVectorIterator(std::move(opt_batches)),
                                  ::arrow::internal::GetCpuThreadPool())
              .ValueOrDie();

    // ensure that callbacks are not executed immediately on a background thread
    gen = MakeTransferredGenerator(std::move(gen), ::arrow::internal::GetCpuThreadPool());
  } else {
    gen = MakeVectorGenerator(std::move(opt_batches));
  }

  return gen;
}

std::shared_ptr<::arrow::RecordBatch> RecordBatchOf(
    std::vector<std::string> names, std::vector<std::shared_ptr<arrow::Array>> data) {
  ::arrow::FieldVector fields(names.size());
  for (size_t i = 0; i < fields.size(); ++i) {
    fields[i] = ::arrow::field(names[i], data[i]->type());
  }
  return ::arrow::RecordBatch::Make(arrow::schema(fields), data[0]->length(), data);
}

}  // namespace generator
}  // namespace upmemeval
