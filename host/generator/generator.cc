#include "generator.h"

#include <arrow/testing/random.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
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
    int32_t batch_size, double outside_ratio,
    std::pair<uint64_t, uint64_t>* counts) {
  arrow::ArrayVector out(num_batches);

  std::mt19937_64 rng(42);
  uint64_t total_inside = 0;
  uint64_t total_outside = 0;

  for (int32_t batch = 0; batch < num_batches; ++batch) {
    arrow::UInt32Builder builder;
    ARROW_RETURN_NOT_OK(builder.Reserve(batch_size));

    uint32_t pk_base = static_cast<uint32_t>(batch) * pk_batch_size;
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_int_distribution<uint32_t> inside_dist(0, pk_batch_size > 0 ? pk_batch_size - 1 : 0);
    std::uniform_int_distribution<uint32_t> outside_dist(0, pk_batch_size > 0 ? pk_batch_size - 1 : 0);

    for (int32_t row = 0; row < batch_size; ++row) {
      uint32_t value;
      if (pk_batch_size == 0 || prob_dist(rng) >= outside_ratio) {
        value = pk_base + inside_dist(rng);
        ++total_inside;
      } else {
        value = pk_base + pk_batch_size + outside_dist(rng);
        ++total_outside;
      }
      builder.UnsafeAppend(value);
    }

    ARROW_ASSIGN_OR_RAISE(out[batch], builder.Finish());
  }

  if (counts != nullptr) {
    counts->first = total_inside;
    counts->second = total_outside;
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
