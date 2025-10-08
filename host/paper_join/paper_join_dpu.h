#pragma once

#include <arrow/api.h>
#include <utility>

#include "partition/partitioner.h"
namespace dpu {
class DpuSet;
}

namespace upmemeval {
namespace paper_join {

struct PaperJoinResult {
  uint64_t total_probes;
  uint64_t matches;
  uint64_t mismatches;
  uint64_t bloom_skipped;
  uint64_t bloom_false_positives;
};

class PaperJoinDpu {
 public:
  PaperJoinDpu(dpu::DpuSet& system_, std::shared_ptr<arrow::Schema> left_schema,
               std::shared_ptr<arrow::Schema> right_schema,
               arrow::RecordBatchVector left_batches,
               arrow::RecordBatchVector right_batches)
      : system_(system_),
        left_schema_(std::move(left_schema)),
        right_schema_(std::move(right_schema)),
        left_batches_(std::move(left_batches)),
        right_batches_(std::move(right_batches)) {}

  arrow::Status Prepare();

  arrow::Result<PaperJoinResult> Run();

 private:
  dpu::DpuSet& system_;
  std::shared_ptr<arrow::Schema> left_schema_;
  std::shared_ptr<arrow::Schema> right_schema_;
  arrow::RecordBatchVector left_batches_;
  arrow::RecordBatchVector right_batches_;

  arrow::Result<std::shared_ptr<partition::Partitioner>> do_partition(
      int k_column_index,
      const std::vector<uint32_t>& dpu_offset,
      std::vector<std::vector<uint32_t>>& metadata, uint32_t buffer_length, bool is_left);
  arrow::Result<PaperJoinResult> Run_internal();
};

}  // namespace paper_join
}  // namespace upmemeval
