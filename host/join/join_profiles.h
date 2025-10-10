#pragma once

#include <optional>
#include <string>
#include <vector>

#include <arrow/api.h>

#include "generator/generator.h"

namespace upmemeval {
namespace join {

enum class JoinDataProfile {
  Uniform = 0,
  RandomMissStorm,
  HotMissCluster,
  SkewedHotMatches,
  NullHeavyHotMiss,
  WidePayloadStress
};

std::string ProfileTag(JoinDataProfile profile);
std::string ProfileDescription(JoinDataProfile profile);

struct JoinScenarioConfig {
  JoinDataProfile profile = JoinDataProfile::Uniform;
  std::string tag = ProfileTag(profile);
  std::string description = ProfileDescription(profile);
  generator::ForeignKeyConfig fk_config{};
  std::vector<std::shared_ptr<arrow::Field>> extra_left_fields;
  std::vector<std::shared_ptr<arrow::Field>> extra_right_fields;
  bool left_payload_nullable = false;
  bool right_payload_nullable = false;
  uint64_t seed_salt = 0;
};

JoinScenarioConfig ResolveScenarioConfig(std::optional<double> outside_ratio_override);

struct GeneratedJoinData {
  std::shared_ptr<arrow::Schema> left_schema;
  std::shared_ptr<arrow::Schema> right_schema;
  arrow::RecordBatchVector left_batches;
  arrow::RecordBatchVector right_batches;
  generator::ForeignKeyStats stats;
};

GeneratedJoinData GenerateJoinData(const JoinScenarioConfig& scenario,
                                   int64_t num_batches,
                                   int64_t left_batch_size,
                                   int64_t right_batch_size);

}  // namespace join
}  // namespace upmemeval
