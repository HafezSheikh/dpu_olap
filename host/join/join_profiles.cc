#include "join_profiles.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <arrow/testing/random.h>

namespace upmemeval {
namespace join {

using generator::ForeignKeyConfig;

std::string ProfileTag(JoinDataProfile profile) {
  switch (profile) {
    case JoinDataProfile::Uniform:
      return "uniform";
    case JoinDataProfile::RandomMissStorm:
      return "miss_storm";
    case JoinDataProfile::HotMissCluster:
      return "hot_miss_cluster";
    case JoinDataProfile::SkewedHotMatches:
      return "skewed_hot_hits";
    case JoinDataProfile::NullHeavyHotMiss:
      return "null_hot_miss";
    case JoinDataProfile::WidePayloadStress:
      return "wide_payload";
  }
  return "unknown";
}

std::string ProfileDescription(JoinDataProfile profile) {
  switch (profile) {
    case JoinDataProfile::Uniform:
      return "Uniform FK distribution with sequential PKs and random payloads.";
    case JoinDataProfile::RandomMissStorm:
      return "Outside-heavy FKs with near-random misses to stress Bloom and cold caches.";
    case JoinDataProfile::HotMissCluster:
      return "Misses recycle a small hot pool to trigger the WRAM hot-miss cache.";
    case JoinDataProfile::SkewedHotMatches:
      return "Reads hammer a small hot PK subset with occasional misses.";
    case JoinDataProfile::NullHeavyHotMiss:
      return "Null-rich fact table plus clustered misses for mixed cache/null handling.";
    case JoinDataProfile::WidePayloadStress:
      return "Wider, fixed-width payloads with moderate skew and reusable misses.";
  }
  return "";
}

JoinScenarioConfig ResolveScenarioConfig(std::optional<double> outside_ratio_override) {
  JoinScenarioConfig cfg;
  std::string requested = "uniform";
  if (const char* env = std::getenv("JOIN_DATA_PROFILE")) {
    requested.assign(env);
    std::transform(requested.begin(), requested.end(), requested.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  }

  auto set_profile = [&](JoinDataProfile profile) {
    cfg.profile = profile;
    cfg.tag = ProfileTag(profile);
    cfg.description = ProfileDescription(profile);
  };

  if (requested == "miss_storm" || requested == "high_mismatch" || requested == "mismatch" ||
      requested == "mismatch_heavy") {
    set_profile(JoinDataProfile::RandomMissStorm);
    cfg.fk_config.outside_ratio = 0.90;
    cfg.fk_config.hot_probability = 0.0;
    cfg.seed_salt = 101;
  } else if (requested == "hot_miss_cluster" || requested == "miss_cluster" ||
             requested == "clustered_miss") {
    set_profile(JoinDataProfile::HotMissCluster);
    cfg.fk_config.outside_ratio = 0.45;
    cfg.fk_config.miss_hot_probability = 0.95;
    cfg.fk_config.miss_hot_key_fraction = 0.002;
    cfg.fk_config.hot_probability = 0.20;
    cfg.fk_config.hot_key_fraction = 0.005;
    cfg.seed_salt = 202;
  } else if (requested == "skewed_hot" || requested == "skew" || requested == "hot_hits" ||
             requested == "hot") {
    set_profile(JoinDataProfile::SkewedHotMatches);
    cfg.fk_config.outside_ratio = 0.05;
    cfg.fk_config.hot_probability = 0.85;
    cfg.fk_config.hot_key_fraction = 0.01;
    cfg.fk_config.miss_hot_probability = 0.10;
    cfg.fk_config.miss_hot_key_fraction = 0.005;
    cfg.seed_salt = 303;
  } else if (requested == "null_hot_miss" || requested == "null_fk" || requested == "fk_nulls" ||
             requested == "nulls") {
    set_profile(JoinDataProfile::NullHeavyHotMiss);
    cfg.fk_config.null_ratio = 0.30;
    cfg.fk_config.outside_ratio = 0.35;
    cfg.fk_config.miss_hot_probability = 0.90;
    cfg.fk_config.miss_hot_key_fraction = 0.003;
    cfg.fk_config.hot_probability = 0.15;
    cfg.fk_config.hot_key_fraction = 0.003;
    cfg.left_payload_nullable = true;
    cfg.seed_salt = 404;
  } else if (requested == "wide_payload" || requested == "wide" || requested == "mixed_payload" ||
             requested == "wide_payload_stress") {
    set_profile(JoinDataProfile::WidePayloadStress);
    cfg.fk_config.outside_ratio = 0.15;
    cfg.fk_config.hot_probability = 0.35;
    cfg.fk_config.hot_key_fraction = 0.01;
    cfg.fk_config.miss_hot_probability = 0.85;
    cfg.fk_config.miss_hot_key_fraction = 0.0025;
    cfg.left_payload_nullable = true;
    cfg.right_payload_nullable = true;
    cfg.seed_salt = 505;
    cfg.extra_right_fields = {
        arrow::field("payload_r_measure", arrow::uint32()),
        arrow::field("payload_r_aux1", arrow::uint32()),
        arrow::field("payload_r_flag", arrow::uint8()),
        arrow::field("payload_r_aux2", arrow::uint32())};
    cfg.extra_left_fields = {
        arrow::field("payload_l_metric", arrow::uint32()),
        arrow::field("payload_l_aux1", arrow::uint32()),
        arrow::field("payload_l_id", arrow::uint32()),
        arrow::field("payload_l_aux2", arrow::uint32())};
  } else {
    set_profile(JoinDataProfile::Uniform);
    cfg.fk_config.outside_ratio = 0.0;
    cfg.seed_salt = 0;
  }

  if (outside_ratio_override.has_value()) {
    cfg.fk_config.outside_ratio = std::clamp(*outside_ratio_override, 0.0, 1.0);
  }

  cfg.fk_config.seed += cfg.seed_salt;
  return cfg;
}

GeneratedJoinData GenerateJoinData(const JoinScenarioConfig& scenario,
                                   int64_t num_batches,
                                   int64_t left_batch_size,
                                   int64_t right_batch_size) {
  GeneratedJoinData output;

  uint64_t scenario_salt = scenario.seed_salt != 0
                               ? scenario.seed_salt
                               : static_cast<uint64_t>(static_cast<int>(scenario.profile) + 1) *
                                     65537ULL;
  int64_t ratio_milli =
      static_cast<int64_t>(std::llround(std::clamp(scenario.fk_config.outside_ratio, 0.0, 1.0) *
                                        1000.0));

  arrow::random::RandomArrayGenerator rng_right(static_cast<int64_t>(
      num_batches * 1315423911ULL ^ left_batch_size * 2654435761ULL ^
      right_batch_size * 97531ULL ^ ratio_milli ^ scenario_salt));
  arrow::random::RandomArrayGenerator rng_left(static_cast<int64_t>(
      num_batches * 89 + left_batch_size * 53 + right_batch_size * 97 + ratio_milli +
      static_cast<int64_t>(scenario_salt)));

  std::vector<std::shared_ptr<arrow::Field>> right_fields;
  right_fields.push_back(
      arrow::field("x", arrow::uint32(), scenario.right_payload_nullable));
  right_fields.insert(right_fields.end(), scenario.extra_right_fields.begin(),
                      scenario.extra_right_fields.end());
  auto right_schema = arrow::schema(right_fields);
  auto right_batches_base =
      generator::MakeRandomRecordBatches(rng_right, right_schema, static_cast<int>(num_batches),
                                         static_cast<int>(right_batch_size));
  auto right_pk_column = generator::MakeIndexColumn(static_cast<int>(num_batches),
                                                    static_cast<int>(right_batch_size))
                             .ValueOrDie();
  output.right_batches =
      generator::AddColumn("pk", right_batches_base, right_pk_column);
  output.right_schema = output.right_batches[0]->schema();

  std::vector<std::shared_ptr<arrow::Field>> left_fields;
  left_fields.push_back(
      arrow::field("y", arrow::uint32(), scenario.left_payload_nullable));
  left_fields.insert(left_fields.end(), scenario.extra_left_fields.begin(),
                     scenario.extra_left_fields.end());
  auto left_schema = arrow::schema(left_fields);
  auto left_batches_base =
      generator::MakeRandomRecordBatches(rng_left, left_schema, static_cast<int>(num_batches),
                                         static_cast<int>(left_batch_size));

  generator::ForeignKeyStats fk_stats;
  auto left_fk_result = generator::MakeForeignKeyColumn(
      rng_left, static_cast<uint32_t>(right_batch_size), static_cast<int32_t>(num_batches),
      static_cast<int32_t>(left_batch_size), scenario.fk_config, &fk_stats);
  if (!left_fk_result.ok()) {
    throw std::runtime_error(left_fk_result.status().ToString());
  }
  auto left_fk_column = left_fk_result.ValueOrDie();
  output.left_batches = generator::AddColumn("fk", left_batches_base, left_fk_column);
  output.left_schema = output.left_batches[0]->schema();
  output.stats = fk_stats;

  return output;
}

}  // namespace join
}  // namespace upmemeval
