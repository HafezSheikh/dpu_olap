#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include "dpuext/api.h"
#include "generator/generator.h"
#include "join/join_profiles.h"
#include "paper_join_dpu.h"

namespace {

struct Options {
  int64_t batches = 4;
  int64_t batch_rows = 64 << 10;
  double fk_outside_ratio = 0.0;
  int nr_dpus = 4;
  double bloom_threshold = 0.9;
};

Options ParseOptions(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto pos = arg.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    std::string key = arg.substr(0, pos);
    std::string value = arg.substr(pos + 1);
    if (key == "--batches") {
      opts.batches = std::stoll(value);
    } else if (key == "--batch_rows") {
      opts.batch_rows = std::stoll(value);
    } else if (key == "--fk_outside_ratio") {
      opts.fk_outside_ratio = std::stod(value);
    } else if (key == "--nr_dpus") {
      opts.nr_dpus = std::stoi(value);
    } else if (key == "--bloom_threshold") {
      opts.bloom_threshold = std::stod(value);
    }
  }
  return opts;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts = ParseOptions(argc, argv);
  opts.fk_outside_ratio = std::clamp(opts.fk_outside_ratio, 0.0, 1.0);
  if (opts.batches <= 0 || opts.batch_rows <= 0 || opts.nr_dpus <= 0) {
    std::cerr << "Invalid arguments" << std::endl;
    return 1;
  }

  setenv("BLOOM_MIN_MISMATCH_RATE", std::to_string(opts.bloom_threshold).c_str(), 1);

  auto scenario =
      upmemeval::join::ResolveScenarioConfig(std::optional<double>(opts.fk_outside_ratio));
  upmemeval::join::GeneratedJoinData generated;
  try {
    generated = upmemeval::join::GenerateJoinData(
        scenario, opts.batches, opts.batch_rows, opts.batch_rows);
  } catch (const std::exception& e) {
    std::cerr << "Failed to generate data: " << e.what() << std::endl;
    return 1;
  }

  auto& left_batches = generated.left_batches;
  auto& right_batches = generated.right_batches;
  auto left_schema_with_fk = generated.left_schema;
  auto right_schema_with_pk = generated.right_schema;

  std::cout << "[Generator("
            << "paper," << upmemeval::join::ProfileTag(scenario.profile) << ")] batches="
            << opts.batches << " batch_rows=" << opts.batch_rows
            << " requested_outside_ratio=" << opts.fk_outside_ratio;
  double considered = static_cast<double>(generated.stats.inside + generated.stats.outside);
  double total = static_cast<double>(generated.stats.total());
  double actual_outside =
      considered > 0.0 ? static_cast<double>(generated.stats.outside) / considered : 0.0;
  double null_ratio = total > 0.0 ? static_cast<double>(generated.stats.nulls) / total : 0.0;
  double hot_ratio = generated.stats.inside > 0
                         ? static_cast<double>(generated.stats.hot) /
                               static_cast<double>(generated.stats.inside)
                         : 0.0;
  double hot_miss_ratio = generated.stats.outside > 0
                              ? static_cast<double>(generated.stats.miss_hot) /
                                    static_cast<double>(generated.stats.outside)
                              : 0.0;
  std::cout << " actual_outside_ratio=" << actual_outside << " null_ratio=" << null_ratio
            << " hot_share=" << hot_ratio << " hot_miss_share=" << hot_miss_ratio << std::endl;
  std::cout << "[Generator] profile_desc=" << upmemeval::join::ProfileDescription(scenario.profile)
            << std::endl;

  auto system = dpu::DpuSet::allocate(opts.nr_dpus,
                                      "backend=simulator,nrJobsPerRank=256,sgXferEnable=true");
  upmemeval::paper_join::PaperJoinDpu join(system, left_schema_with_fk,
                                           right_schema_with_pk, left_batches,
                                           right_batches);

  auto status = join.Prepare();
  if (!status.ok()) {
    std::cerr << "Prepare failed: " << status.ToString() << std::endl;
    return 1;
  }

  auto result = join.Run();
  if (!result.ok()) {
    std::cerr << "Run failed: " << result.status().ToString() << std::endl;
    return 1;
  }

  auto metrics = result.ValueOrDie();
  std::cout << "Total=" << metrics.total_probes << " matches=" << metrics.matches
            << " mismatches=" << metrics.mismatches
            << " bloom_skipped=" << metrics.bloom_skipped
            << " bloom_false_positive=" << metrics.bloom_false_positives << std::endl;
  return 0;
}
