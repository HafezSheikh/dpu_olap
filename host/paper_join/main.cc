#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include "dpuext/api.h"
#include "generator/generator.h"
#include "paper_join_dpu.h"

using upmemeval::generator::MakeForeignKeyColumn;
using upmemeval::generator::MakeIndexColumn;
using upmemeval::generator::MakeRandomRecordBatches;
using upmemeval::generator::AddColumn;

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

  arrow::random::RandomArrayGenerator rng_right(42);
  arrow::random::RandomArrayGenerator rng_left(1337);

  auto right_schema = arrow::schema({arrow::field("payload", arrow::uint32(), false)});
  auto right_batches_base =
      MakeRandomRecordBatches(rng_right, right_schema, static_cast<int>(opts.batches),
                              static_cast<int>(opts.batch_rows));
  auto right_pk_column =
      MakeIndexColumn(static_cast<int>(opts.batches), static_cast<int>(opts.batch_rows)).ValueOrDie();
  auto right_batches =
      AddColumn("pk", right_batches_base, right_pk_column);
  auto right_schema_with_pk = right_batches[0]->schema();

  auto left_schema = arrow::schema({arrow::field("payload", arrow::uint32(), false)});
  auto left_batches_base =
      MakeRandomRecordBatches(rng_left, left_schema, static_cast<int>(opts.batches),
                              static_cast<int>(opts.batch_rows));
  std::pair<uint64_t, uint64_t> counts{0, 0};
  auto left_fk_column =
      MakeForeignKeyColumn(rng_left, static_cast<uint32_t>(opts.batch_rows),
                           static_cast<int32_t>(opts.batches), static_cast<int32_t>(opts.batch_rows),
                           opts.fk_outside_ratio, &counts)
          .ValueOrDie();
  auto left_batches = AddColumn("fk", left_batches_base, left_fk_column);
  auto left_schema_with_fk = left_batches[0]->schema();

  std::cout << "Generated dataset: inside=" << counts.first
            << " outside=" << counts.second << std::endl;

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
  std::cout << "Matches=" << metrics.matches << " mismatches=" << metrics.mismatches
            << " bloom_skipped=" << metrics.bloom_skipped << std::endl;
  return 0;
}
