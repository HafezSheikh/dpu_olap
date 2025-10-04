#include <benchmark/benchmark.h>
#include <iostream>
#include <string>
#include <vector>

#include "system/system.h"

int main(int argc, char** argv) {
  std::vector<std::string> forced_flags = {
      "--benchmark_out=join_benchmark_results.json",
      "--benchmark_out_format=json"};
  std::vector<char*> argv_extended;
  argv_extended.reserve(argc + forced_flags.size());
  for (int i = 0; i < argc; ++i) {
    argv_extended.push_back(argv[i]);
  }
  for (auto& flag : forced_flags) {
    argv_extended.push_back(const_cast<char*>(flag.c_str()));
  }
  int argc_extended = static_cast<int>(argv_extended.size());
  benchmark::Initialize(&argc_extended, argv_extended.data());
  if (::benchmark::ReportUnrecognizedArguments(argc_extended, argv_extended.data())) {
    return 1;
  }

  benchmark::AddCustomContext("NR_DPUS", std::to_string(variables::max_dpus()));
  benchmark::AddCustomContext("SF", std::to_string(variables::scale_factor()));
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
