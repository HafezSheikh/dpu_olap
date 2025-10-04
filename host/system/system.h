#include <cstdlib>
#include <string>
#include <thread>

namespace variables {

static inline int __getenv_int(std::string name, int fallback = 0) {
  if (const char* env_value = std::getenv(name.c_str())) {
    return std::stoi(env_value);
  }
  return fallback;
}

static inline double __getenv_double(std::string name, double fallback = 0.0) {
  if (const char* env_value = std::getenv(name.c_str())) {
    return std::stod(env_value);
  }
  return fallback;
}

static inline int max_dpus() { return __getenv_int("NR_DPUS", NR_DPUS); }

static inline int scale_factor() { return __getenv_int("SF", max_dpus()); }

static inline int bloom_bits_per_key() { return __getenv_int("BLOOM_BITS_PER_KEY", 3); }

static inline int bloom_min_partition_rows() {
  return __getenv_int("BLOOM_MIN_PARTITION_ROWS", 512);
}

static inline int bloom_max_partition_rows() {
  return __getenv_int("BLOOM_MAX_PARTITION_ROWS", 1 << 18);
}

static inline int bloom_min_bits() { return __getenv_int("BLOOM_MIN_BITS", 64); }

static inline int bloom_max_bits() { return __getenv_int("BLOOM_MAX_BITS", 1 << 18); }

static inline double bloom_min_mismatch_rate() {
  return __getenv_double("BLOOM_MIN_MISMATCH_RATE", 0.90);
}

static inline int max_threads() { return __getenv_int("MAX_THREADS", std::thread::hardware_concurrency()); }

static inline int max_threads_no_ht() { return __getenv_int("MAX_THREADS", std::thread::hardware_concurrency() / 2); }

}  // namespace variables
