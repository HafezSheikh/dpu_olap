#include "bloom_profile.h"

#include <alloc.h>
#include <barrier.h>
#include <mutex.h>

#include "bloom.h"
#include "umq/cflags.h"
#include "umq/log.h"

extern barrier_t barrier;
extern mutex_id_t bloom_mutex;

int kernel_bloom_profile(uint32_t tasklet_id, __mram_ptr T* buffer,
                         uint32_t buffer_length, hash_table_t* hashtable,
                         uint32_t bloom_n_bits, __mram_ptr uint8_t* bloom_bits,
                         bloom_profile_counters_t* out) {
  if (tasklet_id == 0) {
    out->total_probes = 0;
    out->bloom_skipped = 0;
    out->bloom_false_positives = 0;
    out->matches = 0;
  }
  barrier_wait(&barrier);

  T* input_cache = (T*)mem_alloc(BLOCK_SIZE_IN_BYTES);

  bloom_t bf;
  if (bloom_n_bits > 0) {
    bloom_init(&bf, bloom_bits, bloom_n_bits);
  }

  uint32_t local_total = 0;
  uint32_t local_skipped = 0;
  uint32_t local_false_positive = 0;
  uint32_t local_matches = 0;

  for (unsigned int block_offset = tasklet_id << BLOCK_LENGTH_LOG2;
       block_offset < buffer_length; block_offset += BLOCK_LENGTH * NR_TASKLETS) {
    mram_read(&buffer[block_offset], input_cache, BLOCK_SIZE_IN_BYTES);
    uint32_t max = BLOCK_LENGTH;
    if (buffer_length >= block_offset && (buffer_length - block_offset) < max) {
      max = buffer_length - block_offset;
    }
    for (unsigned int i = 0; i < max; ++i) {
      T item = input_cache[i];
      ++local_total;

      if (bloom_n_bits > 0 && !bloom_maybe_contains_u64(&bf, (uint64_t)item)) {
        ++local_skipped;
        continue;
      }

      uint32_t index;
      if (ht_get(hashtable, item, &index)) {
        ++local_matches;
      } else {
        ++local_false_positive;
      }
    }
  }

  if (local_total > 0) {
    mutex_lock(bloom_mutex);
    out->total_probes += local_total;
    out->bloom_skipped += local_skipped;
    out->bloom_false_positives += local_false_positive;
    out->matches += local_matches;
    mutex_unlock(bloom_mutex);
  }

  barrier_wait(&barrier);

  return 0;
}
