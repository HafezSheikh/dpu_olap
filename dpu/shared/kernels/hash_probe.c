#include "hash_probe.h"

#include <alloc.h>
#include <assert.h>

#include "umq/cflags.h"
#include "umq/log.h"

#include "bloom.h"
#include "common.h"  // for bloom_bits and bloom_n_bits
#include <barrier.h>
#include <mutex.h>

extern mutex_id_t bloom_mutex;
extern uint32_t bloom_skipped;
extern barrier_t barrier;
extern uint32_t hot_filter_hits[NR_TASKLETS];
extern uint32_t hot_filter_misses[NR_TASKLETS];

int kernel_hash_probe(uint32_t tasklet_id, __mram_ptr T* buffer, uint32_t buffer_length,
                      hash_table_t* hashtable,
                      __mram_ptr uint32_t* selection_indices_vector, uint32_t bloom_n_bits, __mram_ptr uint8_t* bloom_bits) {
  trace("Tasklet %d kernel_hash_probe\n", tasklet_id);
  // Initialize a local WRAM cache for tasklet input
  T* input_cache = (T*)mem_alloc(BLOCK_SIZE_IN_BYTES);
  uint32_t* output_cache = (uint32_t*)mem_alloc(BLOCK_LENGTH * sizeof(uint32_t));

  if (tasklet_id == 0) {
    bloom_skipped = 0;
  }
  barrier_wait(&barrier);

  // Bloom filter view
  bloom_t bf;
  hot_filter_t* hot_filter = NULL;
  if (bloom_n_bits > 0) {
    bloom_init(&bf, bloom_bits, bloom_n_bits);
    hot_filter = (hot_filter_t*)mem_alloc(sizeof(hot_filter_t));
    hot_filter_init(hot_filter);
  }

  uint32_t skipped_local = 0;

  // Scan blocks
  trace("Tasklet %d kernel_hash_probe: scanning blocks\n", tasklet_id);
  for (unsigned int block_offset = tasklet_id << BLOCK_LENGTH_LOG2;
       block_offset < buffer_length; block_offset += BLOCK_LENGTH * NR_TASKLETS) {
    // Load block
    mram_read(&buffer[block_offset], input_cache, BLOCK_SIZE_IN_BYTES);

    // Scan block and write to output cache
    uint32_t index;
    uint32_t max = BLOCK_LENGTH;
    if(buffer_length >= block_offset && (buffer_length - block_offset) < max)
      max = buffer_length - block_offset;

    for (unsigned int i = 0; i < max; ++i) {
      T item = input_cache[i];
      if (hot_filter && hot_filter_contains(hot_filter, (uint32_t)item)) {
        skipped_local++;
        output_cache[i] = UINT32_MAX;
        continue;
      }

      // Bloom check first
      if (bloom_n_bits > 0 && !bloom_maybe_contains_u64(&bf, (uint64_t)item)) {
        if (hot_filter) {
          hot_filter_insert(hot_filter, (uint32_t)item);
        }
        // Definitely not present
        skipped_local++;
        output_cache[i] = UINT32_MAX;  // sentinel for "no match"
        continue;
      }

      // Hash table probe
      bool ok = ht_get(hashtable, item, &index);
      if (ok) {
        output_cache[i] = index;
      } else {
        if (hot_filter) {
          hot_filter_insert(hot_filter, (uint32_t)item);
        }
        output_cache[i] = UINT32_MAX;  // handle not found
      }
    }

    mram_write(output_cache, &selection_indices_vector[block_offset],
               BLOCK_LENGTH * sizeof(uint32_t));
  }

  trace("Tasklet %d kernel_hash_probe: done\n", tasklet_id);

  barrier_wait(&barrier);
  if (skipped_local > 0) {
    mutex_lock(bloom_mutex);
    bloom_skipped += skipped_local;
    mutex_unlock(bloom_mutex);
  }
  if (bloom_n_bits > 0 && hot_filter != NULL) {
    uint32_t total = hot_filter->hits + hot_filter->misses;
    float rate = total ? (100.0f * (float)hot_filter->hits / (float)total) : 100.0f;
    trace("Tasklet %d hot miss filter hits=%u misses=%u hit_rate=%.2f%%\n", tasklet_id,
          hot_filter->hits, hot_filter->misses, rate);
    hot_filter_hits[tasklet_id] = hot_filter->hits;
    hot_filter_misses[tasklet_id] = hot_filter->misses;
  } else {
    hot_filter_hits[tasklet_id] = 0;
    hot_filter_misses[tasklet_id] = 0;
  }
  if (tasklet_id == 0) {
    trace("Tasklet %d bloom skipped local=%u total=%u\n", tasklet_id, skipped_local, bloom_skipped);
  }
  barrier_wait(&barrier);

  return 0;
}
