#include "hash_build.h"

#include <alloc.h>
#include <assert.h>

#include "umq/cflags.h"
#include "umq/log.h"
#include "bloom.h"

int kernel_hash_build(uint32_t tasklet_id, __mram_ptr T* buffer, uint32_t buffer_length,
                      hash_table_t* hashtable, uint32_t bloom_n_bits, __mram_ptr uint8_t* bloom_bits ) {
  // Initialize a local WRAM cache for tasklet input
  T* input_cache = (T*)mem_alloc(BLOCK_SIZE_IN_BYTES);

  // Initialize bloom filter view (shared across tasklets via MRAM)
  bloom_t bf;
  if (bloom_n_bits > 0) {
    bloom_init(&bf, bloom_bits, bloom_n_bits);
  }

  // Scan blocks
  unsigned int tasklet_start_index = tasklet_id << BLOCK_LENGTH_LOG2;
  for (unsigned int block_offset = tasklet_start_index; block_offset < buffer_length;
       block_offset += BLOCK_LENGTH * NR_TASKLETS) {
    // Load block
    mram_read(&buffer[block_offset], input_cache, BLOCK_SIZE_IN_BYTES);

    uint32_t max = BLOCK_LENGTH;
    if (buffer_length >= block_offset && (buffer_length - block_offset) < max)
      max = buffer_length - block_offset;

    for (unsigned int i = 0; i < max; ++i) {
      // Add to map (from an array item's value to its index)
      T item = input_cache[i];
      uint32_t index = block_offset + i;
      bool ok = ht_put(hashtable, item, index);
      assert(ok);

      // Insert into bloom filter
      if (bloom_n_bits > 0) {
        bloom_insert_u64(&bf, (uint64_t)item);
      }
    }
  }
  return 0;
}
