#include <alloc.h>
#include <assert.h>
#include <barrier.h>
#include "mram_alloc.h"

#include "umq/bitops.h"
#include "umq/cflags.h"
#include "umq/kernels.h"
#include "umq/log.h"

#include "kernels/common.h"
#include "kernels/partition.h"
#include "kernels/hash_build.h"
#include "kernels/hash_probe.h"
#include "kernels/take.h"

#include "kernels/bloom.h"
#include "kernels/common.h"
 // include this header
#include <stdio.h>

#if ENABLE_PERF
#include <perfcounter.h>
#endif

#define BUFFER_LENGTH ((1 << 20) + (2 << 15))                // 2MiB items + 32KiB
#define BUFFER_SIZE_IN_BYTES (BUFFER_LENGTH << T_SIZE_LOG2)  // 8MiB
// TODO: the input buffer is reused in the probing stage and can be larger at that stage
// #define BUFFER_LENGTH (4 << 20)  // 4MiB items but only 2MiB usable in hash build stage
// #define BUFFER_SIZE_IN_BYTES (BUFFER_LENGTH << T_SIZE_LOG2)  // 16MiB

#define HASHTABLE_CAPACITY (4 << 20)  // 4MiB (>32MiB allocated)

// kernel
__host enum Kernel kernel;

// inputs
__host kernel_partition_inputs_t INPUT;

// Input buffer (join key column from a table)
__mram_noinit T buffer[BUFFER_LENGTH];
__host uint32_t buffer_length = 0;

// Output selection indices vector
__mram_noinit uint32_t selection_indices_vector[BUFFER_LENGTH];
__host uint32_t selection_indices_vector_length = 0;

// Output buffer (take kernel)
__mram_noinit T output_buffer[BUFFER_LENGTH];
__host uint32_t output_buffer_length = 0;


// after existing __mram_noinit buffers
// Bloom filter bit array for this DPU
// We choose a compile-time max bits value or set it via host. For safety, allocate a decent sized array.
// Example: allow up to MAX_BLOOM_BITS bits (e.g., 1<<20 bits = 1M bits = 128 KiB).
// adjust as needed
__mram_noinit uint8_t bloom_bits[MAX_BLOOM_BITS / 8];
__host uint32_t bloom_n_bits = 0; // actual n_bits used (host will set)

// Shared hash table
hash_table_t hashtable;

// Synchronization
BARRIER_INIT(barrier, NR_TASKLETS);
MUTEX_INIT(writer_mutex);

// Outputs
__host uint32_t nb_cycles = 0;
__host kernel_partition_outputs_t OUTPUT;

// histogram for each partition
__host uint32_t* histogram_wram = 0;



void initialize_hash_table(uint32_t tasklet_id) {
  if (tasklet_id == 0) {
    mram_reset();
    ht_init(&hashtable, writer_mutex, HASHTABLE_CAPACITY);
    // zero bloom bits (host may have set bloom_n_bits earlier)
    if (bloom_n_bits > 0) {
      uint32_t nbytes = (bloom_n_bits + 7) >> 3;
      static __dma_aligned uint8_t zero_block[64] = {0};
      uint32_t offset = 0;
      while (offset < nbytes) {
        uint32_t chunk = nbytes - offset;
        if (chunk > sizeof(zero_block)) {
          chunk = sizeof(zero_block);
        }
        uint32_t chunk_aligned = ROUND_UP_TO_MULTIPLE_OF_8(chunk);
        mram_write(zero_block, &bloom_bits[offset], chunk_aligned);
        offset += chunk;
      }
      trace("Bloom filter enabled: %u bits (%u bytes)\n", bloom_n_bits, nbytes);
    } else {
      trace("Bloom filter disabled for this partition\n");
    }
    trace("Allocated hash table with capacity %d (%d bytes allocated)\n",
          HASHTABLE_CAPACITY, ht_allocated_bytes(&hashtable));
  }
  barrier_wait(&barrier);
}

int main() {
  uint32_t tasklet_id = me();

#if ENABLE_PERF
  if (tasklet_id == 0) {
    perfcounter_config(COUNT_INSTRUCTIONS, true);
  }
#endif
  trace("Tasklet %d main: running kernel %d\n", tasklet_id, kernel);

  // Reset heap
  if (tasklet_id == 0) {
    mem_reset();
    if (kernel == KernelPartition) {
      assert("nr_partitions must be at least 2" && INPUT.nr_partitions >= 2);
      assert("nr_partitions must be a power of two" &&
            INPUT.nr_partitions == ROUND_UP_TO_POWER_OF_2(INPUT.nr_partitions));
    }
  }
  barrier_wait(&barrier);

  int result;
  switch (kernel) {
    case KernelPartition:
      result = kernel_partition(tasklet_id, &barrier, INPUT.nr_partitions,
                                &histogram_wram,
                                selection_indices_vector, buffer,
                                output_buffer, INPUT.buffer_length, INPUT.output_shift);

      // update meta-data for the host to know where to retrieve the 
      // histogram
      OUTPUT.partitions_metadata_offset = (uintptr_t)histogram_wram;
      OUTPUT.output_buffer_length = INPUT.buffer_length;

      break;
    case KernelHashBuild:
      initialize_hash_table(tasklet_id);
      result = kernel_hash_build(tasklet_id, buffer, buffer_length, &hashtable, bloom_n_bits, bloom_bits);
      if (tasklet_id == 0) {
        trace("Hash build completed: buffer_length=%u bloom_bits=%u\n", buffer_length, bloom_n_bits);
      }
      // if(tasklet_id == 0) {
      //   printf("Input: ");
      //   for (int i = 0; i < 10; i++) {
      //     printf("%d ", buffer[i]);
      //   }
      //   printf("\n");
      // }
#if HT_ENABLE_STATS
      if (tasklet_id == 0) {
        log("hash table buckets = %d\n", hashtable.n_buckets);
        log("hash table stats_total_distance = %d\n", hashtable.stats_total_distance);
        log("hash table stats_total_items = %d\n", hashtable.stats_total_items);
        log("hash table stats_total_slowpath = %d\n", hashtable.stats_total_slowpath);
      }
#endif
      break;
    case KernelHashProbe:
      result = kernel_hash_probe(tasklet_id, buffer, buffer_length, &hashtable,
                                 selection_indices_vector, bloom_n_bits, bloom_bits );
      selection_indices_vector_length = buffer_length;
      if (tasklet_id == 0) {
        trace("Hash probe completed: buffer_length=%u bloom_bits=%u\n", buffer_length, bloom_n_bits);
      }
      break;
    case KernelTake:
      result = kernel_take(tasklet_id, buffer, output_buffer, selection_indices_vector, buffer_length);
      output_buffer_length = buffer_length;
      break;
    default:
      log("Unknown kernel");
      result = 1;
      break;
  }

#if ENABLE_PERF
  nb_cycles = perfcounter_get();
#endif

  return result;
}
