#pragma once

#include <mram.h>
#include <stdint.h>

#include "common.h"
#include "bloom.h"
#include "hashtable/hashtable.h"
#include "umq/kernels.h"

int kernel_bloom_profile(uint32_t tasklet_id, __mram_ptr T* buffer,
                         uint32_t buffer_length, hash_table_t* hashtable,
                         uint32_t bloom_n_bits, __mram_ptr uint8_t* bloom_bits,
                         bloom_profile_counters_t* out);
