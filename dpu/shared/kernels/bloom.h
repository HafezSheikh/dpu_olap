// dpu/kernels/bloom.h
#pragma once
#include <stdint.h>
#include <stddef.h>

#include "mram_ra.h"

// Bloom filter stored as MRAM bit-array (byte and word views).
typedef struct {
  __mram_ptr uint8_t* bytes;       // raw byte pointer (for zeroing)
  __mram_ptr uint32_t* words;      // 32-bit view for atomic updates
  uint32_t n_bits;                 // m = number of bits
  uint32_t n_bytes;                // (m + 7) / 8
  uint32_t n_words;                // (m + 31) / 32
} bloom_t;

/* Initialize bloom instance (does not zero memory) */
static inline void bloom_init(bloom_t* bf, __mram_ptr uint8_t* bits_ptr, uint32_t n_bits) {
  bf->bytes = bits_ptr;
  bf->words = (__mram_ptr uint32_t*)bits_ptr;
  bf->n_bits = n_bits;
  bf->n_bytes = (n_bits + 7) >> 3;
  bf->n_words = (n_bits + 31) >> 5;
}

/* bit operations */
static inline void bloom_set_bit(bloom_t* bf, uint32_t pos) {
  uint32_t word_idx = pos >> 5;
  if (word_idx >= bf->n_words) {
    return;
  }
  uint32_t bit_mask = 1u << (pos & 31);

  __dma_aligned uint64_t cache;
  uint32_t word = mram_load32(&cache, (size_t)&bf->words[word_idx]);
  if ((word & bit_mask) != 0) {
    return;
  }
  word |= bit_mask;
  mram_modify32(&cache, (size_t)&bf->words[word_idx], word);
}

static inline int bloom_get_bit(bloom_t* bf, uint32_t pos) {
  uint32_t word_idx = pos >> 5;
  if (word_idx >= bf->n_words) {
    return 0;
  }
  uint32_t bit_mask = 1u << (pos & 31);
  uint32_t word = bf->words[word_idx];
  return (word & bit_mask) != 0;
}

/* simple splitmix64 as a fast hash for 64-bit keys */
static inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

/* insert key (uint64_t) into bloom with two derived hashes */
static inline void bloom_insert_u64(bloom_t* bf, uint64_t key) {
  uint64_t h1 = splitmix64(key);
  uint64_t h2 = splitmix64(h1);
  uint32_t pos1 = (uint32_t)(h1 % bf->n_bits);
  uint32_t pos2 = (uint32_t)((h1 + h2) % bf->n_bits);
  bloom_set_bit(bf, pos1);
  bloom_set_bit(bf, pos2);
}

/* membership test */
static inline int bloom_maybe_contains_u64(bloom_t* bf, uint64_t key) {
  uint64_t h1 = splitmix64(key);
  uint64_t h2 = splitmix64(h1);
  uint32_t pos1 = (uint32_t)(h1 % bf->n_bits);
  uint32_t pos2 = (uint32_t)((h1 + h2) % bf->n_bits);
  return bloom_get_bit(bf, pos1) && bloom_get_bit(bf, pos2);
}
