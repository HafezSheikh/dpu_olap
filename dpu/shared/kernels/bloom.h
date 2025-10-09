#pragma once

#include <stddef.h>
#include <stdint.h>

#include "mram_ra.h"

/*************** Hot miss filter ******************/
#define HOT_FILTER_SIZE 512  // Must be power of two
#define HOT_FILTER_MASK (HOT_FILTER_SIZE - 1)
#define HOT_FILTER_EMPTY 0xFFFFFFFFu

typedef struct {
  uint32_t tag;
} hot_filter_entry_t;

typedef struct {
  hot_filter_entry_t entries[HOT_FILTER_SIZE];
  uint32_t hits;    // number of lookups answered from WRAM filter
  uint32_t misses;  // number of lookups that fell back to Bloom
} hot_filter_t;

static inline uint32_t hot_filter_hash(uint32_t key) {
  return key * 2654435761u;  // Knuth multiplicative hash
}

static inline void hot_filter_init(hot_filter_t* filter) {
  if (!filter) {
    return;
  }
  filter->hits = 0;
  filter->misses = 0;
  for (uint32_t i = 0; i < HOT_FILTER_SIZE; ++i) {
    filter->entries[i].tag = HOT_FILTER_EMPTY;
  }
}

static inline int hot_filter_contains(hot_filter_t* filter, uint32_t key) {
  if (!filter) {
    return 0;
  }
  uint32_t slot = hot_filter_hash(key) & HOT_FILTER_MASK;
  if (filter->entries[slot].tag == key) {
    filter->hits++;
    return 1;
  }
  filter->misses++;
  return 0;
}

static inline void hot_filter_insert(hot_filter_t* filter, uint32_t key) {
  if (!filter) {
    return;
  }
  uint32_t slot = hot_filter_hash(key) & HOT_FILTER_MASK;
  filter->entries[slot].tag = key;
}

/*************** Bloom filter helpers ******************/

typedef struct {
  __mram_ptr uint8_t* bytes;
  __mram_ptr uint32_t* words;
  uint32_t n_bits;
  uint32_t n_bytes;
  uint32_t n_words;
} bloom_t;

static inline void bloom_init(bloom_t* bf, __mram_ptr uint8_t* bits_ptr, uint32_t n_bits) {
  bf->bytes = bits_ptr;
  bf->words = (__mram_ptr uint32_t*)bits_ptr;
  bf->n_bits = n_bits;
  bf->n_bytes = (n_bits + 7) >> 3;
  bf->n_words = (n_bits + 31) >> 5;
}

static inline uint64_t bloom_splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static inline void bloom_set_bit(bloom_t* bf, uint32_t pos) {
  uint32_t word_idx = pos >> 5;
  if (word_idx >= bf->n_words) {
    return;
  }
  uint32_t bit_mask = 1u << (pos & 31);
  __dma_aligned uint64_t tmp;
  uint32_t word = mram_load32(&tmp, (size_t)&bf->words[word_idx]);
  if ((word & bit_mask) == 0) {
    word |= bit_mask;
    mram_modify32(&tmp, (size_t)&bf->words[word_idx], word);
  }
}

static inline void bloom_insert_u64(bloom_t* bf, uint64_t key) {
  if (bf->n_bits == 0) {
    return;
  }
  uint64_t h1 = bloom_splitmix64(key);
  uint64_t h2 = bloom_splitmix64(h1);
  uint32_t pos1 = (uint32_t)(h1 % bf->n_bits);
  uint32_t pos2 = (uint32_t)((h1 + h2) % bf->n_bits);
  bloom_set_bit(bf, pos1);
  bloom_set_bit(bf, pos2);
}

static inline int bloom_maybe_contains_u64(const bloom_t* bf, uint64_t key) {
  if (bf->n_bits == 0) {
    return 1;
  }
  uint64_t h1 = bloom_splitmix64(key);
  uint64_t h2 = bloom_splitmix64(h1);
  uint32_t pos1 = (uint32_t)(h1 % bf->n_bits);
  uint32_t pos2 = (uint32_t)((h1 + h2) % bf->n_bits);
  uint32_t word_idx1 = pos1 >> 5;
  uint32_t word_idx2 = pos2 >> 5;
  uint32_t bit_mask1 = 1u << (pos1 & 31);
  uint32_t bit_mask2 = 1u << (pos2 & 31);
  __dma_aligned uint64_t tmp;
  uint32_t word1 = mram_load32(&tmp, (size_t)&bf->words[word_idx1]);
  uint32_t word2 = mram_load32(&tmp, (size_t)&bf->words[word_idx2]);
  return (word1 & bit_mask1) != 0 && (word2 & bit_mask2) != 0;
}
