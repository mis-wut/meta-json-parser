#pragma once

#include "cascade.hpp"
#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

namespace filter {

__host__ __device__ unsigned int constexpr upp2(unsigned int v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}

template <int WordSize>
__device__ __host__ __forceinline__ constexpr int find_group_size() {
  constexpr int up2v = upp2(WordSize);
  if constexpr (WordSize % 8 == 0) {
    // double doesn't make sense
    // as the memory bus is saturated enough
    // when loading floats
    return up2v / 4;
  } else if constexpr (WordSize % 4 == 0) {
    return up2v / 4;
  }
  return up2v;
}

template <int WordSize> __device__ __host__ constexpr auto comp_type() {
  if constexpr (WordSize % 8 == 0) {
    // 64b doesn't make sense
    // as the memory bus is saturated enough
    // when loading 32b
    return uint32_t{};
  } else if constexpr (WordSize % 4 == 0) {
    return uint32_t{};
  } else {
    return char{};
  }
}

__device__ bool run_filter(const char *json_start, const char *json_end,
                           const char *filter) {
  constexpr auto group_size = find_group_size<cascade::RF>();
  const auto grid = cg::this_grid();
  const auto tid = grid.thread_rank();
  const auto warp = cg::tiled_partition<group_size>(cg::this_thread_block());
  const auto lid = warp.thread_rank();

  bool result = true;
  int done = true;

  using CT = decltype(comp_type<cascade::RF>());
  constexpr int comp_len = sizeof(CT);
  CT f{0}, t{0};
  memcpy((void *)&f, (void *)&filter[comp_len * lid], comp_len);

  for (auto addr = json_start; addr < json_end - cascade::RF + 1; ++addr) {
    memcpy((void *)&t, (void *)&addr[comp_len * lid], comp_len);
    result = ((lid >= cascade::RF / comp_len) || (t == f));
    if constexpr (group_size > 1) {
      warp.sync();
      warp.match_all(result, done);
    } else {
      done = true;
    }
    if (done && result) {
      return true;
    }
  }
  return false;
}

__global__ void filter_warp_per_json(const char *text, size_t num_jsons,
                                     uint32_t *indices, char *out) {
  constexpr auto group_size = find_group_size<cascade::RF>();
  const auto grid = cg::this_grid();
  const auto tid = grid.thread_rank();
  const auto warp = cg::tiled_partition<group_size>(cg::this_thread_block());
  const auto lid = warp.thread_rank();
  const auto wid = tid / group_size;

  if (wid >= num_jsons) {
    return;
  }

  const auto json_start = (wid == 0) ? text : &text[indices[wid - 1]];
  const auto json_end = &text[indices[wid]];

  // cascade tree traversal
  int parent = 0;

  while (parent < (1 << cascade::DNF) - 1) {
    const auto filter = cascade::tree_nodes[parent];
    if (run_filter(json_start, json_end, filter)) {
      // If right child is empty or out of range we have passed the cascade.
      int right = parent * 2 + 2;
      if (right >= (1 << cascade::DNF) - 1 ||
          cascade::tree_nodes[right][0] == 0) {
        if (warp.thread_rank() == 0) {
          out[wid] = true;
        }
        return;
      }
      parent = right;
    } else {
      // If left child is empty or out of range we have failed the cascade.
      int left = parent * 2 + 1;
      if (left >= (1 << cascade::DNF) - 1 ||
          cascade::tree_nodes[left][0] == 0) {
        if (warp.thread_rank() == 0) {
          out[wid] = false;
        }
        return;
      }
      parent = left;
    }
  }

  if (warp.thread_rank() == 0) {
    out[wid] = false;
  }
}

struct gpu_filter {
  constexpr static char name[] = "Cooperative Group Filter";
  int WarpsPerBlock = 8;
  auto operator()(const char *text, size_t num_jsons, uint32_t *indices,
                  char *out) {
    filter_warp_per_json<<<(num_jsons + WarpsPerBlock - 1) / WarpsPerBlock,
                           WarpsPerBlock * 32>>>(text, num_jsons, indices, out);
  }
};

} // namespace filter
