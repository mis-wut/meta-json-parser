//
// Created by placek on 29.08.23.
//

#pragma once

#include "cascade.hpp"

#include <cstdint>

namespace filter {

    __device__ void compute_lps_array(const char *pattern, size_t pattern_size, int *lps) {
        int len = 0;
        lps[0] = 0;

        int i = 1;
        while (i < pattern_size) {
            if (pattern[i] == pattern[len]) {
                len++;
                lps[i] = len;
                i++;
            } else {
                if (len != 0) {
                    len = lps[len - 1];
                } else {
                    lps[i] = 0;
                    i++;
                }
            }
        }
    }

    __device__ bool is_pattern_present_in_text(const char *text, size_t text_size, const char *pattern) {
        constexpr auto pattern_size = cascade::RF;
        int lps[pattern_size];

        compute_lps_array(pattern, pattern_size, lps);

        int i = 0;
        int j = 0;
        while ((text_size - i) >= (pattern_size - j)) {
            if (pattern[j] == text[i]) {
                j++;
                i++;
            }

            if (j == pattern_size) {
                return true;
            } else if (i < text_size && pattern[j] != text[i]) {
                if (j != 0)
                    j = lps[j - 1];
                else
                    i = i + 1;
            }
        }

        return false;
    }

    __global__ void filter_thread_per_json_kernel(const char *text, size_t num_of_jsons, const uint32_t *indices,
                                                  char *filter_result) {
        const auto tid = (blockIdx.x * blockDim.x + threadIdx.x);
        if (tid >= num_of_jsons) {
            return;
        }

        const auto json_start = tid == 0 ? text : &text[indices[tid - 1]] + 1;
        const auto json_end = &text[indices[tid]];

        const auto json_length = json_end - json_start;

        int parent = 0;

        while (parent < (1 << cascade::DNF) - 1) {
            const auto filter = cascade::tree_nodes[parent];
            if (is_pattern_present_in_text(json_start, json_length, filter)) {
                // If right child is empty or out of range we have passed the cascade.
                int right = parent * 2 + 2;
                if (right >= (1 << cascade::DNF) - 1 ||
                    cascade::tree_nodes[right][0] == 0) {
                    filter_result[tid] = true;
                    return;
                }
                parent = right;
            } else {
                // If left child is empty or out of range we have failed the cascade.
                int left = parent * 2 + 1;
                if (left >= (1 << cascade::DNF) - 1 ||
                    cascade::tree_nodes[left][0] == 0) {
                    filter_result[tid] = false;
                    return;
                }
                parent = left;
            }
        }
    }

    struct thread_filter {
        constexpr static char name[] = "Thread Filter";

        constexpr static int ThreadSize = 1024;

        auto operator()(const char *text, size_t num_jsons, uint32_t *indices, char *out) {
            filter_thread_per_json_kernel<<<(num_jsons - 1) / ThreadSize + 1, ThreadSize>>>(text, num_jsons,
                                                                                            indices, out);
        }
    };
}
