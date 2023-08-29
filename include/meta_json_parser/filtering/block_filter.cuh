//
// Created by placek on 30.05.23.
//

#ifndef JSON_FILTERING_BLOCK_FILTER_CUH
#define JSON_FILTERING_BLOCK_FILTER_CUH

#include "cascade.hpp"

namespace filter {

    constexpr int THREAD_SIZE = 1024;

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

    __global__ void filter_block_per_json_kernel(const char *text, size_t num_of_jsons, const uint32_t *indices,
                                                 char *filter_result) {
        extern __shared__ char shared_json[];
        __shared__ bool current_filter_result;

        const auto json_start = blockIdx.x == 0 ? text : &text[indices[blockIdx.x - 1]] + 1;
        const auto json_end = &text[indices[blockIdx.x]];

        const auto json_length = json_end - json_start;

        const auto json_chunk_start = json_start + threadIdx.x * json_length / blockDim.x;
        auto json_chunk_end = json_start + (threadIdx.x + 1) * json_length / blockDim.x;
        json_chunk_end = json_chunk_end > json_end ? json_end : json_chunk_end;
        const auto json_chunk_length = json_chunk_end - json_chunk_start;

        // copy json chunk to shared memory
        for (auto ch = json_chunk_start; ch < json_chunk_end; ch++) {
            shared_json[ch - json_start] = *ch;
        }

        const auto shared_chunk_start = shared_json + (json_chunk_start - json_start);

        // check if pattern starts inside json chunk
        if (threadIdx.x == 0) {
            filter_result[blockIdx.x] = false;
            current_filter_result = false;
        }

        int parent = 0;

        while (parent < (1 << cascade::DNF) - 1) {
            const auto filter = cascade::tree_nodes[parent];

            if (threadIdx.x == 0) {
                current_filter_result = false;
            }

            __syncthreads();

            auto thread_result = is_pattern_present_in_text(shared_chunk_start, json_chunk_length + cascade::RF - 1,
                                                            filter);

            if (thread_result) {
                current_filter_result = true;
            }

            __syncthreads();


            if (current_filter_result) {
                // If right child is empty or out of range we have passed the cascade.
                int right = parent * 2 + 2;
                if (right >= (1 << cascade::DNF) - 1 ||
                    cascade::tree_nodes[right][0] == 0) {

                    filter_result[blockIdx.x] = true;

                    return;
                }
                parent = right;
            } else {
                // If left child is empty or out of range we have failed the cascade.
                int left = parent * 2 + 1;
                if (left >= (1 << cascade::DNF) - 1 ||
                    cascade::tree_nodes[left][0] == 0) {

                    filter_result[blockIdx.x] = false;

                    return;
                }
                parent = left;
            }
        }
    }

    struct block_filter {
        constexpr static char name[] = "Block filter";

        auto operator()(const char *text, size_t num_jsons, uint32_t *indices, char *out) {

            CUDA_CHECK(cudaFuncSetAttribute(filter_block_per_json_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            60 * 1024));

            filter_block_per_json_kernel<<<num_jsons, THREAD_SIZE, 60 * 1024>>>(
                    text,
                    num_jsons,
                    indices,
                    out);

            CUDA_KERNEL_FINISH();
        }
    };
}

#endif //JSON_FILTERING_BLOCK_FILTER_CUH
