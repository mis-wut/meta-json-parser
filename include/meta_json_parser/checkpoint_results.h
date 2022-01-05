//
// Created by Jakub Narębski <jnareb@mat.umk.pl> on 22.12.2021.
//
#pragma once

#ifndef META_JSON_PARSER_CHECKPOINT_RESULTS_H
#define META_JSON_PARSER_CHECKPOINT_RESULTS_H

#include <vector>
#include <utility>
#include <string>

#include <cuda_runtime_api.h>

using checkpoint_event_t = std::pair<cudaEvent_t, std::string>;

void init_checkpoints(size_t reserve);
void checkpoint_event(cudaEvent_t event, cudaStream_t stream, std::string description);
void print_checkpoint_events();
void print_checkpoint_results();

#endif //META_JSON_PARSER_CHECKPOINT_RESULTS_H
