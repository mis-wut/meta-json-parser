#pragma once

namespace cascade {
constexpr int DNF = 4;
constexpr int RF = 5;

constexpr __device__ char tree_nodes[(1 << DNF) - 1][RF + 1]{
    "\0", "\0", "\0", "\0", "\0", "\0", "\0", "\0",
    "\0", "\0", "\0", "\0", "\0", "\0", "\0"};
} // namespace cascade
