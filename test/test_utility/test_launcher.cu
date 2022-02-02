//
// Created by lothedr on 29.01.2022.
//

#include "test_launcher.cuh"

const char *MismatchOutputBufferCount::what() const noexcept {
    return "Count of output buffers returned by test context is different than declared by BaseAction.";
}

const char *MismatchGroupSize::what() const noexcept {
    return "Group size provided in template parameter doesn't match group size used in test context.";
}
