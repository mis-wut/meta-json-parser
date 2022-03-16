#include <iostream>
#include <gtest/gtest.h>

class CudaInit : public ::testing::Environment {
    void SetUp() override {
        std::cout << "Building kernels from PTX is happening right now. Please wait.\n";
        cudaFree(nullptr);
        std::cout << "Kernels build.\n";
    }
};

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new CudaInit);
    return RUN_ALL_TESTS();
}
