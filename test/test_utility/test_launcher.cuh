#ifndef META_JSON_PARSER_TEST_LAUNCHER_CUH
#define META_JSON_PARSER_TEST_LAUNCHER_CUH
#include <exception>
#include <boost/mp11/integral.hpp>
#include <thrust/device_vector.h>
#include <meta_json_parser/runtime_configuration.cuh>
#include <meta_json_parser/parser_configuration.h>
#include <meta_json_parser/parser_kernel.cuh>
#include <meta_json_parser/parser_output_device.cuh>
#include <meta_json_parser/kernel_launch_configuration.cuh>
#include "contexts/test_context.cuh"

class MismatchOutputBufferCount : public std::exception {
    const char * what() const noexcept override;
};

class MismatchGroupSize : public std::exception {
    const char * what() const noexcept override;
};

template<class BaseActionT, class GroupSizeT>
void LaunchTest(TestContext& context) {
    static_assert((GroupSizeT::value != 0), "GroupSizeT cannot be 0.");
    static_assert((1024 % GroupSizeT::value) == 0, "1024 must be divisible by GroupSizeT.");
    if (GroupSizeT::value != context.GroupSize()) {
        throw MismatchGroupSize();
    }
    using BaseAction = BaseActionT;
    using GroupSize = GroupSizeT;
    using GroupCount = boost::mp11::mp_int<1024 / GroupSize::value>;
    using RT = RuntimeConfiguration<GroupSize, GroupCount>;
    using PC = ParserConfiguration<RT, BaseAction>;
    using PK = ParserKernel<PC>;
    using M3 = typename PK::M3;
    using MemoryBuffer = typename M3::ReadOnlyBuffer;
    using Output = ParserOutputDevice<BaseAction>;
    constexpr size_t OUTPUT_BUFFERS_COUNT = boost::mp11::mp_size<
        typename OutputConfiguration<BaseAction>::RequestList
    >::value;

    thrust::device_vector<uint8_t> d_read_only_buffer(sizeof(MemoryBuffer));
    thrust::host_vector<void*> h_output_buffers = context.OutputBuffers();
    thrust::device_vector<void*> d_output_buffers(h_output_buffers);

    if (h_output_buffers.size() != OUTPUT_BUFFERS_COUNT) {
        throw MismatchOutputBufferCount();
    }

    KernelLaunchConfiguration launch_configuration;
    context.SetupConfiguration(launch_configuration);

    Output d_outputs(&launch_configuration, context.TestSize());

    PK pk(&launch_configuration);

    pk.Run(
        context.InputData().data().get(),
        context.InputIndices().data().get(),
        context.Errors().data().get(),
        d_output_buffers.data().get(),
        context.TestSize(),
        h_output_buffers.data()
    );
    cudaDeviceSynchronize();

    context.Validate();
}

#endif //META_JSON_PARSER_TEST_LAUNCHER_CUH
