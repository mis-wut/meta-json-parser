#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <meta_json_parser/config.h>
#include <boost/mp11/function.hpp>
#include <random>
#include <limits>

template<class OutTypeT>
struct TestContext {
	thrust::host_vector<OutTypeT> h_correct;
	thrust::host_vector<char> h_input;
	thrust::host_vector<InputIndex> h_indices;
	thrust::device_vector<OutTypeT> d_correct;
	thrust::device_vector<char> d_input;
	thrust::device_vector<InputIndex> d_indices;

	TestContext(size_t testSize, size_t group_size)
	{
		using GenerateT = boost::mp11::mp_if_c<sizeof(OutTypeT) == 1, uint16_t, OutTypeT>;
		GenerateT MAX_VAL = std::numeric_limits<OutTypeT>::max() - 1;
		size_t MAX_UINT_LEN = (size_t)std::ceil(std::log10((double)MAX_VAL));
		if (MAX_UINT_LEN > group_size - 1)
		{
			MAX_VAL = 1;
			for (int i = 0; i < group_size - 1; ++i)
				MAX_VAL *= 10;
			MAX_VAL -= 1;
			MAX_UINT_LEN = group_size - 1;
		}
		std::minstd_rand rng;
		std::uniform_int_distribution<GenerateT> dist(1, MAX_VAL);
		h_input = thrust::host_vector<char>(testSize * MAX_UINT_LEN);
		h_correct = thrust::host_vector<OutTypeT>(testSize);
		h_indices = thrust::host_vector<InputIndex>(testSize + 1);
		std::generate(h_correct.begin(), h_correct.end(), [&dist, &rng]() { return static_cast<OutTypeT>(dist(rng)); });
		auto inp_it = h_input.data();
		auto ind_it = h_indices.begin();
		*ind_it = 0;
		++ind_it;
		for (auto& x : h_correct)
		{
			inp_it += snprintf(inp_it, MAX_UINT_LEN + 1, "%llu", static_cast<long long unsigned int>(x));
			*ind_it = (inp_it - h_input.data());
			++ind_it;
		}
		d_input = thrust::device_vector<char>(h_input.size() + 256); //256 to allow batch loading
		thrust::copy(h_input.begin(), h_input.end(), d_input.begin());
		d_correct = thrust::device_vector<OutTypeT>(h_correct);
		d_indices = thrust::device_vector<InputIndex>(h_indices);
	}
};
