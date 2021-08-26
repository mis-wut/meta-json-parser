#pragma once

#include "allocator.h"
#include <memory>
#include <cstddef>
#include <cuda_runtime.h>
#include <cstddef>
#include <array>


namespace meta_json { namespace memory {


	/**
	 * @brief Describes configuration of memory pools required for a memory configuration hierarchy given by its root
	 * config_node.
	 *
	 * Calculates which GPU memory segments will be utilized and the size and alignment of required memory buffers
	 * in each.
	 *
	 * @tparam RootConfigNode Root config_node of memory configuration hierarchy.
	*/
	template<typename RootConfigNode>
	struct pool
	{
	private:
		template<typename Segment>
		using init_pool = detail::subtree_init_allocations<RootConfigNode, Segment>;

		template<typename Segment>
		using init_pool_map = detail::buffer_map<init_pool<Segment>, boost::mp11::mp_size_t<0>>;

		template<typename Segment>
		using init_pool_map_idx = boost::mp11::mp_transform< //to each element of init_pool_map
			boost::mp11::mp_push_front, //append at the beginning
			init_pool_map<Segment>, //(elements are mp_list<Alloc, StartOffset>)
			boost::mp11::mp_iota<boost::mp11::mp_size<init_pool_map<Segment>>>>; //its index

		template<typename Segment>
		using init_pool_size = detail::buffer_end<init_pool<Segment>, boost::mp11::mp_size_t<0>>;

		template<typename Segment>
		using scratch_pool_end = detail::subtree_scratch_end<RootConfigNode, Segment, init_pool_size<Segment>>;

		template<typename Segment>
		using pool_size = scratch_pool_end<Segment>;

		template<typename Segment>
		using has_pool = boost::mp11::mp_to_bool<pool_size<Segment>>;

		template<typename Segment>
		using pool_alignment = detail::subtree_alignment<RootConfigNode, Segment>;

		using pool_list = boost::mp11::mp_list<segment::shared, segment::global, segment::constant>;

		using active_pools = boost::mp11::mp_filter<has_pool, pool_list>;

		static constexpr size_t pools_count = boost::mp11::mp_size<active_pools>::value;

		template<typename Segment>
		using pool_index = boost::mp11::mp_find<active_pools, Segment>;

		template<typename Segment>
		struct alignas(pool_alignment<Segment>::value) pool_buffer {
			std::array<std::byte, pool_size<Segment>::value> buffer;
		};

	public:
		template<typename Segment, typename ProcessDataFn>
		static void initialize_pool(ProcessDataFn fn)
		{
			auto host_mem = std::make_unique<pool_buffer<Segment>>();
			boost::mp11::mp_for_each<init_pool_map_idx<Segment>>([data = host_mem->buffer.data()](auto alloc)
			{
				using AllocTriple = decltype(alloc);
				constexpr size_t offset = boost::mp11::mp_third<AllocTriple>::value;
				using Alloc = boost::mp11::mp_second<AllocTriple>;
				using T = typename Alloc::type;
				using InitFn = typename Alloc::init_fn;

				//No need to worry about calling the destructor when done, since all pre-initialized allocations
				//need to be trivially copyable and therefore trivially destructible.
				T* dest = new (data + offset) T;
				InitFn{}(dest);
			});
			fn(host_mem->buffer.data());
		}

		template<typename Segment>
		static void initialize_pool(std::byte* data)
		{
			initialize_pool<Segment>([data](std::byte* host_data)
				{
					// TODO : check for cuda errors
					cudaMemcpy(data, host_data, init_pool_size<Segment>::value, cudaMemcpyDefault);
				});
		}

		template<typename Segment>
		static void initialize_pool(std::byte* data, cudaStream_t stream)
		{
			initialize_pool<Segment>([data, stream](std::byte* host_data)
				{
					// TODO : check for cuda errors
					cudaMemcpyAsync(data, host_data, init_pool_size<Segment>::value, cudaMemcpyDefault, stream);
				});
		}

		static constexpr bool has_shared = has_pool<segment::shared>::value;
		static constexpr bool has_global = has_pool<segment::global>::value;
		static constexpr bool has_constant = has_pool<segment::constant>::value;

		static constexpr size_t shared_size = pool_size<segment::shared>::value;
		static constexpr size_t global_size = pool_size<segment::global>::value;
		static constexpr size_t constant_size = pool_size<segment::constant>::value;

		static constexpr size_t shared_alignment = pool_alignment<segment::shared>::value;
		static constexpr size_t global_alignment = pool_alignment<segment::global>::value;
		static constexpr size_t constant_alignment = pool_alignment<segment::constant>::value;

		using shared_buffer = pool_buffer<segment::shared>;
		using global_buffer = pool_buffer<segment::global>;
		using constant_buffer = pool_buffer<segment::constant>;
	};
}}