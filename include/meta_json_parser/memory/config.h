#pragma once

#include <type_traits>
#include <boost/mp11.hpp>

namespace meta_json{ namespace memory
{
	namespace segment
	{
		/**
		 * @brief GPU global memory space tag.
		*/
		struct global { };
		/**
		 * @brief GPU shared memory space tag.
		*/
		struct shared { };
		/**
		 * @brief GPU constant memory space tag.
		*/
		struct constant { };
	}

	namespace detail {
		template<typename T, typename Segment, typename InitFn>
		using not_const_or_initialized = boost::mp11::mp_or<
											boost::mp11::mp_not<std::is_same<Segment, segment::constant>>,
											boost::mp11::mp_not<std::is_same<InitFn, void>>>;

		template<typename T, typename Pool, typename InitFn>
		using not_shared_or_not_initialized = boost::mp11::mp_or<
												boost::mp11::mp_not<std::is_same<Pool, segment::shared>>,
												std::is_same<InitFn, void>>;

		template<typename T, typename Pool, typename InitFn>
		using not_initialized_or_copyable = boost::mp11::mp_or<
												std::is_same<InitFn, void>,
												std::is_trivially_copyable<T>>;
		
	}

	/**
	 * @brief Compile-time allocation request description.
	 *
	 * Describes a single allocation. The data type determines the memory block size and alignment, array types can
	 * be used if more than one object is needed. Empty types will occupy no memory (as if having [[no_unique_address]]
	 * attribute).
	 *
	 * The Segment tag (which shall be one of the types from meta_json::memory::segment namespace) determines in which
	 * segment of GPU memory the memory block will be allocated in.
	 *
	 * The data type will be properly constructed, so that type constructors can be used to initialize the memory block.
	 *
	 * Alternatively an optional functor type can be specified and its function call operator will be called, passing a
	 * T* pointer to the start of the memory block, allowing for further initialization. The operator shall be a host
	 * function and will be executed once on the CPU before requesting kernel launches. Additionally such memory block
	 * is considered read-only and will be reused between all matching requests - kernels shall not modify
	 * pre-initialized memory blocks they request. InitFn can only be specified for global and constant memory segments.
	 *
	 * @tparam T Data type the memory needs to be allocated for.
	 * @tparam Segment GPU memory segment tag.
	 * @tparam InitFn Optional initialization functor, use void if not needed.
	*/
	template<typename T, typename Segment, typename InitFn = void>
	struct allocation
	{
		using type = T;
		using segment = Segment;
		using is_empty = std::is_empty<T>;
		using size = boost::mp11::mp_size_t<is_empty::value ? 0 : sizeof(T)>;
		using alignment = boost::mp11::mp_size_t<is_empty::value ? 1 : alignof(type)>;
		using init_fn = InitFn;
		using has_init = std::bool_constant<!std::is_same_v<init_fn, void>>;

		static_assert(detail::not_const_or_initialized<T, Segment, InitFn>::value,
					  "Constant memory must be pre-initialized");
		static_assert(detail::not_shared_or_not_initialized<T, Segment, InitFn>::value,
					  "Shared memory cannot be pre-initialized");
		static_assert(detail::not_initialized_or_copyable<T, Segment, InitFn>::value,
					  "Pre-initialized memory needs to be trivially copyable");
	};

	namespace detail
	{
		template<typename LAllocs, typename LSubNodes>
		struct config_node_base
		{
			using sub_nodes = LSubNodes;
			using allocations = LAllocs;
		};
	}

	/**
	 * @brief Describes one level in the configuration of parser memory requests hierarchy.
	 *
	 * One node lists allocation requests that can be satisfied independently of each other. Memory blocks obtained that
	 * way from this node or any of its ancestors can all be used at the same time. However, are allocations from this
	 * node or any of its descendants shall be freed before requesting allocation described by any of its sibling nodes.
	 *
	 * @tparam Allocs allocation describing a single memory request. Use mp_list of allocations if more than
	 *		   one is needed.
	 * @tparam SubNodes Zero or more descendant config_nodes.
	*/
	template<typename Allocs, typename... SubNodes>
	struct config_node : detail::config_node_base<boost::mp11::mp_list<Allocs>, boost::mp11::mp_list<SubNodes...>>
	{ };

	template<typename... Allocs, typename... SubNodes>
	struct config_node<boost::mp11::mp_list<Allocs...>, SubNodes...> 
		: detail::config_node_base<boost::mp11::mp_list<Allocs...>, boost::mp11::mp_list<SubNodes...>>
	{ };
}}
