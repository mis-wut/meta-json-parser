#pragma once

#include "config.h"

namespace meta_json { namespace memory {
	namespace detail
	{
		/**
		 * @brief Checks if given allocation belongs to given GPU memory segment.
		 *
		 * @tparam Segment GPU memory segment tag.
		*/
		template<typename Segment>
		struct segment_predicate
		{
			/**
			 * @brief Alias for boolean type constant determining if the allocation matches filter's criteria.
			 *
			 * @tparam Alloc allocation type to test.
			*/
			template<typename Alloc>
			using fn = boost::mp11::mp_same<Segment, typename Alloc::segment>;
		};

		/**
		 * @brief Checks if given allocation belongs to given GPU memory segment and whether it requires
		 *		  pre-initialization or not.
		 *
		 * @tparam Segment GPU memory segment tag.
		 * @tparam HasInit Boolean type constant (e.g. std::boolean_constant) describing if the allocation should
		 *				   require pre-initialization or not.
		*/
		template<typename Segment, typename HasInit>
		struct segment_init_predicate
		{
			/**
			 * @brief Alias for boolean type constant determining if the allocation matches filter's criteria.
			 *
			 * @tparam Alloc allocation type to test.
			*/
			template<typename Alloc>
			using fn = boost::mp11::mp_and<boost::mp11::mp_same<Segment, typename Alloc::segment>,
				boost::mp11::mp_same<HasInit, typename Alloc::has_init>>;
		};

		/**
		 * @brief Given a list of allocations, filters out all that don't belong to given segment or require
		 *		  pre-initialization.
		 *
		 * Such allocations are default-constructed each time the memory is requested and kernels are free to modify
		 * their contents.
		 *
		 * @tparam LAllocs Input list of allocations.
		 * @tparam Segment GPU memory segment tag all resulting allocation should belong to.
		*/
		template<typename LAllocs, typename Segment>
		using scratch_allocations = boost::mp11::mp_copy_if_q<LAllocs,
			segment_init_predicate<Segment, std::false_type>>;

		/**
		 * @brief Given a list of allocations, filters out all that don't belong to given segment or do not require
		 *		  pre-initialization.
		 *
		 * Such allocations are initialized once before any kernel that requires them is launched using a provided
		 * functor type. Their memory blocks are considered constant from device's point of view and the kernels shall
		 * not modify them.
		 *
		 * @tparam LAllocs Input list of allocations.
		 * @tparam Segment GPU memory segment tag all resulting allocation should belong to.
		*/
		template<typename LAllocs, typename Segment>
		using init_allocations = boost::mp11::mp_copy_if_q<LAllocs, segment_init_predicate<Segment, std::true_type>>;

		template<typename Node, typename Segment>
		struct subtree_init_allocations_impl;

		/**
		 * @brief Traverses the memory configuration hierarchy given by root config_node and obtains a list of all
		 *		  unique memory requests from a given memory segment that require pre-initialization.
		 *
		 * @tparam Node Root config_node of the memory configuration hierarchy.
		 * @tparam Segment GPU memory segment tag
		*/
		template<typename Node, typename Segment>
		using subtree_init_allocations = typename subtree_init_allocations_impl<Node, Segment>::type;

		template<typename Node, typename Pool>
		struct subtree_init_allocations_impl
		{
			//Get all pre-initialized allocations in segment on this level only.
			using node_allocs = boost::mp11::mp_unique<init_allocations<typename Node::allocations, Pool>>;

			//Prepare to recursively execute the search with the same Segment
			using sub_impl = boost::mp11::mp_bind_back<subtree_init_allocations, Pool>;

			//Recursively obtain lists of allocations from all sub-nodes and put those in a list.
			using sub_alloc_lists = boost::mp11::mp_transform_q<sub_impl, typename Node::sub_nodes>;

			//Prepend a list of allocations from this node to the list of lists from previous step.
			using alloc_lists = boost::mp11::mp_push_front<sub_alloc_lists, node_allocs>;

			//Flatten the list of lists from previous step and remove duplicates.
			using type = boost::mp11::mp_apply<
				boost::mp11::mp_set_union,
				alloc_lists
			>;
		};

		/**
		 * @brief Transforms an allocation request to [size, alignment] pair
		 *
		 * Alias for mp_list with two integer type constants representing allocation size and its required alignment.
		 *
		 * @tparam Alloc allocation request
		*/
		template<typename Alloc>
		using size_align_t = boost::mp11::mp_list<typename Alloc::size, typename Alloc::alignment>;

		/**
		 * @brief Transforms a list of allocations into a list of [size, alignment] pairs
		 *
		 * Alias for a list of pairs (mp_lists with two elements) with required size and alignment for each allocation
		 * request in the input.
		 * @tparam LAllocs Input list of allocations
		*/
		template<typename LAllocs>
		using size_align_l = boost::mp11::mp_transform<size_align_t, LAllocs>;

		/**
		 * @brief Pads an offset to a given alignment boundary
		 *
		 * Alias for integer type constant representing the offset after padding.
		 *
		 * @tparam Offset Integer type constant representing initial offset that might require padding
		 * @tparam Alignment Integer type constant representing the required alignment of the result (assumed to be > 0)
		*/
		template<typename Offset, typename Alignment>
		using aligned_offset = boost::mp11::mp_size_t<(Offset::value + Alignment::value - 1) & ~(Alignment::value - 1)>;

		template<typename Offset, typename SizeAlignT>
		struct start_end
		{
			using start = boost::mp11::mp_if_c<
				boost::mp11::mp_first<SizeAlignT>::value == 0,
				Offset,
				aligned_offset<Offset, boost::mp11::mp_second<SizeAlignT>>>;
			using end = boost::mp11::mp_plus<start, boost::mp11::mp_first<SizeAlignT>>;
			using type = boost::mp11::mp_list<start, end>;
		};

		/**
		 * @brief Calculates the beginning and past-the-end offset for a given allocation request.
		 *
		 * Alias for a pair (in boost::mp11 sense) of integer type constants representing the offset of the first
		 * and (last+1) byte where an allocation of given size and alignment requirement can be placed past the starting
		 * offset
		 *
		 * @tparam Offset Starting offset on, or past which the allocation should be positioned.
		 * @tparam SizeAlignT A pair of integer type constants specifying the allocation required size and alignment.
		*/
		template<typename Offset, typename SizeAlignT>
		using start_end_t = typename start_end<Offset, SizeAlignT>::type;

		/**
		 * @brief Creates the beginning and past-the-end offset pair to be appended to the list of allocation positions.
		 *
		 * All positions are stored as a pair of offsets (begin and one-past-end). However, from the previous element's
		 * position only the second offset is used. The starting position of the appended allocation is determined to be
		 * on or past that offset according to alignment requirements. The end position is simply the starting position
		 * plus allocation size. This is an alias for a pair of integer type constants representing those two values.
		 *
		 * @tparam PrevStartEndT A pair of integer type constants describing the position of the previous allocation
		 * @tparam SizeAlignT A pair of integer type constants representing the size and alignment requirements of
		 *					  the allocation for which the position is to be determined.
		*/
		template<typename PrevStartEndT, typename SizeAlignT>
		using start_end_append = start_end_t<boost::mp11::mp_second<PrevStartEndT>, SizeAlignT>;

		/**
		 * @brief Calculates the map of positions of a list of allocations in a buffer
		 *
		 * Type alias for a list of integer type constant pairs describing the positions of allocations from the input
		 * list. Each allocation is placed on or past the starting offset, one after another, according to its alignment
		 * requirement. Positions are stored as offsets to the first and (last+1) bytes dedicated to given allocation.
		 *
		 * @tparam LAllocs Input list of allocation requests.
		 * @tparam StartOffset Starting offset.
		*/
		template<typename LAllocs, typename StartOffset>
		using start_end_l = boost::mp11::mp_partial_sum<
			// Convert allocations to size-alignment pairs
			size_align_l<LAllocs>,
			//starting with the dummy pair excluding the part of the buffer before the starting
			//offset (which will not be included in the result of partial_sum)
			boost::mp11::mp_list<boost::mp11::mp_size_t<0>, StartOffset>,
			//append all allocations one after another
			start_end_append>;

		template<typename LAllocs, typename StartOffset>
		using buffer_end_impl = boost::mp11::mp_second<boost::mp11::mp_back<start_end_l<LAllocs, StartOffset>>>;

		/**
		 * @brief Calculates the buffer size given the starting offset and a list of allocations.
		 *
		 * Alias for integer type constant representing the size of the buffer such that all allocations in the input
		 * list can be placed past the starting offset, one after another, according to their alignment requirements.
		 *
		 * @tparam LAllocs Input list of allocation requests.
		 * @tparam StartOffset Integer type constant representing the start offset within the buffer.
		*/
		template<typename LAllocs, typename StartOffset>
		using buffer_end = boost::mp11::mp_eval_if<
			boost::mp11::mp_empty<LAllocs>,
			StartOffset,
			buffer_end_impl, LAllocs, StartOffset>;

		template<typename LAllocs, typename StartOffset>
		using buffer_map = boost::mp11::mp_transform<boost::mp11::mp_list, //zip all
			LAllocs, //allocation requests
			//with the their starting offsets
			boost::mp11::mp_transform<
			boost::mp11::mp_first, //take the first element (the starting offset)
			start_end_l<LAllocs, StartOffset>>>;//from each begin-end position pair

		/**
		 * @brief Calculates the size of the pre-initialized allocations buffer within a given GPU memory segment
		 *
		 * Alias for an integer type constant representing the size of a buffer needed to store all pre-initialized
		 * memory requests in a specific GPU memory segment for given memory configuration hierarchy represented
		 * by the root config_node.
		 *
		 * @tparam Node Root config_node of memory configuration.
		 * @tparam Segment GPU memory segment tag.
		*/
		template<typename Node, typename Segment>
		using subtree_init_end = buffer_end<subtree_init_allocations<Node, Segment>, boost::mp11::mp_size_t<0>>;

		template<typename Node, typename Segment, typename StartOffset>
		struct subtree_scratch_end_impl;

		/**
		 * @brief Calculates the offset past the end of the scratch allocations in a given memory configuration
		 *		  hierarchy.
		 *
		 * Traverses the memory configuration hierarchy considering only scratch allocations (i.e. those that do not
		 * require pre-initialization) belonging to a specific GPU memory segment.
		 *
		 * For each path from root to leaf in the hierarchy determines the size of a buffer such that all requested
		 * allocations in each node along the path are:
		 *	- placed after the starting offset,
		 *	- non-overlapping,
		 *	- complying with their alignment requirements,
		 *	- positioned after any allocation from ancestor nodes an before any allocations from descendants.
		 *
		 * This is a type alias for an integer type constant representing the largest buffer size found during
		 * the search.
		 *
		 * @tparam Node Root config_node of memory configuration hierarchy.
		 * @tparam Segment GPU memory segment tag.
		 * @tparam StartOffset Starting offset (either the offset after scratch allocations of a subtree's parent
		 *					   or the size of pre-initialized allocations part in a buffer).
		*/
		template<typename Node, typename Segment, typename StartOffset>
		using subtree_scratch_end = typename subtree_scratch_end_impl<Node, Segment, StartOffset>::type;

		template<typename Node, typename Segment, typename StartOffset>
		struct subtree_scratch_end_impl
		{
			//Get scratch allocations in given segment on this level only
			using node_allocs = scratch_allocations<typename Node::allocations, Segment>;

			//Get buffer size for them.
			using node_scratch_end = buffer_end<node_allocs, StartOffset>;

			//Prepare to recursively execute the search with the same Segment, placing sub-node allocations past
			//node_scratch_end.
			using sub_impl = boost::mp11::mp_bind_back<subtree_scratch_end_impl, Segment, node_scratch_end>;

			//Obtain list of possible buffer sizes by recursive search of all sub-trees
			using sub_scratch_ends = boost::mp11::mp_transform_q<sub_impl, typename Node::sub_nodes>;

			//Append node_scratch_end as a sentinel in case there are no sub nodes or their scratch pools are empty
			using all_ends = boost::mp11::mp_push_front<sub_scratch_ends, node_scratch_end>;

			//Recursively obtain max buffer end offset from the list
			using type = boost::mp11::mp_max_element<all_ends, boost::mp11::mp_less>;
		};

		/**
		 * @brief Transforms allocation into integer type constant representing its alignment requirement.
		 *
		 * @tparam Alloc allocation to transform.
		*/
		template<typename Alloc>
		using align_t = typename Alloc::alignment;

		/**
		 * @brief Obtains maximum alignment requirement of all allocations in a list, or 1 if the list is empty.
		 *
		 * @tparam LAllocs Input list of allocations.
		*/
		template<typename LAllocs>
		using align_l = boost::mp11::mp_max_element< //get max alignment requirement
			boost::mp11::mp_push_front< //from list that contains
			boost::mp11::mp_transform<align_t, LAllocs>, //alignment requirements of each allocation
			boost::mp11::mp_size_t<1>>, //and a sentinel value 1 in case list is empty
			boost::mp11::mp_less>;

		template<typename Node, typename Segment>
		struct subtree_alignment_impl;

		template<typename Node, typename Segment>
		using subtree_alignment = typename subtree_alignment_impl<Node, Segment>::type;

		template<typename Node, typename Segment>
		struct subtree_alignment_impl
		{
			// TODO : Similar pattern repeated three times - here, subtree_scratch_end_impl and
			//		  subtree_init_allocations_impl. Consider refactoring.

			//Get all allocations from the same segment (regardless of their pre-init status)
			using node_allocs = boost::mp11::mp_copy_if_q<typename Node::allocations, segment_predicate<Segment>>;

			//Get alignment requirement for that list (i.e. max alignment of all)
			using node_alignment = align_l<node_allocs>;

			//Prepare to recursively execute the search with the same Segment
			using sub_impl = boost::mp11::mp_bind_back<subtree_alignment, Segment>;

			//Get list of max alignment requirement from sub-nodes.
			using sub_alignments = boost::mp11::mp_transform_q<sub_impl, typename Node::sub_nodes>;

			//Push to the front of that list this node alignment requirement (it is at least 1).
			using all_alignments = boost::mp11::mp_push_front<sub_alignments, node_alignment>;

			//Select maximum alignment requirement
			using type = boost::mp11::mp_max_element<all_alignments, boost::mp11::mp_less>;
		};
	}
}}