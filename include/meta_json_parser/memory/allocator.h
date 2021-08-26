#pragma once

#include "config_transform.h"

namespace meta_json { namespace memory {

	namespace detail {

		//segment map contains: segment tag, segment init map, current segment scratch offset
		template<typename SegmentMap>
		struct allocator_state
		{
			using ptr_array = std::array<std::byte*, boost::mp11::mp_size<SegmentMap>::value>;

		private:
			//Create map of (segment,initMap,offset,index) elements
			using segment_map = boost::mp11::mp_transform<
				boost::mp11::mp_push_back,
				SegmentMap,
				boost::mp11::mp_iota<boost::mp11::mp_size<SegmentMap>>>;

			static_assert(boost::mp11::mp_is_map<segment_map>::value, "Segments must be unique");

			template<typename Segment>
			using index = boost::mp11::mp_back<boost::mp11::mp_map_find<segment_map, Segment>>;

			template<typename Segment>
			using offset = boost::mp11::mp_third<boost::mp11::mp_map_find<segment_map, Segment>>;

			template<typename Segment>
			using init_map = boost::mp11::mp_second<boost::mp11::mp_map_find<segment_map, Segment>>;

			template<typename Segment>
			using is_valid_segment = boost::mp11::mp_map_contains<segment_map, Segment>;

			const ptr_array base_ptrs;

			template<typename OtherMap,
				std::enable_if_t<
				std::is_same_v<
				boost::mp11::mp_map_keys<SegmentMap>,
				boost::mp11::mp_map_keys<OtherMap>>,
				int> = 0>
				allocator_state(allocator_state<OtherMap> const& other)
				: base_ptrs{ other.base_ptrs }
			{ }

			template<typename AdvMap, typename Segment>
			using find_offset_impl = boost::mp11::mp_map_find<AdvMap, Segment>;

			template<typename AdvPair>
			using value_or_zero = boost::mp11::mp_eval_if<
				std::is_same<AdvPair, void>,
				boost::mp11::mp_size_t<0>,
				boost::mp11::mp_second, AdvPair>;

			template<typename AdvMap, typename Segment>
			using find_offset = value_or_zero<find_offset_impl<AdvMap, Segment>>;

			template<typename AdvMap, typename SegmentEntry>
			using advance_segment = boost::mp11::mp_replace_third<
				boost::mp11::mp_pop_back<SegmentEntry>,//remove index (fourth element)
				boost::mp11::mp_plus< //and replace scratch offset (third element) with a sum of
				boost::mp11::mp_third<SegmentEntry>, //current scratch offset
				//and the value to advance by
				find_offset<AdvMap, boost::mp11::mp_first<SegmentEntry>>>>;

			template<typename OtherMap>
			friend struct allocator_state;

			template<typename Alloc>
			using is_valid_init_alloc = boost::mp11::mp_map_contains< //check if
				boost::mp11::mp_second< //map stored in the second entry
				boost::mp11::mp_map_find< //of the element in
				segment_map, //segment map
				typename Alloc::segment>>, //under given segment key
				Alloc>;// contains this allocation request.

		public:

			explicit allocator_state(ptr_array const& segment_ptrs)
				: base_ptrs{ segment_ptrs }
			{ }

			template<typename Segment, std::enable_if_t<is_valid_segment<Segment>::value, int> = 0>
			[[nodiscard]] std::byte* scratch_ptr() const
			{
				return std::get<index<Segment>::value>(base_ptrs) + offset<Segment>::value;
			}

			template<typename ...SegmentAdv>
			[[nodiscard]] auto advance() const
			{
				using namespace boost::mp11;
				using advance_map = mp_list<SegmentAdv...>;
				//for each element in segment map find adv or zero and add it to offset and drop index
				using adv_segm_q = mp_bind_front<advance_segment, advance_map>;
				using new_segment_map = mp_transform_q<adv_segm_q, segment_map>; //drops index and advances each entry
				return allocator_state<new_segment_map>{ *this };
			}

			template<typename Alloc, std::enable_if_t<is_valid_init_alloc<Alloc>::value, int> = 0>
			[[nodiscard]] const typename Alloc::type* allocate() const
			{
				using namespace boost::mp11;
				using segment = typename Alloc::segment;
				using ptr_type = const typename Alloc::type*;
				using offset = mp_second<mp_map_find<init_map<segment>, Alloc>>;
				return reinterpret_cast<ptr_type>(std::get<index<segment>::value>(base_ptrs) + offset::value);
			}

		};
	}

	template<typename LAllocs, typename State>
	struct allocator
	{

	};
	
}}