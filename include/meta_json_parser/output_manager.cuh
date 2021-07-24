#pragma once
#include <type_traits>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <boost/mp11/map.hpp>
#include <boost/mp11/bind.hpp>
#include <boost/mp11/list.hpp>
#include <meta_json_parser/meta_math.h>
#include <meta_json_parser/meta_utils.h>
#include <meta_json_parser/memory_request.h>
#include <meta_json_parser/action_iterator.h>
#include <meta_json_parser/kernel_launch_configuration.cuh>

template<class BeforeT>
struct OutputOptElementsBefore
{
	using Value = BeforeT;
	static constexpr auto value = Value::value;
};

template<int BeforeI>
using OutputOptElementsBefore_c = OutputOptElementsBefore<boost::mp11::mp_int<BeforeI>>;

template<class AfterT>
struct OutputOptElementsAfter
{
	using Value = AfterT;
	static constexpr auto value = Value::value;
};

template<int AfterI>
using OutputOptElementsAfter_c = OutputOptElementsAfter<boost::mp11::mp_int<AfterI>>;

template<class T>
using GetValue = typename T::Value;

template<class OutputTagT, class OutputTypeT, class OptionsT = boost::mp11::mp_list<>>
struct OutputRequest
{
	using OutputTag = OutputTagT;
	using OutputType = OutputTypeT;
	using Options = OptionsT;
};

template<class T, typename = int>
struct HaveOutputRequests : std::false_type {};

template<class T>
struct HaveOutputRequests<T, decltype(std::declval<typename T::OutputRequests>(), 0)> : std::true_type {};

template<class T>
using GetOutputRequests = typename T::OutputRequests;

template<class T>
using TryGetOutputRequests = boost::mp11::mp_eval_if_not<
	HaveOutputRequests<T>,
	boost::mp11::mp_list<>,
	GetOutputRequests,
	T
>;

using GetOutputRequests_q = boost::mp11::mp_quote<GetOutputRequests>;

/// <summary>
/// Output request that have dynamic size. E.g. strings in concatenated form.
/// </summary>
/// <typeparam name="OutputTagT"></typeparam>
/// <typeparam name="OutputTypeT"></typeparam>
template<class OutputTagT, class OutputTypeT, class OptionsT = boost::mp11::mp_list<>>
struct DynamicOutputRequest
{
	using OutputTag = OutputTagT;
	using OutputType = OutputTypeT;
	using Options = OptionsT;
};

using DynamicOutputIndex = uint32_t;

template<class RequestListT>
struct DynamicSizesFiller
{
	using RequestList = RequestListT;
	using DynamicOnlyRequestList = boost::mp11::mp_copy_if_q<
		RequestList,
		IsTemplate<DynamicOutputRequest>
	>;

	using DynamicRequestsCount = boost::mp11::mp_size<DynamicOnlyRequestList>;

	using RequestSize = boost::mp11::mp_int<sizeof(uint32_t) * DynamicRequestsCount::value>;

	void static Fill(StaticBuffer<RequestSize>& buff, KernelLaunchConfiguration* launch_config)
	{
		if (DynamicRequestsCount::value == 0)
			return;
		if (launch_config->dynamic_sizes.size() != DynamicRequestsCount::value)
		{
			throw std::runtime_error("Wrong number of KernelLaunchConfiguration::dynamic_sizes");
		}
		std::copy_n(
			launch_config->dynamic_sizes.begin(),
			DynamicRequestsCount::value,
			reinterpret_cast<uint32_t*>(&buff)
		);
	}
};

template<class BaseActionT>
struct OutputConfiguration
{
	using BaseAction = BaseActionT;
	using RequestList = boost::mp11::mp_flatten<
		boost::mp11::mp_transform<
			TryGetOutputRequests,
			ActionIterator<BaseAction>
		>
	>;

	static_assert(boost::mp11::mp_is_map<RequestList>::value, "Requests need to have unique tags");

	using DynamicOnlyRequestList = boost::mp11::mp_copy_if_q<
		RequestList,
		IsTemplate<DynamicOutputRequest>
	>;

	using RequestSize = boost::mp11::mp_int<
		sizeof(uint32_t) * boost::mp11::mp_size<DynamicOnlyRequestList>::value
	>;

	using DynamicSizesMemoryRequest = FilledMemoryRequest<
		RequestSize,
		DynamicSizesFiller<RequestList>,
		MemoryUsage::ReadOnly,
		MemoryType::Shared
	>;

	using MemoryRequests = boost::mp11::mp_list<DynamicSizesMemoryRequest>;
};

template<class BaseActionT>
struct OutputManager
{
	using BaseAction = BaseActionT;
	using OC = OutputConfiguration<BaseAction>;
	using RequestList = typename OC::RequestList;
	using DynamicOnlyRequestList = typename OC::DynamicOnlyRequestList;

	void** mOutputs;
	uint32_t* mDynamicSizes;
	__device__ __forceinline__ OutputManager(
		void** outputs, uint32_t* dynamicSizes
	) : mOutputs(outputs), mDynamicSizes(dynamicSizes) {}

	template<class TagT>
	//mp_second as OutputTypeT is second argument for OutputRequest
	using OutType = boost::mp11::mp_second<boost::mp11::mp_map_find<RequestList, TagT>>;

	template<class TagT>
	//mp_second as OutputTypeT is second argument for OutputRequest
	using Request = boost::mp11::mp_map_find<RequestList, TagT>;

	template<class TagT>
	using TagIndex = boost::mp11::mp_find_if_q<
		boost::mp11::mp_transform<
			boost::mp11::mp_first,
			RequestList
		>,
		boost::mp11::mp_bind_q<
			boost::mp11::mp_quote<boost::mp11::mp_same>,
			boost::mp11::_1,
			TagT
		>
	>;

	template<class TagT>
	using DynamicTagIndex = boost::mp11::mp_find_if_q<
		boost::mp11::mp_transform<
			boost::mp11::mp_first,
			DynamicOnlyRequestList
		>,
		boost::mp11::mp_bind_q<
			boost::mp11::mp_quote<boost::mp11::mp_same>,
			boost::mp11::_1,
			TagT
		>
	>;

	template<class KC, class TagT>
	__device__ __forceinline__
	typename std::enable_if<
		IsTemplate<DynamicOutputRequest>::template fn<Request<TagT>>::value,
		uint32_t
	>::type DynamicSize()
	{
		using Index = DynamicTagIndex<TagT>;
		return mDynamicSizes[Index::value];
	}

	template<class KC, class TagT>
	__device__ __forceinline__
	typename std::enable_if<
		!IsTemplate<DynamicOutputRequest>::template fn<Request<TagT>>::value,
		OutType<TagT>*
	>::type Pointer()
	{
		using Index = TagIndex<TagT>;
		static_assert(Index::value < boost::mp11::mp_size<RequestList>::value, "Requested tag is not present in OutputRequests");
		return reinterpret_cast<OutType<TagT>*>(mOutputs[Index::value]) + KC::RT::InputId();
	}

	template<class KC, class TagT>
	__device__ __forceinline__
	typename std::enable_if<
		IsTemplate<DynamicOutputRequest>::template fn<Request<TagT>>::value,
		OutType<TagT>*
	>::type Pointer()
	{
		using Index = TagIndex<TagT>;
		static_assert(Index::value < boost::mp11::mp_size<RequestList>::value, "Requested tag is not present in OutputRequests");
		return reinterpret_cast<OutType<TagT>*>(mOutputs[Index::value]) + KC::RT::InputId() * DynamicSize<KC, TagT>();
	}

	template<class KC, class TagT>
	__device__ __forceinline__ OutType<TagT>& Get()
	{
		return *(this->Pointer<KC, TagT>());
	}

	template<class TagT, template<class...> class OptionT>
	__host__ __device__ __forceinline__
	constexpr static auto GetOptionValue()
	{
		using Index = TagIndex<TagT>;
		using Request = boost::mp11::mp_at<RequestList, Index>;
		using Options = typename Request::Options;
		using Found = boost::mp11::mp_find_if_q<
			Options,
			IsTemplate<OptionT>
		>;

		return boost::mp11::mp_eval_if_q<
			boost::mp11::mp_bool<
				Found::value == boost::mp11::mp_size<Options>::value
			>,
			boost::mp11::mp_int<0>,
			boost::mp11::mp_compose_q<
				boost::mp11::mp_bind<
					boost::mp11::mp_at,
					Options,
					boost::mp11::_1
				>,
				boost::mp11::mp_quote<GetValue>
			>,
			Found
		>::value;

	}

	template<class TagT>
	__host__ __device__ __forceinline__
	constexpr static size_t ElementsBefore()
	{
		return GetOptionValue<TagT, OutputOptElementsBefore>();
	}

	template<class TagT>
	__host__ __device__ __forceinline__
	constexpr static size_t ElementsAfter()
	{
		return GetOptionValue<TagT, OutputOptElementsAfter>();
	}

	template<class TagT>
	__host__ __device__ __forceinline__
	static size_t ToAlloc(const KernelLaunchConfiguration* launch_config, const size_t size)
	{
		using Index = TagIndex<TagT>;
		using Request = boost::mp11::mp_at<RequestList, Index>;
		using DynamicIndex = boost::mp11::mp_find<DynamicOnlyRequestList, Request>;
		size_t to_alloc = size;
		//to_alloc += ElementsBefore<TagT>();
		//to_alloc += ElementsAfter<TagT>();
		if (IsTemplate<DynamicOutputRequest>::template fn<Request>::value)
		{
			to_alloc *= launch_config->dynamic_sizes[DynamicIndex::value];
		}
		else
		{
			to_alloc *= sizeof(typename Request::OutputType);
		}
		return to_alloc;
	}
};
