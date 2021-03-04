#pragma once
#include <cuda_runtime_api.h>
#include <boost/mp11/map.hpp>
#include <boost/mp11/bind.hpp>
#include <meta_json_parser/meta_math.h>

template<class RequestListT>
struct OutputConfiguration
{
	using RequestList = RequestListT;
	static_assert(boost::mp11::mp_is_map<RequestList>::value, "Requests need to have unique tags");
};

template<class OutputConfigurationT, class RequestT>
using AppendOutputRequest = OutputConfiguration<boost::mp11::mp_push_back<OutputConfigurationT::RequestList, RequestT>>;

template<class OutputConfigurationT, class RequestsT>
using AppendOutputRequests = OutputConfiguration<boost::mp11::mp_append<OutputConfigurationT::RequestList, RequestsT>>;

template<class OutputTagT, class OutputTypeT>
struct OutputRequest
{
	using OutputTag = OutputTagT;
	using OutputType = OutputTypeT;
};

template<class OutputConfigurationT>
struct OutputManager
{
	using RequestList = OutputConfigurationT::RequestList;

	void** mOutputs;
	__device__ __forceinline__ OutputManager(void** outputs) : mOutputs(outputs) {}

	template<class TagT>
	//mp_second as OutputTypeT is second argument for OutputRequest
	using OutType = boost::mp11::mp_second<boost::mp11::mp_map_find<RequestList, TagT>>;

	template<class KC, class TagT>
	__device__ __forceinline__ OutType<TagT>* Pointer()
	{
		using Index = boost::mp11::mp_find_if_q<
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
		static_assert(Index::value < boost::mp11::mp_size<RequestList>::value, "Requested tag is not present in OutputRequests");
		return reinterpret_cast<OutType<TagT>*>(mOutputs[Index::value]) + KC::RT::InputId();
	}

	template<class KC, class TagT>
	__device__ __forceinline__ OutType<TagT>& Get()
	{
		return *(this->Pointer<KC, TagT>());
	}
};