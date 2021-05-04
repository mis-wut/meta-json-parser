#pragma once
#ifndef PARSER_INPUT_H
#define PARSER_INPUT_H
#include <cuda_runtime_api.h>
#include <meta_json_parser/config.h>
#include <cstdint>

struct ParserInput
{
private:
	/// <summary>
	/// Valid items. Jsons to load. Negative for unknown.
	/// </summary>
	int64_t m_size;
	/// <summary>
	/// Size of json buffer. Negative for unknown.
	/// </summary>
	int64_t m_capacity_json;
	/// <summary>
	/// Size of indices buffer. Negative for unknown.
	/// </summary>
	int64_t m_capacity_indices;

	/// <summary>
	/// Indices buffer.
	/// </summary>
	InputIndex* m_d_indices;

	/// <summary>
	/// Json buffer.
	/// </summary>
	char* m_d_json;

public:
	/// <summary>
	/// Valid items. Jsons to load. Negative for unknown.
	/// </summary>
	inline int64_t Size() const { return m_size; }
	/// <summary>
	/// Size of json buffer. Negative for unknown.
	/// </summary>
	inline int64_t CapacityJson() const { return m_capacity_json; }
	/// <summary>
	/// Size of indices buffer. Negative for unknown.
	/// </summary>
	inline int64_t CapacityIndices() const { return m_capacity_indices; }
	/// <summary>
	/// Indices buffer.
	/// </summary>
	inline const InputIndex* DeviceIndices() const { return m_d_indices; }
	/// <summary>
	/// Json buffer.
	/// </summary>
	inline const char* DeviceJson() const { return m_d_json; }

	/// <summary>
	/// Empty parser input. Need to be fill manually.
	/// </summary>
	/// <returns></returns>
	ParserInput() : m_size(-1), m_capacity_json(-1), m_capacity_indices(-1),
		m_d_indices(nullptr), m_d_json(nullptr)
	{
	}

	/// <summary>
	/// Empty parser input. Need to be fill manually.
	/// </summary>
	/// <returns></returns>
	ParserInput(int64_t json_capacity, int64_t indices_capacity) :
		m_size(0), m_capacity_json(json_capacity), m_capacity_indices(indices_capacity),
		m_d_indices(nullptr), m_d_json(nullptr)
	{
		cudaMalloc(&m_d_json, sizeof(char) * json_capacity);
		cudaMalloc(&m_d_indices, sizeof(InputIndex) * indices_capacity);
	}

	void WriteJsonsAsync(const char* h_jsons, int64_t size, cudaStream_t stream = 0)
	{
		if (size > m_capacity_json)
		{
			cudaFree(m_d_json);
			m_capacity_json = size;
			cudaMalloc(&m_d_json, sizeof(char) * m_capacity_json);
		}
		cudaMemcpyAsync(m_d_json, h_jsons, sizeof(char) * m_capacity_json, cudaMemcpyHostToDevice, stream);
	}

	void WriteIndicesAsync(const InputIndex* h_indices, int64_t size, cudaStream_t stream = 0)
	{
		if (size > m_capacity_indices)
		{
			cudaFree(m_d_indices);
			m_capacity_indices = size;
			cudaMalloc(&m_d_indices, sizeof(InputIndex) * m_capacity_indices);
		}
		cudaMemcpyAsync(m_d_indices, h_indices, sizeof(InputIndex) * m_capacity_indices, cudaMemcpyHostToDevice, stream);
	}
};
#endif
