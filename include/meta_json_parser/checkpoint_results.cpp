//
// Created by Jakub Narębski <jnareb@mat.umk.pl> on 22.12.2021.
//
#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include <iomanip>

#include <cuda_runtime_api.h>

#include "checkpoint_results.h"

static std::vector<checkpoint_event_t> checkpoints;

void init_checkpoints(size_t reserve)
{
	checkpoints.reserve(reserve);
}

void checkpoint_event(cudaEvent_t event, cudaStream_t stream, std::string description)
{
	// TODO: uncomment the following line after the conversion to using this module
	//cudaEventRecord(event, stream);
	checkpoints.push_back(std::make_pair(event, description));
}

static bool is_subevent(checkpoint_event_t pair)
{
	return !pair.second.compare(0, 2, "- ");
}

// it is template only because iterator types are difficult;
// let compiler figure that one out instead.
template <typename IteratorT>
static IteratorT find_next_main_event(IteratorT &next, IteratorT &last)
{
	while (next++ != last) {
		if (!is_subevent(*next))
			return next;
	}

	return next;
}

void print_checkpoint_events()
{
	// TODO: make it automatic by finding the maximal length of descriptions
	//       and finding the length of decimal representation of total time
	const int c1 = 40-2;
	const int c2 = 10+1;

	float acc_ms;

	std::cout << "\nTotal accumulated time measured by GPU:\n";

	cudaEvent_t first_event = checkpoints[0].first;
	for (const auto &pair : checkpoints) {
		if (is_subevent(pair)) {
			std::cout << "--";
		} else {
			std::cout << "* ";
		}
		std::cout
			<< std::setw(c1) << std::left
			<< pair.second + ": ";

		cudaEventElapsedTime(&acc_ms, first_event, pair.first);
		std::cout
			<< std::setw(c2) << std::right << std::showpos
			<< static_cast<int64_t>(acc_ms * 1'000'000.0)
			<< " ns\n";
	}

	std::cout << "\n";
}

void print_checkpoint_results()
{
	// TODO: make it automatic by finding the maximal length of descriptions
	//       and finding the length of decimal representation of total time
	const int c1 = 40-2;
	const int c2 = 10;

	float dt_ms = -1.0f;

	std::cout << "\nTime measured by GPU:\n";

	auto curr = std::cbegin(checkpoints);
	auto last = std::cend(checkpoints);
	if (curr == last) return;

	auto next = curr;
	++next;
	while (next != last) {
		bool curr_is_subevent = is_subevent(*curr);
		bool next_is_subevent = is_subevent(*next);

		if (curr_is_subevent) {
			std::cout << "  ";
		} else {
			std::cout << "+ ";
		}
		std::cout << std::setw(c1) << std::left;
		if (!curr_is_subevent && next_is_subevent) {
			std::cout << (*curr).second + " (sum of the following): ";
			next = find_next_main_event(next, last);
		} else {
			std::cout << (*curr).second + ": ";
		}

		if (next != last) {
			cudaEventElapsedTime(&dt_ms, (*curr).first, (*next).first);
			std::cout
				<< std::setw(c2) << std::right << std::noshowpos
				<< static_cast<int64_t>(dt_ms * 1'000'000.0)
				<< " ns";
		}

		std::cout << "\n";

		next = ++curr;
		++next;
	}

	cudaEventElapsedTime(&dt_ms, checkpoints.front().first,
	                             checkpoints.back().first);
	std::cout
		<< "= "
		<< std::setw(c1) << std::left
		<< checkpoints.back().second + ": "
		<< std::setw(c2) << std::right
		<< static_cast<int64_t>(dt_ms * 1'000'000.0)
		<< " ns\n\n";
}
