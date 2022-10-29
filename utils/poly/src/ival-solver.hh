
#pragma once

#include "solver.hh"

namespace smlp {

struct ival_solver : solver {

	ival_solver() = default;

	void declare(const domain &d) override
	{
		assert(empty(dom));
		dom = d;
	}

	void add(const form2 &f) override
	{
		asserts.push_back(f);
	}

	result check() override;

private:
	domain dom;
	vec<form2> asserts;
};

}
