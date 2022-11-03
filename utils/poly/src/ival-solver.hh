
#pragma once

#include "solver.hh"

namespace smlp {

struct ival_solver : solver {

	ival_solver(size_t max_subdivs = 0)
	: max_subdivs(max_subdivs)
	{}

	void declare(const domain &d) override
	{
		assert(empty(dom));
		dom = d;
	}

	void add(const sptr<form2> &f) override
	{
		conj.args.push_back(f);
	}

	result check() override;

private:
	domain dom;
	lbop2 conj { lbop2::AND, {} };
	size_t max_subdivs;
};

}
