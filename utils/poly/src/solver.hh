/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "domain.hh"

namespace smlp {

struct sat { hmap<str,sptr<term2>> model; };
struct unsat {};
struct unknown { str reason; };

typedef sumtype<sat,unsat,unknown> result;

struct solver {

	virtual ~solver() = default;
	virtual void declare(const domain &d) = 0;
	virtual void add(const form2 &f) = 0;
	virtual result check() = 0;
};

struct solver_seq : solver {

	const vec<uptr<solver>> solvers;

	explicit solver_seq(vec<uptr<solver>> solvers)
	: solvers(move(solvers))
	{ assert(!empty(this->solvers)); }

	void declare(const domain &d) override
	{
		for (const uptr<solver> &s : solvers)
			s->declare(d);
	}

	void add(const form2 &f) override
	{
		for (const uptr<solver> &s : solvers)
			s->add(f);
	}

	result check() override
	{
		result r;
		for (const uptr<solver> &s : solvers)
			if (!(r = s->check()).get<unknown>())
				return r;
		return r;
	}
};

struct interruptible {

	virtual ~interruptible() = default;
	virtual void interrupt() = 0;

	static interruptible *is_active;
};

}
