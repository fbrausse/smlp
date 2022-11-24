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
	virtual void add(const sptr<form2> &f) = 0;
	virtual result check() = 0;

	friend vec<hmap<str,sptr<term2>>> all_solutions(solver &s)
	{
		vec<hmap<str,sptr<term2>>> v;
		sat *c;
		for (result r; (c = (r = s.check()).get<sat>());) {
			vec<sptr<form2>> ne;
			v.emplace_back(move(c->model));
			for (const auto &[var,t] : v.back())
				ne.push_back(make2f(prop2 { NE, make2t(name { var }), t }));
			s.add(disj(move(ne)));
		}
		return v;
	}
};

struct acc_solver : solver {

	void declare(const domain &d) override { assert(empty(dom)); dom = d; }
	void add(const sptr<form2> &f) override { asserts.args.push_back(f); }
	result check() override { return static_cast<const acc_solver *>(this)->check(); }
	virtual result check() const = 0;

protected:
	domain dom;
	lbop2 asserts = { lbop2::AND, {} };
};

uptr<solver> mk_solver0(bool incremental, const char *logic);
vec<hmap<str,sptr<term2>>> all_solutions(const domain &dom, const sptr<form2> &f);

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

	void add(const sptr<form2> &f) override
	{
		for (const uptr<solver> &s : solvers)
			s->add(f);
	}

	result check() override
	{
		result r = unknown { "solver sequence is empty" };
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
