/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "solver.hh"

#include <z3++.h>

namespace smlp {

struct z3_solver : solver, interruptible {

	explicit z3_solver(const char *logic = nullptr);
	void declare(const domain &d) override;
	result check() override;

	void interrupt() override { ctx.interrupt(); }

	void add(const sptr<form2> &f) override
	{
		hmap<void *, z3::expr> m;
		slv.add(interp(f, m));
	}

private:
	z3::context ctx;
	z3::solver slv;
	hmap<str,z3::expr> symbols;

	z3::expr interp(const sptr<term2> &e, hmap<void *, z3::expr> &m);
	z3::expr interp(const sptr<form2> &f, hmap<void *, z3::expr> &m);
	z3::expr interp(const term2 &e, hmap<void *, z3::expr> &m);
	z3::expr interp(const form2 &f, hmap<void *, z3::expr> &m);
};

}
