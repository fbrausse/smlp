/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "domain.hh"

#include <z3++.h>

namespace smlp {

struct z3_solver {

	z3::context ctx;
	z3::solver slv;
	hmap<str,z3::expr> symbols;

	explicit z3_solver(const domain &d);

	void add(const form2 &f)
	{
		slv.add(interp(f));
	}

	z3::expr interp(const expr2 &e);
	z3::expr interp(const form2 &f);
};

}
