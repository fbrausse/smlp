/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "domain.hh"

#include <z3++.h>

namespace smlp {

struct sat { hmap<str,sptr<expr2>> model; };
struct unsat {};
struct unknown { str reason; };

typedef sumtype<sat,unsat,unknown> result;

class z3_solver {

	z3::context ctx;
	z3::solver slv;
	hmap<str,z3::expr> symbols;

	z3::expr interp(const sptr<expr2> &e, hmap<void *, z3::expr> &m);
	z3::expr interp(const sptr<form2> &f, hmap<void *, z3::expr> &m);
	z3::expr interp(const expr2 &e, hmap<void *, z3::expr> &m);
	z3::expr interp(const form2 &f, hmap<void *, z3::expr> &m);
public:
	explicit z3_solver(const domain &d, const char *logic = nullptr);

	result check();

	static z3::context *is_checking;

	void add(const form2 &f)
	{
		hmap<void *, z3::expr> m;
		slv.add(interp(f, m));
	}
};

}
