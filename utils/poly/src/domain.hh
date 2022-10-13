/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "expr2.hh"

namespace smlp {

/* Closed interval with rational endpoints */
struct ival { kay::Q lo, hi; };

/* Explicit list of rational values */
struct list { vec<kay::Q> values; };

/* A component (of the domain) is either an interval or a list of rational
 * values */
struct component : sumtype<ival,list> {

	using sumtype<ival,list>::sumtype;

	friend bool is_real(const component &c)
	{
		if (c.get<ival>())
			return true;
		for (const kay::Q &q : c.get<list>()->values)
			if (q.get_den() != 1)
				return true;
		return 1 || false; /* always real: Z3 falls back to a slow method otherwise */
	}
};

/* Translates a component 'rng' and the appropriate variable name 'var' into a
 * constraint in form of a 'form2' formula. */
form2 domain_constraint(const str &var, const component &rng);

/* The domain is an (ordered) list of pairs (name, component) */
struct domain : vec<pair<str,component>> {
};

/* Parses the DOMAIN-FILE, see poly.cc for details. */
domain parse_simple_domain(FILE *f);

}
