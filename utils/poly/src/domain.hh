
#pragma once

#include "expr2.hh"

namespace smlp {

struct ival { kay::Q lo, hi; };
struct list { vec<kay::Q> values; };

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

form2 domain_constraint(const str &var, const component &rng);

struct domain : vec<pair<str,component>> {
};

domain parse_domain(FILE *f);

}
