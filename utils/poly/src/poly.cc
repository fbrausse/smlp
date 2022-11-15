/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "poly.hh"
#include "prefix.hh"
#include "infix.hh"

using namespace smlp;

namespace {
struct Match {

	/* Match could be a partial function (if . was on of the arguments).
	 * This (conjunction) of constraints are required to be satisfied in
	 * order for Match() to produce a value. */
	vec<sptr<form2>> constraints;

	/* Match(expr, cnst1, expr1, cnst2, expr2, [...], .) */
	expr2s operator()(vec<expr2s> args)
	{
		assert(args.size() >= 2);
		assert(args.size() % 2 == 0);
		const sptr<term2> &var = *args.front().get<sptr<term2>>();
		sptr<term2> r = move(*args.back().get<sptr<term2>>());
		int k = args.size()-3;
		if (!r) {
			vec<sptr<form2>> d;
			for (int i=k; i >= 1; i-=2)
				d.emplace_back(make2f(prop2 { EQ, var, *args[i].get<sptr<term2>>() }));
			constraints.emplace_back(disj(move(d)));
			r = move(*args[k+1].get<sptr<term2>>());
			k -= 2;
		}
		assert(r);
		for (int i=k; i >= 1; i-=2) {
			sptr<term2> *rhs = args[i].get<sptr<term2>>();
			sptr<term2> *yes = args[i+1].get<sptr<term2>>();
			assert(rhs);
			assert(yes);
			r = make2t(ite2 {
				make2f(prop2 { EQ, var, move(*rhs) }),
				move(*yes),
				move(r),
			});
		}
		return r;
	}
};
}

static domain parse_domain_file(const char *path)
{
	if (file f { path, "r" })
		return parse_simple_domain(f);
	DIE(1,"error opening domain file path: %s: %s\n",path,strerror(errno));
}

static expr parse_expression_file(const char *path, bool infix, bool python_compat)
{
	if (file f { path, "r" })
		return infix ? parse_infix(f, python_compat) : parse_pe(f);
	DIE(1,"error opening expression file path: %s: %s\n",path,strerror(errno));
}

pre_problem smlp::parse_poly_problem(const char *simple_domain_path,
                                     const char *poly_expression_path,
                                     bool python_compat,
                                     bool dump_pe,
                                     bool infix)
{
	/* parse the input */
	domain d = parse_domain_file(simple_domain_path);
	expr e = parse_expression_file(poly_expression_path, infix, python_compat);

	/* optionally dump the prefix notation of the expression */
	if (dump_pe)
		::dump_pe(stdout, e);

	/* interpret symbols of known non-recursive functions and numeric
	 * constants */
	Match match;
	sptr<term2> e2 = *unroll(e, { {"Match", std::ref(match)} }).get<sptr<term2>>();

	return pre_problem {
		move(d),
		move(e2),
		{},
		{},
		true2,
		conj(move(match.constraints)),
		all_eq,
	};
}
