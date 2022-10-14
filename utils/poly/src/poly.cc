/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "common.hh"
#include "expr.hh"
#include "prefix.hh"
#include "infix.hh"
#include "expr2.hh"
#include "domain.hh"
#include "dump-smt2.hh"
#include "z3-solver.hh"

using namespace smlp;

namespace {

// using domain_ref = typename domain::const_iterator;

struct pre_problem {
	domain dom;
	sptr<expr2> func;
};

struct problem {

	domain dom;
	form2 p;
};

struct file {

	FILE *f;

	file(const char *path, const char *mode)
	: f(fopen(path, mode))
	{}

	~file() { if (f) fclose(f); }

	file(const file &) = delete;
	file & operator=(const file &) = delete;

	operator FILE *() const { return f; }
};

}

static void dump_smt2(FILE *f, const char *logic, const problem &p)
{
	fprintf(f, "(set-logic %s)\n", logic);
	dump_smt2(f, p.dom);
	fprintf(f, "(assert ");
	dump_smt2(f, p.p);
	fprintf(f, ")\n");
	fprintf(f, "(check-sat)\n");
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

static sptr<expr2> Match(vec<sptr<expr2>> args)
{
	assert(args.size() >= 2);
	const sptr<expr2> &var = args.front();
	sptr<expr2> r = move(args.back());
	for (int i=args.size()-3; i >= 1; i-=2)
		r = make2e(ite2 {
			make2f(prop2 { EQ, var, move(args[i]) }),
			move(args[i+1]),
			move(r),
		});
	return r;
}

static pre_problem parse_poly_problem(const char *simple_domain_path,
                                      const char *poly_expression_path,
                                      bool python_compat,
                                      bool dump_pe = false,
                                      bool infix = true)
{
	/* parse the input */
	domain d = parse_domain_file(simple_domain_path);
	expr e = parse_expression_file(poly_expression_path, infix, python_compat);

	/* optionally dump the prefix notation of the expression */
	if (dump_pe)
		::dump_pe(stdout, e);

	/* interpret symbols of known non-recursive functions and numeric
	 * constants */
	sptr<expr2> e2 = unroll(e, { {"Match", Match} });

	return pre_problem { move(d), move(e2) };
}

static result solve_exists(const domain &dom,
                           const form2 &f,
                           const char *logic = nullptr)
{
	z3_solver s(dom, logic);
	s.add(f);
	return s.check();
}

struct smlp_result {
	kay::Q threshold;
	hmap<str,sptr<expr2>> point;

	kay::Q center_value(const sptr<expr2> &obj) const
	{
		return to_Q(cnst_fold(subst(obj, point))->get<cnst2>()->value);
	}
};

/* assumes the relation underlying theta is symmetric! */
static vec<smlp_result>
optimize_EA(cmp_t direction,
            const domain &dom,
            const sptr<expr2> &objective,
            const sptr<form2> &alpha,
            const sptr<form2> &beta,
            ival &obj_range,
            const kay::Q &max_prec,
            const fun<sptr<form2>(const hmap<str,sptr<expr2>> &)> theta,
            const char *logic = nullptr)
{
	assert(is_order(direction));
	vec<smlp_result> results, counter_examples;

	while (length(obj_range) > max_prec) {
		kay::Q T = mid(obj_range);
		sptr<expr2> threshold = make2e(cnst2 { T });

		z3_solver exists(dom, logic);
		exists.add(prop2 { direction, objective, threshold });
		exists.add(*alpha);
		exists.add(*beta);

		while (true) {
			result e = exists.check();
			if (unknown *u = e.get<unknown>())
				DIE(2,"exists is unknown: %s\n", u->reason.c_str());
			if (e.get<unsat>()) {
				if (is_less(direction))
					obj_range.lo = T;
				else
					obj_range.hi = T;
				break;
			}
			auto &candidate = e.get<sat>()->model;

			z3_solver forall(dom, logic);
			/* ! ( eta y -> theta x y -> alpha y -> beta y /\ obj y >= T ) =
			 * ! ( ! eta y \/ ! theta x y \/ ! alpha y \/ beta y /\ obj y >= T ) =
			 * eta y /\ theta x y /\ alpha y /\ ( ! beta y \/ obj y < T) */
			forall.add(*theta(candidate));
			forall.add(*alpha);
			forall.add(lbop2 { lbop2::OR, {
				make2f(lneg2 { beta }),
				make2f(prop2 { ~direction, objective, threshold })
			} });

			result a = forall.check();
			if (unknown *u = a.get<unknown>())
				DIE(2,"forall is unknown: %s\n", u->reason.c_str());
			if (a.get<unsat>()) {
				results.emplace_back(T, candidate);
				if (is_less(direction))
					obj_range.hi = T;
				else
					obj_range.lo = T;
				break;
			}
			exists.add(lneg2 { theta(a.get<sat>()->model) });
		}
	}

	return results;
}

[[noreturn]]
static void usage(const char *program_name, int exit_code)
{
	FILE *f = exit_code ? stderr : stdout;
	fprintf(f, "\
usage: %s [-OPTS] [--] DOMAIN EXPR OP CNST\n\
", program_name);
	if (!exit_code)
		fprintf(f,"\
\n\
Options [defaults]:\n\
  -C COMPAT   use a compatibility layer, can be given multiple times; supported\n\
              values for COMPAT:\n\
              - python: reinterpret floating point constants as python would\n\
                        print them\n\
  -F IFORMAT  determines the format of the EXPR-FILE; can be one of: 'infix',\n\
              'prefix' [infix]\n\
  -h          displays this help message\n\
  -n          dry run, do not solve the problem [no]\n\
  -p          dump the expression in Polish notation to stdout [no]\n\
  -s          dump the problem in SMT-LIB2 format to stdout [no]\n\
\n\
The DOMAIN is a text file containing the bounds for all variables in the\n\
form 'NAME -- RANGE' where NAME is the name of the variable and RANGE is either\n\
an interval of the form '[a,b]' or a list of specific values '{a,b,c,d,...}'.\n\
Empty lines are skipped.\n\
\n\
The EXPR file contains a polynomial expression in the variables specified by the\n\
DOMAIN-FILE. The format is either an infix notation or the prefix notation also\n\
known as Polish notation. The expected format can be specified through the -F\n\
switch.\n\
\n\
The problem to be solved is specified by the two parameters OP CNST where OP is\n\
one of '<=', '<', '>=', '>', '==' and '!='. Remember quoting the OP on the shell\n\
to avoid unwanted redirections. CNST is a rational constant in the same format\n\
as those in the EXPR-FILE (if any).\n\
\n\
Developed by Franz Brausse <franz.brausse@manchester.ac.uk>.\n\
License: Apache 2.0; part of SMLP.\n\
");
	exit(exit_code);
}

int main(int argc, char **argv)
{
	/* these determine the mode of operation of this program */
	bool solve         = true;
	bool dump_pe       = false;
	bool dump_smt2     = false;
	bool infix         = true;
	bool python_compat = false;

	/* parse options from the command-line */
	for (int opt; (opt = getopt(argc, argv, ":C:F:hnps")) != -1;)
		switch (opt) {
		case 'C':
			if (optarg == "python"sv)
				python_compat = true;
			else
				DIE(1,"\
error: option '-C' only supports 'python'\n");
			break;
		case 'F':
			if (!strcmp(optarg, "infix"))
				infix = true;
			else if (!strcmp(optarg, "prefix"))
				infix = false;
			else
				DIE(1,"\
error: option '-F' only supports 'infix' and 'prefix'\n");
			break;
		case 'h': usage(argv[0], 0);
		case 'n': solve = false; break;
		case 'p': dump_pe = true; break;
		case 's': dump_smt2 = true; break;
		case ':': DIE(1,"error: option '-%c' requires an argument\n",
		              optopt);
		case '?': DIE(1,"error: unknown option '-%c'\n",optopt);
		}
	if (argc - optind != 4)
		usage(argv[0], 1);

	auto [dom,lhs] = parse_poly_problem(argv[optind], argv[optind+1],
	                                    python_compat, dump_pe, infix);

	/* hint for the solver later: non-linear real arithmetic, potentially
	 * also with integers */
	const char *logic = "QF_NRA";
	for (const auto &[_,rng] : dom)
		if (!is_real(rng))
			logic = "QF_NIRA";

	/* find out about the OP comparison operation */
	size_t c;
	for (c=0; c<ARRAY_SIZE(cmp_s); c++)
		if (std::string_view(cmp_s[c]) == argv[optind+2])
			break;
	if (c == ARRAY_SIZE(cmp_s))
		DIE(1,"OP '%s' unknown\n",argv[optind+2]);

	/* interpret the CNST on the right hand side */
	sptr<expr2> rhs = unroll(cnst { argv[optind+3] }, {});

	/* the problem consists of domain and the (EXPR OP CNST) constraint */
	problem p = {
		move(dom),
		prop2 { (cmp_t)c, lhs, rhs, },
	};

	/* optionally dump the smt2 representation of the problem */
	if (dump_smt2)
		::dump_smt2(stdout, logic, p);

	/* optionally solve the problem */
	if (solve)
		solve_exists(p.dom, p.p, logic).match(
		[&](const sat &s) {
			kay::Q q = to_Q(cnst_fold(subst(lhs, s.model))->get<cnst2>()->value);
			fprintf(stderr, "sat, value: %s ~ %g, model:\n", q.get_str().c_str(), q.get_d());
			for (const auto &[n,c] : s.model) {
				kay::Q q = to_Q(c->get<cnst2>()->value);
				fprintf(stderr, "  %s = %s\n", n.c_str(),
				        q.get_str().c_str());
				assert(p.dom[n]->contains(q));
			}
		},
		[](const unsat &) { fprintf(stderr, "unsat\n"); },
		[](const unknown &u) { fprintf(stderr, "unknown: %s\n", u.reason.c_str()); }
		);
}
