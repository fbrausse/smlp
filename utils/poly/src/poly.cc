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
	sptr<form2> partial_domain = true2;
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
	fprintf(f, "(get-model)\n");
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

struct Match {

	/* Match could be a partial function (if None was on of the arguments).
	 * This (conjunction) of constraints are required to be satisfied in
	 * order for Match() to produce a value. */
	vec<sptr<form2>> constraints;

	sptr<expr2> operator()(vec<sptr<expr2>> args)
	{
		assert(args.size() >= 2);
		const sptr<expr2> &var = args.front();
		sptr<expr2> r = move(args.back());
		int k = args.size()-3;
		if (!r) {
			vec<sptr<form2>> disj;
			for (int i=k; i >= 1; i-=2)
				disj.emplace_back(make2f(prop2 { EQ, var, args[i] }));
			constraints.emplace_back(make2f(lbop2 { lbop2::OR, move(disj) }));
			r = move(args[k+1]);
			k -= 2;
		}
		for (int i=k; i >= 1; i-=2) {
			assert(args[i]);
			assert(args[i+1]);
			r = make2e(ite2 {
				make2f(prop2 { EQ, var, move(args[i]) }),
				move(args[i+1]),
				move(r),
			});
		}
		return r;
	}
};

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
	Match match;
	sptr<expr2> e2 = unroll(e, { {"Match", std::ref(match)} });

	return pre_problem {
		move(d),
		move(e2),
		make2f(lbop2 { lbop2::AND, move(match.constraints) })
	};
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
		return to_Q(cnst_fold(obj, point)->get<cnst2>()->value);
	}
};

static vec<smlp_result>
optimize_EA(cmp_t direction,
            const domain &dom,
            const sptr<expr2> &objective,
            const sptr<form2> &alpha,
            const sptr<form2> &beta,
            ival &obj_range,
            const kay::Q &max_prec,
            const fun<sptr<form2>(bool left, const hmap<str,sptr<expr2>> &)> theta,
            const char *logic = nullptr)
{
	assert(is_order(direction));

	/* optimize T in obj_range such that (assuming direction is >=):
	 *
	 * E x . eta x /\
	 * A y . eta y -> theta x y -> alpha y -> (beta y /\ obj y >= T)
	 *
	 * In this implementation, eta is represented as the domain constraints
	 * from 'dom'.
	 */

	vec<smlp_result> results, counter_examples;

	while (length(obj_range) > max_prec) {
		kay::Q T = mid(obj_range);
		sptr<expr2> threshold = make2e(cnst2 { T });

		/* eta x /\ alpha x /\ beta x /\ obj x >= T */
		z3_solver exists(dom, logic);
		exists.add(*alpha);
		exists.add(*beta);
		exists.add(prop2 { direction, objective, threshold });

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
			forall.add(*theta(true, candidate));
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
			auto &counter_example = a.get<sat>()->model;
			counter_examples.emplace_back(T, counter_example);
			exists.add(lneg2 { theta(false, counter_example) });
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
  -t TIMEOUT  set the solver timeout in seconds, 0 to disable [0]\n\
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

static void alarm_handler(int sig)
{
	if (z3::context *p = z3_solver::is_checking)
		p->interrupt();
	signal(sig, alarm_handler);
}

static void print_model(FILE *f, const hmap<str,sptr<expr2>> &model, int indent)
{
	size_t k = 0;
	for (const auto &[n,_] : model)
		k = max(k, n.length());
	for (const auto &[n,c] : model) {
		kay::Q q = to_Q(c->get<cnst2>()->value);
		fprintf(f, "%*s%*s = %s\n", indent, "", -(int)k, n.c_str(),
		        q.get_str().c_str());
	}
}

int main(int argc, char **argv)
{
	/* these determine the mode of operation of this program */
	bool solve         = true;
	bool dump_pe       = false;
	bool dump_smt2     = false;
	bool infix         = true;
	bool python_compat = false;
	int  timeout       = 0;

	/* parse options from the command-line */
	for (int opt; (opt = getopt(argc, argv, ":C:F:hnpst:")) != -1;)
		switch (opt) {
		case 'C':
			if (optarg == "python"sv)
				python_compat = true;
			else
				DIE(1,"error: option '-C' only supports "
				      "'python'\n");
			break;
		case 'F':
			if (optarg == "infix"sv)
				infix = true;
			else if (optarg == "prefix"sv)
				infix = false;
			else
				DIE(1,"error: option '-F' only supports "
				      "'infix' and 'prefix'\n");
			break;
		case 'h': usage(argv[0], 0);
		case 'n': solve = false; break;
		case 'p': dump_pe = true; break;
		case 's': dump_smt2 = true; break;
		case 't': timeout = atoi(optarg); break;
		case ':': DIE(1,"error: option '-%c' requires an argument\n",
		              optopt);
		case '?': DIE(1,"error: unknown option '-%c'\n",optopt);
		}
	if (argc - optind != 4)
		usage(argv[0], 1);

	auto [dom,lhs,pc] = parse_poly_problem(argv[optind], argv[optind+1],
	                                       python_compat, dump_pe, infix);

	/* hint for the solver later: non-linear real arithmetic, potentially
	 * also with integers */
	const char *logic = "QF_NRA";
	for (const auto &[_,rng] : dom)
		if (!is_real(rng))
			logic = "QF_NIRA";

	/* Check that the constraints from partial function evaluation are met
	 * on the domain. */
	z3_solver ood(dom, logic);
	ood.add(lneg2 { pc });
	ood.check().match(
	[](const sat &s) {
		fprintf(stderr, "error: DOMAIN constraints do not imply that "
		                "all function parameters are inside the "
		                "respective function's domain, e.g.:\n");
		print_model(stderr, s.model, 2);
		exit(4);
	},
	[](const auto &) {}
	);

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

	if (timeout > 0) {
		signal(SIGALRM, alarm_handler);
		alarm(timeout);
	}

	/* optionally solve the problem */
	if (solve)
		solve_exists(p.dom, p.p, logic).match(
		[&](const sat &s) {
			kay::Q q = to_Q(cnst_fold(lhs, s.model)->get<cnst2>()->value);
			fprintf(stderr, "sat, lhs value: %s ~ %g, model:\n",
			        q.get_str().c_str(), q.get_d());
			print_model(stderr, s.model, 2);
			for (const auto &[n,c] : s.model) {
				kay::Q q = to_Q(c->get<cnst2>()->value);
				assert(p.dom[n]->contains(q));
			}
		},
		[](const unsat &) { fprintf(stderr, "unsat\n"); },
		[](const unknown &u) {
			fprintf(stderr, "unknown: %s\n", u.reason.c_str());
		}
		);
}
