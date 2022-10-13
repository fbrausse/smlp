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
		return parse_domain(f);
	DIE(1,"error opening domain file path: %s: %s\n",path,strerror(errno));
}

static expr parse_expression_file(const char *path, bool infix, bool python_compat)
{
	if (file f { path, "r" })
		return infix ? parse_infix(f, python_compat) : parse_pe(f);
	DIE(1,"error opening expression file path: %s: %s\n",path,strerror(errno));
}

static expr2 Match(vec<expr2> args)
{
	assert(args.size() >= 2);
	const name *var = args.front().get<name>();
	expr2 r = move(args.back());
	for (int i=args.size()-3; i >= 1; i-=2)
		r = ite2 {
			make2f(prop2 {
				EQ,
				make2e(*var),
				make2e(move(args[i]))
			}),
			make2e(move(args[i+1])),
			make2e(move(r)),
		};
	return r;
}

[[noreturn]]
static void usage(const char *program_name, int exit_code)
{
	FILE *f = exit_code ? stderr : stdout;
	fprintf(f, "usage: %s [-OPTS] [--] DOMAIN-FILE EXPR-FILE OP CNST\n",
	        program_name);
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
The DOMAIN-FILE is a text file containing the bounds for all variables in the\n\
form 'NAME -- RANGE' where NAME is the name of the variable and RANGE is either\n\
an interval of the form '[a,b]' or a list of specific values '{a,b,c,d,...}'.\n\
Empty lines are skipped.\n\
\n\
The EXPR-FILE contains a polynomial expression in the variables specified by the\n\
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

	/* parse the input */
	domain d = parse_domain_file(argv[optind]);
	expr e = parse_expression_file(argv[optind+1], infix, python_compat);

	/* optionally dump the prefix notation of the expression */
	if (dump_pe)
		::dump_pe(stdout, e);

	/* interpret symbols of known non-recursive functions and numeric
	 * constants */
	hmap<str,fun<expr2(vec<expr2>)>> funs;
	funs["Match"] = Match;
	expr2 e2 = unroll(e, funs);

	/* find out about the OP comparison operation */
	size_t c;
	for (c=0; c<ARRAY_SIZE(cmp_s); c++)
		if (std::string_view(cmp_s[c]) == argv[optind+2])
			break;
	if (c == ARRAY_SIZE(cmp_s))
		DIE(1,"OP '%s' unknown\n",argv[optind+2]);

	/* interpret the CNST on the right hand side */
	expr2 rhs = unroll(cnst { argv[optind+3] }, funs);

	/* the problem consists of domain and the (EXPR OP CNST) constraint */
	problem p = {
		move(d),
		prop2 {
			(cmp_t)c,
			make2e(move(e2)),
			make2e(move(rhs)),
		},
	};

	/* hint for the solver later: non-linear real arithmetic, potentially
	 * also with integers */
	const char *logic = "QF_NRA";
	for (const auto &[_,rng] : p.dom)
		if (!is_real(rng))
			logic = "QF_NIRA";

	/* optionally dump the smt2 representation of the (unrolled) expression
	 */
	if (dump_smt2)
		::dump_smt2(stdout, logic, p);

	/* optionally solve the problem */
	if (solve) {
		z3_solver s(p.dom);
		s.slv.set("logic", logic);
		s.add(p.p);
		z3::check_result r = s.slv.check();
		switch (r) {
		case z3::sat: fprintf(stderr, "sat\n"); break;
		case z3::unsat: fprintf(stderr, "unsat\n"); break;
		case z3::unknown: fprintf(stderr, "unknown\n"); break;
		}
	}
}
