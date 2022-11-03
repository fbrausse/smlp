/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "common.hh"
#include "expr.hh"
#include "infix.hh"
#include "expr2.hh"
#include "domain.hh"
#include "dump-smt2.hh"
#include "ext-solver.hh"
#include "ival-solver.hh"
#include "nn.hh"
#include "poly.hh"

#ifdef SMLP_ENABLE_Z3_API
# include "z3-solver.hh"
# include <z3_version.h>
#endif

#ifdef SMLP_ENABLE_KERAS_NN
# include <H5public.h>
# include <kjson.h>
#endif

#include <signal.h>
#include <time.h>

using namespace smlp;

namespace {

// using domain_ref = typename domain::const_iterator;

struct problem {

	domain dom;
	form2 p;
};

/* max T, s.t.
 * E x. eta x /\
 * A y. eta y -> theta x y -> alpha y -> beta y /\ obj y >= T
 */
struct maxprob {

	domain dom;
	term2  obj;
	sptr<form2> theta;
	sptr<form2> alpha = true2;
	sptr<form2> beta  = true2;
};

struct pareto {

	domain dom;
	vec<term2> objs;
	sptr<form2> alpha = true2;
	sptr<form2> beta  = true2;
};

struct timing : timespec {

	timing()
	{
		if (clock_gettime(CLOCK_MONOTONIC, this) == -1)
			throw std::error_code(errno, std::system_category());
	}

	friend timing & operator-=(timing &a, const timing &b)
	{
		a.tv_sec -= b.tv_sec;
		if ((a.tv_nsec -= b.tv_nsec) < 0) {
			a.tv_sec--;
			a.tv_nsec += 1e9;
		}
		return a;
	}

	friend timing operator-(timing a, const timing &b)
	{
		return a -= b;
	}

	operator double() const { return tv_sec + tv_nsec / 1e9; }
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

static const char *ext_solver_cmd;
static const char *inc_solver_cmd;
static long        intervals = -1;

uptr<solver> mk_solver0(bool incremental, const char *logic = nullptr);

uptr<solver> mk_solver0(bool incremental, const char *logic)
{
	const char *ext = ext_solver_cmd;
	const char *inc = inc_solver_cmd;
	const char *cmd = (inc && ext ? incremental : !ext) ? inc : ext;
	if (cmd)
		return std::make_unique<ext_solver>(cmd, logic);
#ifdef SMLP_ENABLE_Z3_API
	return std::make_unique<z3_solver>(logic);
#endif
	DIE(1,"error: no solver specified and none are built-in, require "
	      "external solver via -S or -I\n");
}

static uptr<solver> mk_solver(bool incremental, const char *logic = nullptr)
{
	if (intervals >= 0)
		return std::make_unique<ival_solver>(intervals, logic);
	return mk_solver0(incremental, logic);
}

template <typename T>
static str smt2_logic_str(const domain &dom, const T &e)
{
	bool reals = false;
	bool ints = false;
	for (const auto &[_,rng] : dom)
		switch (rng.type) {
		case component::INT: ints = true; break;
		case component::REAL: reals = true; break;
		}
	str logic = "QF_";
	if (ints || reals) {
		logic += is_nonlinear(e) ? 'N' : 'L';
		if (ints)
			logic += 'I';
		if (reals)
			logic += 'R';
		logic += 'A';
	} else
		logic += "UF";
	return logic;
}

static result solve_exists(const domain &dom,
                           const sptr<form2> &f,
                           const char *logic = nullptr)
{
	uptr<solver> s = mk_solver(false,
		logic ? logic : smt2_logic_str(dom, f).c_str());
	s->declare(dom);
	s->add(f);
	return s->check();
}

struct smlp_result {
	kay::Q threshold;
	hmap<str,sptr<term2>> point;

	smlp_result(kay::Q threshold, hmap<str,sptr<term2>> point)
	: threshold(move(threshold))
	, point(move(point))
	{}

	kay::Q center_value(const sptr<term2> &obj) const
	{
		return to_Q(cnst_fold(obj, point)->get<cnst2>()->value);
	}
};

static void trace_result(FILE *f, const char *lbl, const result &r,
                         const kay::Q &T, double t)
{
	const char *state = nullptr;
	vec<str> info;
	r.match(
	[&](const sat &s) {
		state = "sat";
		vec<const decltype(s.model)::value_type *> v;
		for (const auto &p : s.model)
			v.push_back(&p);
		sort(begin(v), end(v), [](const auto *a, const auto *b) {
			return a->first < b->first;
		});
		for (const auto *p : v) {
			info.emplace_back(p->first);
			info.emplace_back(to_string(p->second->get<cnst2>()->value));
		}
	},
	[&](const unsat &) { state = "unsat"; },
	[&](const unknown &u) {
		state = "unknown";
		info.emplace_back(u.reason);
	}
	);
	assert(state);
	fprintf(f, "%s,%s,%g,%5.3f", lbl, state, T.get_d(), t);
	for (const str &s : info)
		fprintf(f, ",%s", s.c_str());
	fprintf(f, "\n");
}

static vec<smlp_result>
optimize_EA(cmp_t direction,
            const domain &dom,
            const sptr<term2> &objective,
            const sptr<form2> &alpha,
            const sptr<form2> &beta,
            const sptr<form2> &eta,
            const kay::Q &delta,
            ival &obj_range,
            const kay::Q &max_prec,
            const fun<sptr<form2>(opt<kay::Q> delta, const hmap<str,sptr<term2>> &)> &theta,
            const char *logic = nullptr)
{
	assert(is_order(direction));

	/* optimize T in obj_range such that (assuming direction is >=):
	 *
	 * E x . eta x /\
	 * A y . theta x y -> alpha y -> (beta y /\ obj y >= T)
	 *
	 * domain constraints from 'dom' have to hold for x and y.
	 */
/*
	fprintf(stderr, "dom: ");
	dump_smt2(stderr, dom);
	fprintf(stderr, "alpha: ");
	dump_smt2(stderr, *alpha);
	fprintf(stderr, "\nbeta: ");
	dump_smt2(stderr, *beta);
	fprintf(stderr, "\n");
*/
	vec<smlp_result> results, counter_examples;

	while (length(obj_range) > max_prec) {
		printf("r,%g,%g,%g\n",
		       obj_range.lo.get_d(), obj_range.hi.get_d(),
		       max_prec.get_d());
		kay::Q T = mid(obj_range);
		sptr<term2> threshold = make2t(cnst2 { T });

		/* eta x /\ alpha x /\ (beta x /\ obj x >= T) */
		sptr<form2> target = conj({
			beta,
			make2f(prop2 { direction, objective, threshold })
		});
		uptr<solver> exists = mk_solver(true, logic);
		exists->declare(dom);
		exists->add(eta);
		exists->add(alpha);
		exists->add(target);

		while (true) {
			timing e0;
			result e = exists->check();
			trace_result(stdout, "a", e, T, timing() - e0);
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

			uptr<solver> forall = mk_solver(false, logic);
			forall->declare(dom);
			/* ! ( theta x y -> alpha y -> beta y /\ obj y >= T ) =
			 * ! ( ! theta x y \/ ! alpha y \/ beta y /\ obj y >= T ) =
			 * theta x y /\ alpha y /\ ! ( beta y /\ obj y >= T) */
			forall->add(theta({}, candidate));
			forall->add(alpha);
			forall->add(neg(target));
			/*
			file test("ce.smt2", "w");
			smlp::dump_smt2(test, dom);
			fprintf(test, "(assert ");
			smlp::dump_smt2(test, *theta(true, candidate));
			fprintf(test, ")\n");
			fprintf(test, "(assert ");
			smlp::dump_smt2(test, *alpha);
			fprintf(test, ")\n");
			fprintf(test, "(assert ");
			smlp::dump_smt2(test, lbop2 { lbop2::OR, {
				make2f(lneg2 { beta }),
				make2f(prop2 { ~direction, objective, threshold })
			} });
			fprintf(test, ")\n");
			*/

			timing a0;
			result a = forall->check();
			trace_result(stdout, "b", a, T, timing() - a0);
			if (unknown *u = a.get<unknown>())
				DIE(2,"forall is unknown: %s\n", u->reason.c_str());
			if (a.get<unsat>()) {
				// fprintf(file("ce-z3.smt2", "w"), "%s\n", forall.slv.to_smt2().c_str());
				results.emplace_back(T, candidate);
				if (is_less(direction))
					obj_range.hi = T;
				else
					obj_range.lo = T;
				break;
			}
			auto &counter_example = a.get<sat>()->model;
			counter_examples.emplace_back(T, counter_example);
			exists->add(make2f(lneg2 { theta({ delta }, counter_example) }));
		}
	}

	return results;
}

static void alarm_handler(int sig)
{
	if (interruptible *p = interruptible::is_active)
		p->interrupt();
	signal(sig, alarm_handler);
}

static void sigint_handler(int sig)
{
	if (interruptible *p = interruptible::is_active)
		p->interrupt();
	signal(sig, sigint_handler);
	// raise(sig);
}

static void print_model(FILE *f, const hmap<str,sptr<term2>> &model, int indent)
{
	size_t k = 0;
	for (const auto &[n,_] : model)
		k = max(k, n.length());
	for (const auto &[n,c] : model)
		fprintf(f, "%*s%*s = %s\n", indent, "", -(int)k, n.c_str(),
		        to_string(c->get<cnst2>()->value).c_str());
}

#ifdef SMLP_ENABLE_KERAS_NN
# define USAGE "{ DOMAIN EXPR | H5-NN SPEC GEN IO-BOUNDS }"
#else
# define USAGE "DOMAIN EXPR"
#endif

[[noreturn]]
static void usage(const char *program_name, int exit_code)
{
	FILE *f = exit_code ? stderr : stdout;
	fprintf(f, "\
usage: %s [-OPTS] [--] " USAGE " OP [CNST]\n\
", program_name);
	if (!exit_code)
		fprintf(f,"\
\n\
Options [defaults]:\n\
  -1           use single objective from GEN instead of all H5-NN outputs [no]\n\
  -a ALPHA     additional ALPHA constraints restricting candidates *and*\n\
               counter-examples (only points in regions satisfying ALPHA\n\
               are considered counter-examples to safety); can be given multiple\n\
               times, the conjunction of all is used [true]\n\
  -b BETA      additional BETA constraints restricting candidates and safe\n\
               regions (all points in safe regions satisfy BETA); can be given\n\
               multiple times, the conjunction of all is used [true]\n\
  -c           clamp inputs (only meaningful for NNs) [no]\n\
  -C COMPAT    use a compatibility layer, can be given multiple times; supported\n\
               values for COMPAT:\n\
               - python: reinterpret floating point constants as python would\n\
                         print them\n\
  -d DELTA     increase radius around counter-examples by factor (1+DELTA) [0]\n\
  -e ETA       additional ETA constraints restricting only candidates, can be\n\
               given multiple times, the conjunction of all is used [true]\n\
  -F IFORMAT   determines the format of the EXPR file; can be one of: 'infix',\n\
               'prefix' [infix]\n\
  -h           displays this help message\n\
  -i SUBDIVS   use interval evaluation (only when CNST is given) with SUBDIVS\n\
               subdivision [no]\n\
  -I EXT-INC   optional external incremental SMT solver [value for -S]\n\
  -n           dry run, do not solve the problem [no]\n\
  -O OUT-BNDS  scale output according to min-max output bounds (.csv, only\n\
               meaningful for NNs) [none]\n\
  -p           dump the expression in Polish notation to stdout [no]\n\
  -P PREC      maximum precision to obtain the optimization result for [0.05]\n\
  -Q QUERY     answer a query about the problem; supported QUERY:\n\
               - vars: list all variables\n\
  -r           re-cast bounded integer variables as reals with equality\n\
               constraints\n\
  -R LO,HI     optimize threshold in the interval [LO,HI] [0,1]\n\
  -s           dump the problem in SMT-LIB2 format to stdout [no]\n\
  -S EXT-CMD   invoke external SMT solver instead of the built-in one via\n\
               'SHELL -c EXT-CMD' where SHELL is taken from the environment or\n\
               'sh' if that variable is not set []\n\
  -t TIMEOUT   set the solver timeout in seconds, 0 to disable [0]\n\
  -V           display version information\n\
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
as those in the EXPR file (if any).\n\
\n\
Exit codes are as follows:\n\
  0: normal operation\n\
  1: invalid user input\n\
  2: unexpected SMT solver output (e.g., 'unknown' on interruption)\n\
  3: unhandled SMT solver result (e.g., non-rational assignments)\n\
  4: partial function applicable outside of its domain (e.g., 'Match(expr, .)')\n\
\n\
Developed by Franz Brausse <franz.brausse@manchester.ac.uk>.\n\
License: Apache 2.0; part of SMLP.\n\
");
	exit(exit_code);
}

static void version_info()
{
	printf("SMLP version %d.%d.%d\n", SMLP_VERSION_MAJOR,
	       SMLP_VERSION_MINOR, SMLP_VERSION_PATCH);
	printf("Built with features:"
#ifdef KAY_USE_FLINT
	       " flint"
#endif
#ifdef SMLP_ENABLE_KERAS_NN
	       " keras-nn"
#endif
#ifdef SMLP_ENABLE_Z3_API
	       " z3"
#endif
	       "\n");
	printf("Libraries:\n");
	printf("  GMP version %d.%d.%d linked %s\n",
	       __GNU_MP_VERSION, __GNU_MP_VERSION_MINOR,
	       __GNU_MP_VERSION_PATCHLEVEL, __gmp_version);
#ifdef KAY_USE_FLINT
	printf("  Flint version %s linked %s\n",
	       FLINT_VERSION, flint_version);
	/*
	printf("  MPFR version %s linked %s\n",
	       MPFR_VERSION_STRING, mpfr_get_version());*/
#endif
	unsigned maj, min, pat, rev;
#ifdef SMLP_ENABLE_Z3_API
	Z3_get_version(&maj, &min, &pat, &rev);
	printf("  Z3 version %d.%d.%d linked %d.%d.%d\n",
	       Z3_MAJOR_VERSION, Z3_MINOR_VERSION,
	       Z3_BUILD_NUMBER, maj, min, pat);
#endif
#ifdef SMLP_ENABLE_KERAS_NN
	uint32_t kjson_v = kjson_version();
	printf("  kjson version %d.%d.%d linked %d.%d.%d\n",
	       KJSON_VERSION >> 16, (KJSON_VERSION >> 8) & 0xff,
	       KJSON_VERSION & 0xff, kjson_v >> 16,
	       (kjson_v >> 8) & 0xff, kjson_v & 0xff);
	H5get_libversion(&maj, &min, &pat);
	printf("  HDF5 version %d.%d.%d linked %d.%d.%d\n",
	       H5_VERS_MAJOR, H5_VERS_MINOR, H5_VERS_RELEASE,
	       maj, min, pat);
#endif
	(void)maj;
	(void)min;
	(void)pat;
	(void)rev;
}

template <decltype(lbop2::op) op>
static expr2s mk_lbop2(vec<expr2s> args)
{
	vec<sptr<form2>> b;
	b.reserve(args.size());
	for (expr2s &t : args)
		b.emplace_back(move(*t.get<sptr<form2>>()));
	return make2f(lbop2 { op, move(b) });
}

static expr2s mk_lneg2(vec<expr2s> args)
{
	assert(size(args) == 1);
	return make2f(lneg2 { move(*args.front().get<sptr<form2>>()) });
}

static sptr<form2> parse_infix_form2(const char *s)
{
	static const unroll_funs_t logic = {
		{"And", mk_lbop2<lbop2::AND>},
		{"Or", mk_lbop2<lbop2::OR>},
		{"Not", mk_lneg2},
	};
	return *unroll(parse_infix(s, false), logic).get<sptr<form2>>();
}

static void dump_smt2_line(FILE *f, const char *pre, const sptr<form2> &g)
{
	fprintf(f, "%s", pre);
	smlp::dump_smt2(f, *g);
	fprintf(f, "\n");
}

interruptible *interruptible::is_active;

template <typename T>
static bool from_string(const char *s, T &v)
{
	using std::from_chars;
	using kay::from_chars;
	auto [end,ec] = from_chars(s, s + strlen(s), v);
	return !*end && ec == std::errc {};
}

int main(int argc, char **argv)
{
	/* these determine the mode of operation of this program */
	bool             single_obj    = false;
	bool             solve         = true;
	bool             dump_pe       = false;
	bool             dump_smt2     = false;
	bool             infix         = true;
	bool             python_compat = false;
	bool             inject_reals  = false;
	int              timeout       = 0;
	bool             clamp_inputs  = false;
	const char      *out_bounds    = nullptr;
	const char      *max_prec      = "0.05";
	vec<sptr<form2>> alpha_conj    = {};
	vec<sptr<form2>> beta_conj     = {};
	vec<sptr<form2>> eta_conj      = {};
	const char      *delta_s       = "0";
	ival             obj_range     = { 0, 1 };
	vec<strview>     queries;

	/* parse options from the command-line */
	for (int opt; (opt = getopt(argc, argv,
	                            ":1a:b:cC:d:e:F:hi:I:nO:pP:Q:rR:sS:t:V")) != -1;)
		switch (opt) {
		case '1': single_obj = true; break;
		case 'a': alpha_conj.emplace_back(parse_infix_form2(optarg)); break;
		case 'b': beta_conj.emplace_back(parse_infix_form2(optarg)); break;
		case 'c': clamp_inputs = true; break;
		case 'C':
			if (optarg == "python"sv)
				python_compat = true;
			else
				DIE(1,"error: option '-C' only supports "
				      "'python'\n");
			break;
		case 'd': delta_s = optarg; break;
		case 'e': eta_conj.emplace_back(parse_infix_form2(optarg)); break;
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
		case 'i': {
			if (from_string(optarg, intervals))
				break;
			DIE(1,"error: SUBDIVS argument to '-i' must be numeric\n");
		}
		case 'I': inc_solver_cmd = optarg; break;
		case 'n': solve = false; break;
		case 'p': dump_pe = true; break;
		case 'P': max_prec = optarg; break;
		case 'Q': queries.push_back(optarg); break;
		case 'r': inject_reals = true; break;
		case 'R': {
			char *comma = strchr(optarg, ',');
			if (!comma || !comma[1])
				DIE(1,"error: '-R' expects two comma-separated "
				      "parameters, got: '%s'",optarg);
			*comma = '\0';
			obj_range.lo = kay::Q_from_str(optarg);
			obj_range.hi = kay::Q_from_str(comma+1);
			break;
		}
		case 'O': out_bounds = optarg; break;
		case 's': dump_smt2 = true; break;
		case 'S': ext_solver_cmd = optarg; break;
		case 't': timeout = atoi(optarg); break;
		case 'V': version_info(); exit(0);
		case ':': DIE(1,"error: option '-%c' requires an argument\n",
		              optopt);
		case '?': DIE(1,"error: unknown option '-%c'\n",optopt);
		}

	pre_problem pp;

#ifdef SMLP_ENABLE_KERAS_NN
	if (argc - optind >= 5) {
		/* Solve NN problem */
		const char *hdf5_path = argv[optind];
		const char *spec_path = argv[optind+1];
		const char *gen_path = argv[optind+2];
		const char *io_bounds = argv[optind+3];
		pp = parse_nn(gen_path, hdf5_path, spec_path, io_bounds,
		              out_bounds, clamp_inputs, single_obj);
		optind += 4;
	} else
#else
	/* these are unused w/o NN support */
	(void)single_obj;
	(void)clamp_inputs;
	(void)out_bounds;
#endif
	if (argc - optind >= 3) {
		/* Solve polynomial problem */
		pp = parse_poly_problem(argv[optind], argv[optind+1],
		                        python_compat, dump_pe, infix);
		optind += 2;
	} else
		usage(argv[0], 1);

	auto &[dom,lhs,funs,in_bnds,eta,pc,theta] = pp;

	if (inject_reals)
		for (const auto &[n,i] : in_bnds) {
			component *c = dom[n];
			assert(c);
			if (c->type != component::INT)
				continue;
			if (!c->range.get<entire>())
				continue;
			list l;
			using namespace kay;
			for (Z z = ceil(i.lo); z <= floor(i.hi); z++)
				l.values.emplace_back(z);
			c->range = move(l);
			c->type = component::REAL;
		}

	for (const auto &[n,i] : in_bnds) {
		sptr<term2> v = make2t(name { n });
		alpha_conj.emplace_back(make2f(lbop2 { lbop2::AND, {
			make2f(prop2 { GE, v, make2t(cnst2 { i.lo }) }),
			make2f(prop2 { LE, v, make2t(cnst2 { i.hi }) }),
		} }));
	}
	sptr<form2> alpha = make2f(lbop2 { lbop2::AND, move(alpha_conj) });
	dump_smt2_line(stderr, "alpha:", alpha);
	alpha = subst(alpha, funs);

	sptr<form2> beta = make2f(lbop2 { lbop2::AND, move(beta_conj) });
	dump_smt2_line(stderr, "beta: ", beta);
	beta = subst(beta, funs);

	eta_conj.emplace_back(move(eta));
	eta = make2f(lbop2 { lbop2::AND, move(eta_conj) });
	dump_smt2_line(stderr, "eta: ", eta);
	eta = subst(eta, funs);

	fprintf(stderr, "domain:\n");
	smlp::dump_smt2(stderr, dom);

	for (const strview &q : queries) {
		fprintf(stderr, "query '%.*s':\n", (int)q.size(),q.data());
		if (q == "vars") {
			hset<str> h = free_vars(lhs);
			vec<str> v(begin(h), end(h));
			sort(begin(v), end(v));
			for (const str &id : v)
				fprintf(stderr, "  '%s': %s\n", id.c_str(),
				        dom[id] ? "bound" : "free");
		} else
			DIE(1,"error: unknown query '%.*s'\n",(int)q.size(),q.data());
	}

	/* Check that the constraints from partial function evaluation are met
	 * on the domain. */
	result ood = solve_exists(dom, conj({ alpha, make2f(lneg2 { pc }) }));
	if (const sat *s = ood.get<sat>()) {
		fprintf(stderr,
		        "error: ALPHA and DOMAIN constraints do not imply that "
		        "all function parameters are inside the respective "
		        "function's domain, e.g.:\n");
		print_model(stderr, s->model, 2);
		DIE(4, "");
	}

	/* find out about the OP comparison operation */
	size_t c;
	for (c=0; c<ARRAY_SIZE(cmp_s); c++)
		if (std::string_view(cmp_s[c]) == argv[optind])
			break;
	if (c == ARRAY_SIZE(cmp_s))
		DIE(1,"OP '%s' unknown\n",argv[optind]);
	optind++;

	/* hint for the solver: (non-)linear real arithmetic, potentially also
	 * with integers */
	str logic = smt2_logic_str(dom, lhs); /* TODO: check all expressions in funs */

	if (argc - optind >= 1) {
		if (*beta != *true2)
			DIE(1,"-b BETA is not supported when CNST is given\n");

		/* interpret the CNST on the right hand side */
		kay::Q cnst;
		if (!from_string(argv[optind], cnst))
			DIE(1,"CNST must be a numeric constant\n");
		fprintf(stderr, "cnst: %s\n", cnst.get_str().c_str());
		sptr<term2> rhs = make2t(cnst2 { move(cnst) });

		/* the problem consists of domain and the (EXPR OP CNST) constraint */
		problem p = {
			move(dom),
			lbop2 { lbop2::AND, {
				eta,
				alpha,
				make2f(prop2 { (cmp_t)c, lhs, rhs, }) }
			},
		};

		/* optionally dump the smt2 representation of the problem */
		if (dump_smt2)
			::dump_smt2(stdout, logic.c_str(), p);

		if (timeout > 0) {
			signal(SIGALRM, alarm_handler);
			alarm(timeout);
		}

		// signal(SIGINT, sigint_handler);

		/* optionally solve the problem */
		if (solve)
			solve_exists(p.dom, make2f(p.p), logic.c_str()).match(
			[&,lhs=lhs](const sat &s) {
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
	} else if (solve) {
		if (intervals)
			DIE(1,"error: -i is not supported for optimization\n");

		kay::Q max_p = kay::Q_from_str(str(max_prec).data());
		vec<smlp_result> r = optimize_EA((cmp_t)c, dom, lhs,
		                                 alpha, beta, eta,
		                                 kay::Q_from_str(str(delta_s).data()),
		                                 obj_range, max_p,
		                                 theta, logic.c_str());
		if (empty(r)) {
			fprintf(stderr,
			        "no solution for objective in theta in "
			        "[%s, %s] ~ [%f, %f]\n",
			        obj_range.lo.get_str().c_str(),
			        obj_range.hi.get_str().c_str(),
			        obj_range.lo.get_d(), obj_range.hi.get_d());
		} else {
			fprintf(stderr,
			        "%s of objective in theta in [%s, %s] ~ [%f, %f] around:\n",
			        is_less((cmp_t)c) ? "min max" : "max min",
			        obj_range.lo.get_str().c_str(),
			        obj_range.hi.get_str().c_str(),
			        obj_range.lo.get_d(), obj_range.hi.get_d());
			print_model(stderr, r.back().point, 2);
			for (const auto &s : r) {
				kay::Q c = s.center_value(lhs);
				fprintf(stderr,
				        "T: %s ~ %f -> center: %s ~ %f\n",
				        s.threshold.get_str().c_str(),
				        s.threshold.get_d(),
				        c.get_str().c_str(), c.get_d());
			}
		}
	}
}
