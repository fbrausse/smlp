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
#include "nn.hh"

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

struct maxprob_solver {

	vec<kay::Q> Ts; /* ordered list of thresholds to test */
	
};

struct pareto {

	domain dom;
	vec<term2> objs;
	sptr<form2> alpha = true2;
	sptr<form2> beta  = true2;
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

	term2s operator()(vec<term2s> args)
	{
		assert(args.size() >= 2);
		const sptr<term2> &var = *args.front().get<sptr<term2>>();
		sptr<term2> r = move(*args.back().get<sptr<term2>>());
		int k = args.size()-3;
		if (!r) {
			vec<sptr<form2>> disj;
			for (int i=k; i >= 1; i-=2)
				disj.emplace_back(make2f(prop2 { EQ, var, *args[i].get<sptr<term2>>() }));
			constraints.emplace_back(make2f(lbop2 { lbop2::OR, move(disj) }));
			r = move(*args[k+1].get<sptr<term2>>());
			k -= 2;
		}
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

static sptr<form2> id_theta(bool /* left */, const hmap<str,sptr<term2>> &v)
{
	vec<sptr<form2>> conj;
	for (const auto &[n,e] : v)
		conj.emplace_back(make2f(prop2 { EQ, make2t(name { n }), e }));
	return make2f(lbop2 { lbop2::AND, move(conj) });
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
	Match match;
	sptr<term2> e2 = *unroll(e, { {"Match", std::ref(match)} }).get<sptr<term2>>();

	return pre_problem {
		move(d),
		move(e2),
		{},
		true2,
		make2f(lbop2 { lbop2::AND, move(match.constraints) }),
		id_theta,
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
	hmap<str,sptr<term2>> point;

	kay::Q center_value(const sptr<term2> &obj) const
	{
		return to_Q(cnst_fold(obj, point)->get<cnst2>()->value);
	}
};

static pair<vec<smlp_result>,opt<kay::Q>>
optimize_EA(cmp_t direction,
            const domain &dom,
            const sptr<term2> &objective,
            const sptr<form2> &alpha,
            const sptr<form2> &beta,
            ival &obj_range,
            const kay::Q &max_prec,
            const fun<sptr<form2>(bool left, const hmap<str,sptr<term2>> &)> theta,
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

	fprintf(stderr, "dom: ");
	dump_smt2(stderr, dom);
	fprintf(stderr, "alpha: ");
	dump_smt2(stderr, *alpha);
	fprintf(stderr, "\nbeta: ");
	dump_smt2(stderr, *beta);
	fprintf(stderr, "\n");

	vec<smlp_result> results, counter_examples;

	opt<kay::Q> known_safe;

	while (length(obj_range) > max_prec) {
		kay::Q T = mid(obj_range);
		sptr<term2> threshold = make2t(cnst2 { T });

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
				known_safe = T;
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

	return { move(results), move(known_safe) };
}

static void alarm_handler(int sig)
{
	if (z3::context *p = z3_solver::is_checking)
		p->interrupt();
	signal(sig, alarm_handler);
}

static void sigint_handler(int sig)
{
	signal(sig, sigint_handler);
	raise(sig);
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

static str smt2_logic_str(const domain &dom, const sptr<term2> &e)
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

template <decltype(lbop2::op) op>
static term2s mk_lbop2(vec<term2s> args)
{
	vec<sptr<form2>> b;
	b.reserve(args.size());
	for (term2s &t : args)
		b.emplace_back(move(*t.get<sptr<form2>>()));
	return make2f(lbop2 { op, move(b) });
}

[[noreturn]]
static void usage(const char *program_name, int exit_code)
{
	FILE *f = exit_code ? stderr : stdout;
	fprintf(f, "\
usage: %s [-OPTS] [--] { DOMAIN EXPR | H5-NN SPEC GEN IO-BOUNDS } OP [CNST]\n\
", program_name);
	if (!exit_code)
		fprintf(f,"\
\n\
Options [defaults]:\n\
  -a ALPHA     additional ALPHA constraints restricting candidates *and*\n\
               counter-examples (only points in regions satsifying ALPHA\n\
               are considered counter-examples to safety) [true]\n\
  -b BETA      additional BETA constraints restricting only candidates\n\
               (all points in safe regions satisfy BETA) [true]\n\
  -c           clamp inputs (only meaningful for NNs) [no]\n\
  -C COMPAT    use a compatibility layer, can be given multiple times; supported\n\
               values for COMPAT:\n\
               - python: reinterpret floating point constants as python would\n\
                         print them\n\
  -F IFORMAT   determines the format of the EXPR file; can be one of: 'infix',\n\
               'prefix' [infix]\n\
  -h           displays this help message\n\
  -n           dry run, do not solve the problem [no]\n\
  -O OUT-BNDS  scale output according to min-max output bounds (.csv) [none]\n\
  -p           dump the expression in Polish notation to stdout [no]\n\
  -P PREC      maximum precision to obtain the optimization result for [0.05]\n\
  -s           dump the problem in SMT-LIB2 format to stdout [no]\n\
  -t TIMEOUT   set the solver timeout in seconds, 0 to disable [0]\n\
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
	bool        single_obj    = false;
	bool        solve         = true;
	bool        dump_pe       = false;
	bool        dump_smt2     = false;
	bool        infix         = true;
	bool        python_compat = false;
	int         timeout       = 0;
	bool        clamp_inputs  = false;
	const char *out_bounds    = nullptr;
	str         max_prec      = "0.05";
	const char *alpha_s       = nullptr;
	const char *beta_s        = nullptr;

	/* parse options from the command-line */
	for (int opt; (opt = getopt(argc, argv, ":1a:b:cC:F:hnO:pP:st:")) != -1;)
		switch (opt) {
		case '1': single_obj = true; break;
		case 'a': alpha_s = optarg; break;
		case 'b': beta_s = optarg; break;
		case 'c': clamp_inputs = true; break;
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
		case 'P': max_prec = optarg; break;
		case 'O': out_bounds = optarg; break;
		case 's': dump_smt2 = true; break;
		case 't': timeout = atoi(optarg); break;
		case ':': DIE(1,"error: option '-%c' requires an argument\n",
		              optopt);
		case '?': DIE(1,"error: unknown option '-%c'\n",optopt);
		}

	pre_problem pp;

	if (argc - optind >= 5) {
		/* Solve NN problem */
		const char *hdf5_path = argv[optind];
		const char *spec_path = argv[optind+1];
		const char *gen_path = argv[optind+2];
		const char *io_bounds = argv[optind+3];
		pp = parse_nn(gen_path, hdf5_path, spec_path, io_bounds,
		              out_bounds, clamp_inputs, single_obj);
		optind += 4;
	} else if (argc - optind >= 3) {
		/* Solve polynomial problem */
		pp = parse_poly_problem(argv[optind], argv[optind+1],
		                        python_compat, dump_pe, infix);
		optind += 2;
	} else
		usage(argv[0], 1);

	auto &[dom,lhs,funs,alpha,pc,theta] = pp;

	fprintf(stderr, "defined outputs:\n");
	for (const auto &[k,e] : funs) {
		fprintf(stderr, "- '%s': ", k.c_str());
		smlp::dump_smt2(stderr, *e);
		fprintf(stderr, "\n");
	}

	/* hint for the solver: non-linear real arithmetic, potentially also
	 * with integers */
	str logic = smt2_logic_str(dom, lhs); /* TODO: check all expressions in funs */

	/* Check that the constraints from partial function evaluation are met
	 * on the domain. */
	z3_solver ood(dom, logic.c_str());
	ood.add(lneg2 { pc });
	ood.check().match(
	[](const sat &s) {
		fprintf(stderr, "error: DOMAIN constraints do not imply that "
		                "all function parameters are inside the "
		                "respective function's domain, e.g.:\n");
		print_model(stderr, s.model, 2);
		DIE(4, "");
	},
	[](const auto &) {}
	);

	/* find out about the OP comparison operation */
	size_t c;
	for (c=0; c<ARRAY_SIZE(cmp_s); c++)
		if (std::string_view(cmp_s[c]) == argv[optind])
			break;
	if (c == ARRAY_SIZE(cmp_s))
		DIE(1,"OP '%s' unknown\n",argv[optind]);
	optind++;

	if (alpha_s) {
		fprintf(stderr, "---------- extra alpha:\n");
		expr e = parse_infix(alpha_s, false);
		::dump_pe(stderr, e);
		fprintf(stderr, "---------- extra alpha done\n");
		term2s a = unroll(e, {
			{"And", mk_lbop2<lbop2::AND>},
			{"Or", mk_lbop2<lbop2::OR>}
		});
		fprintf(stderr, "extra alpha: ");
		a.match([](const auto &v) { smlp::dump_smt2(stderr, *v); });
		fprintf(stderr, "\n");
		alpha = make2f(lbop2 { lbop2::AND, { alpha, move(*a.get<sptr<form2>>()) } });
	}
	alpha = subst(alpha, funs);

	sptr<form2> beta = true2;
	if (beta_s) {
		fprintf(stderr, "---------- extra beta:\n");
		expr e = parse_infix(beta_s, false);
		::dump_pe(stderr, e);
		fprintf(stderr, "---------- extra beta done\n");
		term2s b = unroll(e, {
			{"And", mk_lbop2<lbop2::AND>},
			{"Or", mk_lbop2<lbop2::OR>}
		});
		fprintf(stderr, "extra beta: ");
		b.match([](const auto &v) { smlp::dump_smt2(stderr, *v); });
		fprintf(stderr, "\n");
		beta = make2f(lbop2 { lbop2::AND, { beta, move(*b.get<sptr<form2>>()) } });
	}
	beta = subst(beta, funs);

	if (argc - optind >= 1) {
		if (*beta != *true2)
			DIE(1,"-b BETA is not supported when CNST is given\n");

		/* interpret the CNST on the right hand side */
		sptr<term2> rhs = *unroll(cnst { argv[optind] }, {}).get<sptr<term2>>();

		/* the problem consists of domain and the (EXPR OP CNST) constraint */
		problem p = {
			move(dom),
			lbop2 { lbop2::AND, {
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

		signal(SIGINT, sigint_handler);

		/* optionally solve the problem */
		if (solve)
			solve_exists(p.dom, p.p, logic.c_str()).match(
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
	} else {
		ival obj_range(0,30);
		auto [r,t] = optimize_EA((cmp_t)c, dom, lhs,
		                         alpha, beta, obj_range,
		                         kay::Q_from_str(max_prec.data()),
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
			        "%s of objective in theta known safe threshold %s ~ %f, search range in "
			        "[%s, %s] ~ [%f, %f] around:\n",
			        is_less((cmp_t)c) ? "min max" : "max min",
			        t ? t->get_str().c_str() : "(none)",
			        t ? t->get_d() : 0.0,
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
				hmap<str,sptr<term2>> repl = s.point;
				for (int i=1; i<=4; i++) {
					repl["stackup"]->get<cnst2>()->value = kay::Z(i);
					kay::Q v = to_Q(cnst_fold(lhs, repl)->get<cnst2>()->value);
					fprintf(stderr, "  stackup=%d: %s ~ %f\n", i, v.get_str().c_str(), v.get_d());
				}
			}
		}
	}
}
