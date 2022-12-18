/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "ext-solver.hh"
#include "dump-smt2.hh"
#include "infix.hh"

#include <unistd.h>   /* pipe2(), dup2() */
#include <signal.h>   /* kill() */
#include <sys/wait.h> /* waitpid() */

using namespace smlp;

namespace {

struct Pipe {

	file rd, wr;

	explicit Pipe(int flags = 0)
	{
		int rw[2];
		if (pipe2(rw, flags))
			throw std::error_code(errno, std::system_category());
		rd = file(rw[0], "r");
		wr = file(rw[1], "w");
	}
};

}

process::process(const char *cmd)
{
	Pipe i, o/*, e*/;
	pid = fork();
	if (pid == -1)
		throw std::error_code(errno, std::system_category());
	if (!pid) {
		i.wr.close();
		o.rd.close();
		// e.rd.close();
		if (dup2(fileno(i.rd), STDIN_FILENO) == -1)
			throw std::error_code(errno, std::system_category());
		if (dup2(fileno(o.wr), STDOUT_FILENO) == -1)
			throw std::error_code(errno, std::system_category());
		/*
		if (dup2(fileno(e.wr), STDERR_FILENO) == -1)
			throw std::error_code(errno, std::system_category());*/
		const char *shell = getenv("SHELL");
		if (!shell)
			shell = "sh";
		execlp(shell, shell, "-c", cmd, NULL);
		throw std::error_code(errno, std::system_category());
	}
	i.rd.close();
	o.wr.close();
	// e.wr.close();
	in = move(i.wr);
	out = move(o.rd);
	// err = move(e.rd);
}

process::~process()
{
	if (pid == -1)
		return;
	in.close();
	out.close();
	// err.close();
	int status;
	waitpid(pid, &status, WNOHANG);
	if (!WIFEXITED(status)) {
		kill(pid, SIGTERM);
		waitpid(pid, &status, WUNTRACED);
	}
	log(mod_ext, WEXITSTATUS(status) ? WARN : NOTE,
	    "child %d exited with code %d\n", pid, WEXITSTATUS(status));
}

str ext_solver::get_info(const char *what)
{
	fprintf(in, "(get-info %s)\n", what);
	opt<es::sexpr> reply = out_s.next();
	assert(reply);
	assert(reply->size() == 2);
	assert(std::get<es::slit>((*reply)[0]) == what);
	const es::slit &s = std::get<es::slit>((*reply)[1]);
	assert(s.length() >= 2);
	if (s[0] == '"') {
		assert(s[s.length()-1] == '"');
		return s.substr(1, s.length() - 2);
	}
	return s;
}

ext_solver::ext_solver(const char *cmd, const char *logic)
: process(cmd)
, out_s((ungetc(' ', out), out))
{
	setvbuf(in, NULL, _IOLBF, 0);

	fprintf(in, "(set-option :print-success false)\n");

	name = get_info(":name");
	version = get_info(":version");
	std::string_view v = version;
	if (name == "MathSAT5" && v.starts_with("MathSAT5 version "))
		v = v.substr("MathSAT5 version "sv.length());
	else if (name == "ksmt" && v.starts_with("v"))
		v = v.substr(1);
	auto [_,ec] = from_chars(v.data(), v.data() + v.length(), parsed_version);
	assert(ec == std::errc {});
	note(mod_ext, "ext-solver pid %d: %s %s\n", pid, name.c_str(),
	     to_string(parsed_version).c_str());

	assert(!(name == "Yices" && parsed_version <= split_version {2,6,1}) || logic);
	if (name != "ksmt")
		fprintf(in, "(set-option :produce-models true)\n");
	if (logic)
		fprintf(in, "(set-logic %s)\n", logic);
}

void ext_solver::declare(const domain &d)
{
	assert(!n_vars);
	dump_smt2(in, d);
	n_vars = size(d);
}

static bool matches(const es::arg &a, const std::string_view &v)
{
	using es::slit;
	if (const slit *s = std::get_if<slit>(&a))
		return *s == v;
	return false;
}

static bool is_smt2_numeral(const es::slit &s)
{
	kay::Q q;
	const char *end = s.data() + s.length();
	auto r = kay::from_chars(s.data(), end, q);
	return r.ec == std::errc {} && r.ptr == end;
}

template <typename Numeral = decltype(is_smt2_numeral)>
static expr parse_smt2(const es::arg &a, Numeral &&is_num = is_smt2_numeral)
{
	using es::sexpr;
	using es::slit;

	if (const slit *s = std::get_if<slit>(&a)) {
		if (is_num(*s))
			return cnst { *s };
		return name { *s };
	}
	const sexpr &s = std::get<sexpr>(a);
	assert(size(s) > 0);
	expr e;
	vec<expr> v;
	for (size_t i=0; i<size(s); i++) {
		expr f = parse_smt2(s[i], is_num);
		if (!i)
			e = move(f);
		else
			v.emplace_back(move(f));
	}
	if (const name *n = std::get_if<name>(&e))
		for (size_t i=0; i<ARRAY_SIZE(cmp_smt2); i++)
			if (n->id == cmp_smt2[i]) {
				assert(size(v) >= 2);
				assert(size(v) == 2); /* more not implemented */
				return cop { static_cast<cmp_t>(i),
					make1e(move(v[0])),
					make1e(move(v[1])),
				};
			}
	return call { make1e(move(e)), move(v) };
}

/* returns whether it was a lower bound */
static bool parse_linear_bound(prop2 p, ival &v, const str &var)
{
	assert(is_order(p.cmp));
	hmap<size_t,kay::Q> monomials = parse_upoly(make2t(bop2 { bop2::SUB, p.left, p.right }), var);
	kay::Q c0 = monomials[0];
	kay::Q c1 = monomials[1];
	assert(size(monomials) == 2);
	assert(c1 != 0);
	if (c1 < 0)
		p.cmp = -p.cmp;
	c0 /= -c1;
	if (is_less(p.cmp)) {
		v.hi = c0;
		return false;
	} else {
		v.lo = c0;
		return true;
	}
}

static pair<hmap<size_t,kay::Q>,ival> parse_algebraic_cvc5(str s) /* v1.0.2 */
{
	using namespace kay;
	using es::slit;
	using es::sexpr;

	assert(s.length() > 4);
	assert(s[0] == '(' && s[s.length()-1] == ')');
	assert(s[1] == '<' && s[s.length()-2] == '>');
	s = s.substr(2, s.length() - 4);
	size_t c = s.find(',');
	assert(c != str::npos);
	str t = s.substr(0, c);
	fprintf(stderr, "'%s'\n", t.c_str());
	sptr<term2> poly = *unroll(parse_infix(t, false), {
		{ "+", unroll_add },
		{ "-", unroll_sub },
		{ "*", unroll_mul },
		{ "^", unroll_expz },
	}, unroll_cnst_ZQ).get<sptr<term2>>();
	assert(poly);
	/* univariate non-linear polynomial */
	hset<str> vars = free_vars(poly);
	assert(size(vars) == 1);
	assert(is_nonlinear(poly));
	s = s.substr(c+1);
	fprintf(stderr, "'%s'\n", s.c_str());
	c = s.find(',');
	assert(c != str::npos);
	size_t l = s.find('(');
	assert(l < c);
	size_t d = c+1;
	while (d < s.length() && isspace(s[d]))
		d++;
	size_t r = s.find(')', d);
	assert(r != str::npos);
	ival i;
	auto res = from_chars(s.data() + l + 1, s.data() + c, i.lo);
	assert(res.ec == std::errc {});
	res = from_chars(s.data() + d, s.data() + r, i.hi);
	assert(res.ec == std::errc {});
	fprintf(stderr, "range ~ [%g,%g]\n", i.lo.get_d(), i.hi.get_d());
	hmap<size_t,kay::Q> up = parse_upoly(poly, *begin(vars));

	return { move(up), move(i) };
}

static Ap parse_algebraic_cvc4(const es::sexpr &vars, const es::arg &witness) /* v1.8 */
{
	using namespace kay;
	using es::slit;
	using es::sexpr;

	assert(vars.size() == 1); /* univariate */
	const sexpr &var_type = std::get<sexpr>(vars[0]);
	assert(var_type.size() == 2);
	const slit &var = std::get<slit>(var_type[0]);
	expr c = parse_smt2(witness);
	sptr<form2> cc = *unroll(c, {
		{ "and", unroll_and },
		{ "+", unroll_add },
		{ "-", unroll_sub },
		{ "*", unroll_mul },
		{ "/", unroll_div_cnst },
	}, unroll_cnst_ZQ).get<sptr<form2>>();
	assert(cc);
	const lbop2 *cc2 = cc->get<lbop2>();
	assert(cc2);
	assert(cc2->op == lbop2::AND);
	assert(size(cc2->args) == 2);
	const prop2 *p0 = cc2->args[0]->get<prop2>();
	const prop2 *p1 = cc2->args[1]->get<prop2>();
	assert(p0);
	assert(p1);
	ival i;
	bool l0 = parse_linear_bound(*p0, i, var);
	bool l1 = parse_linear_bound(*p1, i, var);
	assert(l0 != l1); /* lower and upper bound */
	fprintf(stderr, "range ~ [%g,%g]\n", i.lo.get_d(), i.hi.get_d());

	return { i };
}

pair<hmap<size_t,kay::Q>,ival>
ext_solver::parse_algebraic_z3(const str &var, const es::arg &p, const es::slit &n)
{
	using namespace kay;

	expr c = parse_smt2(p);
	sptr<term2> poly = *unroll(c, {
		{ "+", unroll_add },
		{ "-", unroll_sub },
		{ "*", unroll_mul },
		{ "^", unroll_expz },
		{ "/", unroll_div_cnst },
	}, unroll_cnst_ZQ).get<sptr<term2>>();
	assert(poly);
	/* univariate non-linear polynomial */
	hset<str> vars = free_vars(poly);
	assert(size(vars) == 1);
	assert(is_nonlinear(poly));
	hmap<size_t,Q> up = parse_upoly(poly, *begin(vars));

	const unsigned &dec_prec = solver::alg_dec_prec_approx;

	assert(name == "Z3");
	fprintf(in, "(set-option :pp.decimal true)\n");
	fprintf(in, "(set-option :pp.decimal_precision %u)\n", dec_prec);
	fprintf(in, "(get-value (%s))\n", var.c_str());
	using es::sexpr;
	using es::slit;
	opt<sexpr> apx_asgns = out_s.next();
	assert(apx_asgns);
	fprintf(in, "(set-option :pp.decimal false)\n");
	fprintf(stderr, "got apx: %s\n", to_string(*apx_asgns).c_str());
	auto [apx_asgn] = as_tuple_ex<sexpr>(*apx_asgns);
	auto [v,t] = as_tuple_ex<slit,es::arg>(apx_asgn);
	assert(v == var);

	expr apx = parse_smt2(t, [](const slit &s){
		kay::Q q;
		const char *end = s.data() + s.length();
		auto r = kay::from_chars(s.data(), end, q);
		if (r.ec != std::errc {})
			return false;
		if (*r.ptr == '?')
			r.ptr++;
		return r.ptr == end;
	});

	hmap<str,ival> approximations;
	sptr<term2> t2 = *unroll(apx, { { "-", unroll_sub } },
	                         [&](const cnst &c) -> sptr<term2> {
		Q q;
		const char *begin = c.value.data();
		const char *end = begin + c.value.length();
		auto r = kay::from_chars(begin, end, q);
		if (r.ec != std::errc {})
			return nullptr;
		const char *dot = nullptr;
		if (*r.ptr == '?') {
			dot = strchr(begin, '.');
			assert(dot); /* no root-obj for integers */
			r.ptr++;
		}
		if (r.ptr != end)
			return nullptr;
		if (!dot)
			return make2t(cnst2 { move(q) });
		ssize_t prec = r.ptr - dot - 2;
		assert(prec >= dec_prec);
		str new_var = "_" + std::to_string(size(approximations));
		Q rad = pow(Q(1,10), prec);
		approximations[new_var] = { q - rad, q + rad };
		return make2t(smlp::name { move(new_var) });
	}).get<sptr<term2>>();
	assert(t2);
	dump_smt2(stderr, *t2);
	fprintf(stderr, "\n");
	assert(size(approximations) == 1);
	const auto &[w,x] = *begin(approximations);
	Q lo = to_Q(cnst_fold(t2, { { w, make2t(cnst2 { x.lo }) } })->get<cnst2>()->value);
	Q hi = to_Q(cnst_fold(t2, { { w, make2t(cnst2 { x.hi }) } })->get<cnst2>()->value);
	if (hi < lo)
		swap(lo, hi);
	ival i = { move(lo), move(hi) };
	fprintf(stderr, "range ~ [%g,%g]\n", i.lo.get_d(), i.hi.get_d());

	return { move(up), move(i) };
}

static kay::Q Q_from_smt2(const str &var, const es::arg &s)
{
	using namespace kay;
	using es::slit;
	using es::sexpr;

	if (const slit *sls = std::get_if<slit>(&s))
		return Q_from_str(str(*sls).data());
	const sexpr &se = std::get<sexpr>(s);
	if (size(se) == 2 && matches(se[0], "to_real"))
		return Q_from_smt2(var, se[1]);
	if (size(se) == 2 && matches(se[0], "-"))
		return -Q_from_smt2(var, se[1]);
	if (size(se) == 2 && matches(se[0], "+"))
		return +Q_from_smt2(var, se[1]);
	if (size(se) == 3 && matches(se[0], "/"))
		return Q_from_smt2(var, se[1]) / Q_from_smt2(var, se[2]);
	MDIE(mod_ext,2,"unhandled expression in SMT2 model: %s\n",to_string(se).c_str());
}

cnst2 ext_solver::cnst2_from_smt2(const str &var, const es::arg &s)
{
	using namespace kay;
	using es::slit;
	using es::sexpr;

	if (const slit *sls = std::get_if<slit>(&s))
		return cnst2 { Q_from_str(str(*sls).data()) };
	const sexpr &se = std::get<sexpr>(s);
	if (size(se) > 2 && matches(se[0], "_") &&
	    matches(se[1], "real_algebraic_number")) {
		/* cvc5 1.0.2 algebraics, opt --nl-cov (requires libpoly):
		 * (_ real_algebraic_number (<1*x^2 + (-2), (5/4, 3/2)>)) */
		es::sexpr a;
		for (size_t i=2; i<size(se); i++)
			a.emplace_back(se[i]);
		str s = to_string(a); /* (<1*x^2 + (-2) , (5/4, 3/2) >) */
		auto [up, i] = parse_algebraic_cvc5(move(s));
		return cnst2 { A(move(i), var, upoly(move(up)), 0) };
	} else if (size(se) == 3 && matches(se[0], "witness")) {
		/* cvc4 1.8 algebraics:
		 * (witness ((BOUND_VARIABLE_775 Real))
		 *          (and (>= BOUND_VARIABLE_775 (/ 741455 524288))
		 *               (>= (* (- 1.0) BOUND_VARIABLE_775)
		 *                   (/ (- 1482917) 1048576)))) */
		const sexpr &vars = std::get<sexpr>(se[1]);
		Ap ap = parse_algebraic_cvc4(vars, se[2]);
		assert(0);
	} else if (size(se) == 3 && matches(se[0], "root-obj")) {
		/* z3: (root-obj (+ (^ x 2) (- 2)) 1) */
		const slit &n = std::get<slit>(se[2]);
		auto [up, i] = parse_algebraic_z3(var, se[1], n);
		size_t root_idx;
		auto r = std::from_chars(n.data(), n.data() + n.length(), root_idx);
		assert(r.ec == std::errc {});
		return cnst2 { A(move(i), var, upoly(move(up)), root_idx) };
	} else
		return { Q_from_smt2(var, s) };
}

pair<str,sptr<term2>> ext_solver::parse_smt2_asgn(const es::sexpr &a)
{
	using es::slit;
	using es::arg;
	using es::sexpr;

	str v, t;
	arg b = slit("");
	if (size(a) == 2) {
		/* (name cnst) [MathSAT 5.6.3] */
		const auto &[var,s] = as_tuple_ex<slit,arg>(a);
		v = var;
		t = "Real"; /* dummy, unused */
		b = s;
	} else if (size(a) == 3) {
		/* (= name cnst) [Yices 2.6.1] */
		const auto &[eq,var,s] = as_tuple_ex<slit,slit,arg>(a);
		assert(eq == "=");
		v = var;
		t = "Real"; /* dummy, unused */
		b = s;
	} else if (size(a) == 4) {
		/* (define-const name type cnst) */
		const auto &[def,var,ty,s] = as_tuple_ex<slit,slit,slit,arg>(a);
		assert(def == "define-const");
		v = var;
		t = ty;
		b = s;
	} else {
		/* (define-fun name () type cnst) */
		assert(size(a) == 5);
		const auto &[def,var,none,ty,s] =
			as_tuple_ex<slit,slit,sexpr,slit,arg>(a);
		assert(def == "define-fun");
		assert(size(none) == 0);
		v = var;
		t = ty;
		b = s;
	}

	assert(t == "Real" || t == "Int");
	return { v, make2t(cnst2_from_smt2(v, b)) };
}

result ext_solver::check()
{
	using es::slit;
	using es::arg;
	using es::sexpr;

	info(mod_ext,"solving...\n");
	timing t;

	fprintf(in, "(check-sat)\n");
	out_s.skip_space();
	if (out_s.c == '(') {
		opt<sexpr> e = out_s.compound();
		assert(e);
		es::formatter f;
		f.f = stderr;
		f.emit(*e);
		fflush(stderr);
		abort();
	}

	opt<slit> res = out_s.atom();
	assert(res);
	note(mod_ext,"solved '%s' in %5.3fs\n", res->c_str(), (double)(timing {} - t));
	if (*res == "unsat")
		return unsat {};
	if (*res == "unknown")
		return unknown { get_info(":reason-unknown") };
	assert(*res == "sat");

	fprintf(in, "(get-model)\n");
	hmap<str,sptr<term2>> m;
	if (name == "Yices") {
		for (size_t i=0; i<n_vars; i++) {
			opt<sexpr> n = out_s.next();
			assert(n);
			assert(size(*n) == 3);
			auto [it,ins] = m.insert(parse_smt2_asgn(*n));
			assert(ins);
		}
		return sat { move(m) };
	}
	opt<sexpr> no = out_s.next();
	assert(no);
	const sexpr &n = *no;
	size_t off = 0;
	if (name == "cvc4" || name == "ksmt" ||
	    (name == "MathSAT5" && parsed_version >= split_version {5,6,8})) {
		assert(std::get<slit>(n[0]) == "model");
		off = 1;
	}
	assert(size(n) == off+n_vars);
	for (size_t i=0; i<n_vars; i++) {
		auto [it,ins] = m.insert(parse_smt2_asgn(std::get<sexpr>(n[off+i])));
		assert(ins);
	}
	return sat { move(m) };
}

void ext_solver::add(const sptr<form2> &f)
{
	fprintf(in, "(assert ");
	dump_smt2(in, *f);
	fprintf(in, ")\n");
}
