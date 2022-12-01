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
	assert(s[0] == '"');
	assert(s[s.length()-1] == '"');
	return s.substr(1, s.length() - 2);
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

static expr parse_smt2(const es::arg &a)
{
	using es::sexpr;
	using es::slit;

	if (const slit *s = std::get_if<slit>(&a)) {
		kay::Q q;
		auto r = kay::from_chars(s->data(), s->data() + s->length(), q);
		if (r.ec == std::errc {} && r.ptr == s->data() + s->length())
			return cnst { *s };
		return name { *s };
	}
	const sexpr &s = std::get<sexpr>(a);
	assert(size(s) > 0);
	expr e;
	vec<expr> v;
	for (size_t i=0; i<size(s); i++) {
		expr f = parse_smt2(s[i]);
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

static hmap<size_t,kay::Q> parse_upoly(const sptr<term2> &t, const str &var)
{
	using poly = hmap<size_t,kay::Q>;

	poly ret;
	t->match(
	[&](const name &n) { assert(n.id == var); ret[1] = 1; },
	[&](const bop2 &b) {
		poly l = parse_upoly(b.left, var);
		poly r = parse_upoly(b.right, var);
		switch (b.op) {
		case bop2::SUB:
			for (auto &[e,c] : r)
				neg(c);
			/* fall through */
		case bop2::ADD:
			ret = move(l);
			for (const auto &[e,c] : r)
				ret[e] += c;
			break;
		case bop2::MUL:
			for (const auto &[e,c] : l)
				for (const auto &[f,d] : r)
					ret[e+f] += c * d;
			break;
		}
	},
	[&](const uop2 &u) {
		poly o = parse_upoly(u.operand, var);
		switch (u.op) {
		case uop2::USUB:
			for (auto &[e,c] : o)
				neg(c);
			/* fall through */
		case uop2::UADD:
			ret = move(o);
			break;
		}
	},
	[&](const cnst2 &c) { ret[0] = to_Q(c.value); },
	[&](const ite2 &) { assert(0); }
	);
	erase_if(ret, [](const auto &p){ return p.second == 0; });
	return ret;
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

static void parse_algebraic_cvc5(str s)
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
	}).get<sptr<term2>>();
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
}

static void parse_algebraic_cvc4(const es::sexpr &vars, const es::arg &witness)
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
	}).get<sptr<form2>>();
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
}

static void parse_algebraic_z3(const es::arg &p, const es::slit &n)
{
	expr c = parse_smt2(p);
	sptr<term2> poly = *unroll(c, {
		{ "+", unroll_add },
		{ "-", unroll_sub },
		{ "*", unroll_mul },
		{ "^", unroll_expz },
		{ "/", unroll_div_cnst },
	}).get<sptr<term2>>();
	assert(poly);
	/* univariate non-linear polynomial */
	hset<str> vars = free_vars(poly);
	assert(size(vars) == 1);
	assert(is_nonlinear(poly));
	hmap<size_t,kay::Q> up = parse_upoly(poly, *begin(vars));
}

static kay::Q Q_from_smt2(const es::arg &s)
{
	using namespace kay;
	using es::slit;
	using es::sexpr;

	if (const slit *sls = std::get_if<slit>(&s))
		return Q_from_str(str(*sls).data());
	const sexpr &se = std::get<sexpr>(s);
	if (size(se) == 2 && matches(se[0], "to_real"))
		return Q_from_smt2(se[1]);
	if (size(se) == 2 && matches(se[0], "-"))
		return -Q_from_smt2(se[1]);
	if (size(se) == 2 && matches(se[0], "+"))
		return +Q_from_smt2(se[1]);
	if (size(se) > 2 && matches(se[0], "_") &&
	    matches(se[1], "real_algebraic_number")) {
		/* cvc5 1.0.2 algebraics, opt --nl-cov (requires libpoly):
		 * (_ real_algebraic_number (<1*x^2 + (-2), (5/4, 3/2)>)) */
		es::sexpr a;
		for (size_t i=2; i<size(se); i++)
			a.emplace_back(se[i]);
		str s = to_string(a); /* (<1*x^2 + (-2) , (5/4, 3/2) >) */
		parse_algebraic_cvc5(move(s));
		abort();
	}
	if (size(se) == 3 && matches(se[0], "witness")) {
		/* cvc4 1.8 algebraics:
		 * (witness ((BOUND_VARIABLE_775 Real))
		 *          (and (>= BOUND_VARIABLE_775 (/ 741455 524288))
		 *               (>= (* (- 1.0) BOUND_VARIABLE_775)
		 *                   (/ (- 1482917) 1048576)))) */
		const sexpr &vars = std::get<sexpr>(se[1]);
		parse_algebraic_cvc4(vars, se[2]);
		abort();
	}
	if (size(se) == 3 && matches(se[0], "root-obj")) {
		/* z3: (root-obj (+ (^ x 2) (- 2)) 1) */
		parse_algebraic_z3(se[1], std::get<slit>(se[2]));
		abort();
	}
	assert(size(se) == 3);
	assert(matches(se[0], "/"));
	return Q_from_smt2(se[1]) / Q_from_smt2(se[2]);
}

static pair<str,sptr<term2>> parse_smt2_asgn(const es::sexpr &a)
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
	return { v, make2t(cnst2 { Q_from_smt2(b) }) };
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
