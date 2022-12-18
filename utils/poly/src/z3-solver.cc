/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "z3-solver.hh"
#include "dump-smt2.hh"

#include <sstream>

#include <sys/time.h>
#include <signal.h>

using namespace smlp;

z3_solver::z3_solver(const char *logic)
: slv(ctx)
{
	if (logic)
		slv.set("logic", logic);
}

void z3_solver::declare(const domain &d)
{
	for (const auto &[var,rng] : d) {
		const char *s = var.c_str();
		z3::expr e(ctx);
		switch (rng.type) {
		case type::INT: e = ctx.int_const(s); break;
		case type::REAL: e = ctx.real_const(s); break;
		}
		symbols.emplace(var, move(e));
		add(domain_constraint(var, rng));
	}
}

z3::expr z3_solver::interp(const sptr<form2> &f, hmap<void *, z3::expr> &m)
{
	auto it = m.find(f.get());
	if (it == m.end())
		it = m.emplace(f.get(), interp(*f, m)).first;
	return it->second;
}

z3::expr z3_solver::interp(const sptr<term2> &e, hmap<void *, z3::expr> &m)
{
	auto it = m.find(e.get());
	if (it == m.end())
		it = m.emplace(e.get(), interp(*e, m)).first;
	return it->second;
}

z3::expr z3_solver::interp(const form2 &f, hmap<void *, z3::expr> &m)
{
	return f.match(
	[&](const prop2 &p) {
		return do_cmp(interp(p.left, m), p.cmp, interp(p.right, m));
	},
	[&](const lbop2 &b) {
		z3::expr_vector a(ctx);
		for (const sptr<form2> &f : b.args)
			a.push_back(interp(f, m));
		switch (b.op) {
		case lbop2::OR: return z3::mk_or(a);
		case lbop2::AND: return z3::mk_and(a);
		}
		unreachable();
	},
	[&](const lneg2 &n) {
		return !interp(n.arg, m);
	}
	);
}

z3::expr z3_solver::interp(const term2 &e, hmap<void *, z3::expr> &m)
{
	return e.match(
	[&](const name &n){
		auto it = symbols.find(n.id);
		if (it == symbols.end())
			MDIE(mod_z3,1,"reference to symbol '%s' not in domain\n",
			     n.id.c_str());
		return it->second;
	},
	[&](const cnst2 &c){
		return c.value.match(
		[&](const kay::Z &v){ return ctx.int_val(v.get_str().c_str()); },
		[&](const kay::Q &v){ return ctx.real_val(v.get_str().c_str()); },
		[&](const A &) -> z3::expr { MDIE(mod_z3,5,"algebraic real constants are unsupported\n"); }
		);
	},
	[&](const ite2 &i){
		return ite(interp(i.cond, m), interp(i.yes, m), interp(i.no, m));
	},
	[&](const bop2 &b){
		z3::expr l = interp(b.left, m);
		z3::expr r = interp(b.right, m);
		switch (b.op) {
		case bop2::ADD: return l + r;
		case bop2::SUB: return l - r;
		case bop2::MUL: return l * r;
		}
		unreachable();
	},
	[&](const uop2 &u){
		z3::expr a = interp(u.operand, m);
		switch (u.op) {
		case uop2::UADD: return a;
		case uop2::USUB: return -a;
		}
		unreachable();
	}
	);
}

static struct itimerval old_timer;
static struct sigaction old_alrm_handler;
static bool replace_sigint_handler_has_run;

static void replace_sigint_handler(int)
{
	signal(SIGINT, SIG_DFL);
	sigaction(SIGVTALRM, &old_alrm_handler, NULL);
	setitimer(ITIMER_VIRTUAL, &old_timer, NULL);
	replace_sigint_handler_has_run = true;
}

static cnst2 parse_z3_cnst(const str &id, const z3::expr &e)
{
	cnst2 c;
	if (str num; e.is_numeral(num)) {
		//std::cerr << "z3: parsing cnst " << num << " -> ";
		if (strchr(num.c_str(), '/'))
			c.value = kay::Q(num.c_str());
		else
			c.value = kay::Z(num.c_str());
	} else if (e.is_algebraic()) {
		z3::expr_vector coeffs = e.algebraic_poly();
		unsigned i = e.algebraic_i();
		z3::expr l = e.algebraic_lower(solver::alg_dec_prec_approx);
		z3::expr u = e.algebraic_upper(solver::alg_dec_prec_approx);

		cnst2 cl = parse_z3_cnst(id, l);
		cnst2 cu = parse_z3_cnst(id, u);

		hmap<size_t,kay::Q> cc;
		for (size_t i=0; i<coeffs.size(); i++)
			cc[i] = to_Q(parse_z3_cnst(id, coeffs[i]).value);

		c.value = A({ to_Q(cl.value), to_Q(cu.value) }, "x",
		            upoly(move(cc)), i);
	} else {
		std::stringstream ss;
		ss << e;
		MDIE(mod_z3,3,"expected numeral or algebraic "
		     "assignment for %s, got %s\n", id.c_str(),
		     ss.str().c_str());
	}
	return c;
}

result z3_solver::check()
{
	info(mod_z3,"solving...\n");

	/* Z3 overrides the SIGINT handler with something that does not work
	 * reliably; we create a timer to signal us and then replace the SIGINT
	 * handler back to the default. */
	replace_sigint_handler_has_run = false;

	/* Set up the alarm handler for the timer */
	struct sigaction signew;
	memset(&signew, 0, sizeof(signew));
	signew.sa_handler = replace_sigint_handler;
	sigaction(SIGVTALRM, &signew, &old_alrm_handler);

	/* Set up the timer */
	struct itimerval timnew;
	getitimer(ITIMER_VIRTUAL, &timnew);
	timnew.it_value.tv_usec += 100000; /* 100ms "ought to be enough for every CPU" */
	setitimer(ITIMER_VIRTUAL, &timnew, &old_timer);

	interruptible::is_active = this;
	z3::check_result r = slv.check();
	interruptible::is_active = nullptr;

	/* Restore previous timer and alarm handler */
	sigset_t newset;
	sigemptyset(&newset);
	sigaddset(&newset, SIGVTALRM);
	sigprocmask(SIG_BLOCK, &newset, NULL);
	if (!replace_sigint_handler_has_run) {
		sigaction(SIGVTALRM, &old_alrm_handler, NULL);
		setitimer(ITIMER_VIRTUAL, &old_timer, NULL);
	}
	sigprocmask(SIG_UNBLOCK, &newset, NULL);

	switch (r) {
	case z3::sat: {
		z3::model m = slv.get_model();
		dbg(mod_z3, "model #const: %zu, #symbs: %zu\n", m.num_consts(), size(symbols));
		assert(m.num_consts() == size(symbols));
		hmap<str,sptr<term2>> r;
		for (size_t i=0; i<size(symbols); i++)
		{
			z3::func_decl fd = m.get_const_decl(i);
			z3::expr e = m.get_const_interp(fd);
			str id = fd.name().str();
			r[id] = make2t(parse_z3_cnst(id, e));
			//std::cerr << to_string(r[id]->get<cnst2>()->value) << "\n";
		}
		//fprintf(stderr, "z3 model:\n");
		//std::cerr << m << "\n";
		return sat { move(r) };
	}
	case z3::unsat: return unsat {};
	case z3::unknown: return unknown { slv.reason_unknown() };
	}
	unreachable();
}
