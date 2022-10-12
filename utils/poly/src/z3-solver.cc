/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "z3-solver.hh"

using namespace smlp;

z3_solver::z3_solver(const domain &d)
: slv(ctx)
{
	for (const auto &[var,rng] : d) {
		const char *s = var.c_str();
		symbols.emplace(var, is_real(rng) ? ctx.real_const(s)
		                                  : ctx.int_const(s));
		add(domain_constraint(var, rng));
	}
}

z3::expr z3_solver::interp(const form2 &f)
{
	return f.match(
	[&](const prop2 &p) {
		z3::expr l = interp(*p.left), r = interp(*p.right);
		switch (p.cmp) {
		case LE: return l <= r;
		case LT: return l <  r;
		case GE: return l >= r;
		case GT: return l >  r;
		case EQ: return l == r;
		case NE: return l != r;
		}
		unreachable();
	},
	[&](const lbop2 &b) {
		z3::expr_vector a(ctx);
		for (const form2 &f : b.args)
			a.push_back(interp(f));
		switch (b.op) {
		case lbop2::OR: return z3::mk_or(a);
		case lbop2::AND: return z3::mk_and(a);
		}
		unreachable();
	},
	[&](const lneg2 &n) { return !interp(*n.arg); }
	);
}

z3::expr z3_solver::interp(const expr2 &e)
{
	return e.match(
	[&](const name &n){
		auto it = symbols.find(n.id);
		assert(it != symbols.end());
		return it->second;
	},
	[&](const cnst2 &c){
		return c.value.match(
		[&](const str &s) -> z3::expr { abort(); },
		[&](const kay::Z &v){ return ctx.int_val(v.get_str().c_str()); },
		[&](const kay::Q &v){ return ctx.real_val(v.get_str().c_str()); }
		);
	},
	[&](const ite2 &i){
		return ite(interp(i.cond), interp(*i.yes), interp(*i.no));
	},
	[&](const bop2 &b){
		z3::expr l = interp(*b.left);
		z3::expr r = interp(*b.right);
		switch (b.op) {
		case bop::ADD: return l + r;
		case bop::SUB: return l - r;
		case bop::MUL: return l * r;
		}
		unreachable();
	},
	[&](const uop2 &u){
		z3::expr a = interp(*u.operand);
		switch (u.op) {
		case uop::UADD: return a;
		case uop::USUB: return -a;
		}
		unreachable();
	}
	);
}
