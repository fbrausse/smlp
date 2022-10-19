/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "z3-solver.hh"

#include <sstream>

using namespace smlp;

z3_solver::z3_solver(const domain &d, const char *logic)
: slv(ctx)
{
	if (logic)
		slv.set("logic", logic);
	for (const auto &[var,rng] : d) {
		const char *s = var.c_str();
		z3::expr e(ctx);
		switch (rng.type) {
		case component::INT: e = ctx.int_const(s); break;
		case component::REAL: e = ctx.real_const(s); break;
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
			DIE(1,"error: reference to symbol '%s' not in domain\n",
			    n.id.c_str());
		return it->second;
	},
	[&](const cnst2 &c){
		return c.value.match(
		[&](const kay::Z &v){ return ctx.int_val(v.get_str().c_str()); },
		[&](const kay::Q &v){ return ctx.real_val(v.get_str().c_str()); }
		);
	},
	[&](const ite2 &i){
		return ite(interp(i.cond, m), interp(i.yes, m), interp(i.no, m));
	},
	[&](const bop2 &b){
		z3::expr l = interp(b.left, m);
		z3::expr r = interp(b.right, m);
		switch (b.op) {
		case bop::ADD: return l + r;
		case bop::SUB: return l - r;
		case bop::MUL: return l * r;
		}
		unreachable();
	},
	[&](const uop2 &u){
		z3::expr a = interp(u.operand, m);
		switch (u.op) {
		case uop::UADD: return a;
		case uop::USUB: return -a;
		}
		unreachable();
	}
	);
}

z3::context *z3_solver::is_checking;

result z3_solver::check()
{
	is_checking = &ctx;
	z3::check_result r = slv.check();
	is_checking = nullptr;
	switch (r) {
	case z3::sat: {
		z3::model m = slv.get_model();
		assert(m.num_consts() == size(symbols));
		hmap<str,sptr<term2>> r;
		for (size_t i=0; i<size(symbols); i++)
		{
			z3::func_decl fd = m.get_const_decl(i);
			z3::expr e = m.get_const_interp(fd);
			str id = fd.name().str();
			str num;
			if (!e.is_numeral(num)) {
				std::stringstream ss;
				ss << e;
				DIE(3,"error: expected numeral assignment "
				      "for %s, got %s\n",
				    id.c_str(), ss.str().c_str());
			}
			//std::cerr << "z3: parsing cnst " << num << " -> ";
			cnst2 c;
			if (strchr(num.c_str(), '/'))
				c.value = kay::Q(num.c_str());
			else
				c.value = kay::Z(num.c_str());
			r[id] = make2t(move(c));
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
