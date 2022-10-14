/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "expr2.hh"

using namespace smlp;

bool prop2::operator==(const prop2 &b) const
{
	return cmp == b.cmp &&
	       (left == b.left || *left == *b.left) &&
	       (right == b.right || *right == *b.right);
}

bool lbop2::operator==(const lbop2 &b) const
{
	if (op != b.op)
		return false;
	if (args == b.args)
		return true;
	if (args.size() != b.args.size())
		return false;
	for (size_t i=0; i<args.size(); i++)
		if (*args[i] != *b.args[i])
			return false;
	return true;
}

bool lneg2::operator==(const lneg2 &b) const
{
	return arg == b.arg || *arg == *b.arg;
}

bool name::operator==(const name &b) const
{
	return id == b.id;
}

bool ite2::operator==(const ite2 &b) const
{
	return (cond == b.cond || *cond == *b.cond) &&
	       (yes == b.yes || *yes == *b.yes) &&
	       (no == b.no || *no == *b.no);
}

bool bop2::operator==(const bop2 &b) const
{
	return op == b.op &&
	       (left == b.left || *left == *b.left) &&
	       (right == b.right || *right == *b.right);
}

bool uop2::operator==(const uop2 &b) const
{
	return op == b.op && (operand == b.operand || *operand == *b.operand);
}

bool cnst2::operator==(const cnst2 &b) const
{
	return value == b.value;
}

sptr<expr2>
smlp::unroll(const expr &e,
             const hmap<str,fun<sptr<expr2>(vec<sptr<expr2>>)>> &funs)
{
	return e.match<sptr<expr2>>(
	[&](const name &n) { return make2e(n); },
	[&](const cnst &c) -> sptr<expr2> {
		if (c.value == "None")
			return nullptr;
		if (c.value.find('.') == str::npos &&
		    c.value.find('e') == str::npos &&
		    c.value.find('E') == str::npos)
			return make2e(cnst2 { kay::Z(c.value) });
		return make2e(cnst2 { kay::Q_from_str(str(c.value).data()) });
	},
	[&](const bop &b) {
		return make2e(bop2 {
			b.op,
			unroll(*b.left, funs),
			unroll(*b.right, funs),
		});
	},
	[&](const uop &u) {
		return make2e(uop2 {
			u.op,
			unroll(*u.operand, funs),
		});
	},
	[&](const call &c) {
		vec<sptr<expr2>> args;
		args.reserve(c.args.size());
		for (const expr &e : c.args)
			args.push_back(unroll(e, funs));
		auto f = funs.find(c.func->get<name>()->id);
		assert(f != funs.end());
		return f->second(move(args));
	}
	);
}

sptr<form2> smlp::subst(const sptr<form2> &f, const hmap<str,sptr<expr2>> &repl)
{
	return f->match(
	[&](const prop2 &p){
		sptr<expr2> l = subst(p.left, repl);
		sptr<expr2> r = subst(p.right, repl);
		return l == p.left && r == p.right
		     ? f : make2f(prop2 { p.cmp, move(l), move(r) });
	},
	[&](const lbop2 &b){
		vec<sptr<form2>> a = b.args;
		bool changed = false;
		for (sptr<form2> &o : a) {
			sptr<form2> q = subst(o, repl);
			changed |= o == q;
			o = move(q);
		}
		return !changed ? f : make2f(lbop2 { b.op, move(a) });
	},
	[&](const lneg2 &n){
		sptr<form2> m = subst(n.arg, repl);
		return m == n.arg ? f : make2f(lneg2 { move(m) });
	}
	);
}

sptr<expr2> smlp::subst(const sptr<expr2> &e, const hmap<str,sptr<expr2>> &repl)
{
	return e->match(
	[&](const name &n) {
		auto it = repl.find(n.id);
		return it == repl.end() ? e : it->second;
	},
	[&](const bop2 &b) {
		sptr<expr2> l = subst(b.left, repl);
		sptr<expr2> r = subst(b.right, repl);
		return l == b.left && r == b.right
		     ? e : make2e(bop2 { b.op, move(l), move(r) });
	},
	[&](const uop2 &u) {
		sptr<expr2> a = subst(u.operand, repl);
		return a == u.operand ? e : make2e(uop2 { u.op, move(a) });
	},
	[&](const cnst2 &) { return e; },
	[&](const ite2 &i) {
		sptr<form2> c = subst(i.cond, repl);
		sptr<expr2> y = subst(i.yes, repl);
		sptr<expr2> n = subst(i.no, repl);
		return c == i.cond && y == i.yes && n == i.no
		     ? e : make2e(ite2 { move(c), move(y), move(n) });
	}
	);
}

bool smlp::is_ground(const sptr<form2> &f)
{
	return f->match(
	[](const prop2 &p) { return is_ground(p.left) && is_ground(p.right); },
	[](const lbop2 &b) {
		for (const auto &o : b.args)
			if (!is_ground(o))
				return false;
		return true;
	},
	[](const lneg2 &n) { return is_ground(n.arg); }
	);
}

bool smlp::is_ground(const sptr<expr2> &e)
{
	return e->match(
	[](const name &) { return false; },
	[](const bop2 &b) { return is_ground(b.left) && is_ground(b.right); },
	[](const uop2 &a) { return is_ground(a.operand); },
	[](const cnst2 &) { return true; },
	[](const ite2 &i) {
		return is_ground(i.cond) && is_ground(i.yes) && is_ground(i.no);
	}
	);
}

sptr<form2> smlp::cnst_fold(const sptr<form2> &f, const hmap<str,sptr<expr2>> &repl)
{
	return f->match(
	[&](const prop2 &p) {
		sptr<expr2> l = cnst_fold(p.left, repl);
		sptr<expr2> r = cnst_fold(p.right, repl);
		const cnst2 *lc = l->get<cnst2>();
		const cnst2 *rc = r->get<cnst2>();
		if (!lc || !rc)
			return l == p.left && r == p.right ? f
			     : make2f(prop2 { p.cmp, move(l), move(r) });
		bool v = do_cmp(to_Q(lc->value), p.cmp, to_Q(rc->value));
		return v ? true2 : false2;
	},
	[&](const lbop2 &b) {
		vec<sptr<form2>> args;
		for (const sptr<form2> &a : b.args) {
			sptr<form2> f = cnst_fold(a, repl);
			if (*f == *true2) {
				if (b.op == lbop2::OR)
					return true2;
				continue;
			}
			if (*f == *false2) {
				if (b.op == lbop2::AND)
					return false2;
				continue;
			}
			args.emplace_back(move(f));
		}
		return args == b.args ? f : make2f(lbop2 { b.op, move(args) });
	},
	[&](const lneg2 &n) {
		sptr<form2> o = cnst_fold(n.arg, repl);
		if (*o == *true2)
			return false2;
		if (*o == *false2)
			return true2;
		return o == n.arg ? f : make2f(lneg2 { move(o) });
	}
	);
}

sptr<expr2> smlp::cnst_fold(const sptr<expr2> &e, const hmap<str,sptr<expr2>> &repl)
{
	return e->match(
	[&](const name &n) {
		auto it = repl.find(n.id);
		return it == repl.end() ? e : cnst_fold(it->second, repl);
	},
	[&](const cnst2 &) { return e; },
	[&](const bop2 &b) {
		sptr<expr2> l = cnst_fold(b.left, repl);
		sptr<expr2> r = cnst_fold(b.right, repl);
		const cnst2 *lc = l->get<cnst2>();
		const cnst2 *rc = r->get<cnst2>();
		if (!lc || !rc) {
			if (l == b.left && r == b.right)
				return e;
			return make2e(bop2 { b.op, move(l), move(r) });
		}
		kay::Q q;
		switch (b.op) {
		case bop::ADD: q = to_Q(lc->value) + to_Q(rc->value); break;
		case bop::SUB: q = to_Q(lc->value) - to_Q(rc->value); break;
		case bop::MUL: q = to_Q(lc->value) * to_Q(rc->value); break;
		}
		return make2e(cnst2 { move(q) });
	},
	[&](const uop2 &u) {
		sptr<expr2> o = cnst_fold(u.operand, repl);
		const cnst2 *c = o->get<cnst2>();
		if (!c)
			return o == u.operand ? e : make2e(uop2 { u.op, move(o) });
		kay::Q q;
		switch (u.op) {
		case uop::UADD: q = +to_Q(c->value); break;
		case uop::USUB: q = -to_Q(c->value); break;
		}
		return make2e(cnst2 { move(q) });
	},
	[&](const ite2 &i) {
		sptr<form2> c = cnst_fold(i.cond, repl);
		sptr<expr2> y = cnst_fold(i.yes, repl);
		sptr<expr2> n = cnst_fold(i.no, repl);
		if (*c == *true2)
			return y;
		if (*c == *false2)
			return n;
		if (c == i.cond && y == i.yes && n == i.no)
			return e;
		return make2e(ite2 { move(c), move(y), move(n) });
	}
	);
}

static bool is_nonlinear(const sptr<form2> &f)
{
	return f->match(
	[](const prop2 &p) { return is_nonlinear(p.left) || is_nonlinear(p.right); },
	[](const lbop2 &b) {
		for (const sptr<form2> &f : b.args)
			if (is_nonlinear(f))
				return true;
		return false;
	},
	[](const lneg2 &n) { return is_nonlinear(n.arg); }
	);
}

bool smlp::is_nonlinear(const sptr<expr2> &e)
{
	return e->match(
	[](const name &) { return false; },
	[](const cnst2 &) { return false; },
	[](const bop2 &b) {
		switch (b.op) {
		case bop::ADD:
		case bop::SUB:
			return is_nonlinear(b.left) || is_nonlinear(b.right);
		case bop::MUL:
			if (is_nonlinear(b.left) || is_nonlinear(b.right))
				return true;
			return !(is_ground(b.left) || is_ground(b.right));
		}
		unreachable();
	},
	[](const uop2 &u) { return is_nonlinear(u.operand); },
	[](const ite2 &i) {
		return ::is_nonlinear(i.cond) ||
		       is_nonlinear(i.yes) ||
		       is_nonlinear(i.no);
	}
	);
}
