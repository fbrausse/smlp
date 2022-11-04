/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "expr2.hh"

using namespace smlp;

sptr<term2> smlp::abs(const sptr<term2> &e)
{
	return make2t(ite2 {
		make2f(prop2 { LT, e, zero }),
		make2t(uop2 { uop::USUB, e }),
		e
	});
}

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

expr2s smlp::unroll(const expr &e,
                    const hmap<str,fun<expr2s(vec<expr2s>)>> &funs)
{
	return e.match<expr2s>(
	[&](const name &n) { return make2t(n); },
	[&](const cnst &c) -> sptr<term2> {
		if (c.value == "None")
			return nullptr;
		if (c.value.find('.') == str::npos &&
		    c.value.find('e') == str::npos &&
		    c.value.find('E') == str::npos)
			return make2t(cnst2 { kay::Z(c.value.c_str()) });
		return make2t(cnst2 { kay::Q_from_str(str(c.value).data()) });
	},
	[&](const bop &b) {
		return make2t(bop2 {
			b.op,
			*unroll(*b.left, funs).get<sptr<term2>>(),
			*unroll(*b.right, funs).get<sptr<term2>>(),
		});
	},
	[&](const uop &u) {
		return make2t(uop2 {
			u.op,
			*unroll(*u.operand, funs).get<sptr<term2>>(),
		});
	},
	[&](const cop &c) -> sptr<form2> {
		return make2f(prop2 {
			c.cmp,
			*unroll(*c.left, funs).get<sptr<term2>>(),
			*unroll(*c.right, funs).get<sptr<term2>>(),
		});
	},
	[&](const call &c) {
		vec<expr2s> args;
		args.reserve(c.args.size());
		for (const expr &e : c.args)
			args.push_back(unroll(e, funs));
		const str &s = c.func->get<name>()->id;
		auto f = funs.find(s);
		if (f == funs.end())
			DIE(1,"error: function '%s' called but not defined\n",
			    s.c_str());
		return f->second(move(args));
	}
	);
}

sptr<form2> smlp::subst(const sptr<form2> &f, const hmap<str,sptr<term2>> &repl)
{
	return f->match(
	[&](const prop2 &p){
		sptr<term2> l = subst(p.left, repl);
		sptr<term2> r = subst(p.right, repl);
		return l == p.left && r == p.right
		     ? f : make2f(prop2 { p.cmp, move(l), move(r) });
	},
	[&](const lbop2 &b){
		vec<sptr<form2>> a = b.args;
		for (sptr<form2> &o : a)
			o = subst(o, repl);
		return a == b.args ? f : make2f(lbop2 { b.op, move(a) });
	},
	[&](const lneg2 &n){
		sptr<form2> m = subst(n.arg, repl);
		return m == n.arg ? f : make2f(lneg2 { move(m) });
	}
	);
}

sptr<term2> smlp::subst(const sptr<term2> &e, const hmap<str,sptr<term2>> &repl)
{
	return e->match(
	[&](const name &n) {
		auto it = repl.find(n.id);
		return it == repl.end() ? e : it->second;
	},
	[&](const bop2 &b) {
		sptr<term2> l = subst(b.left, repl);
		sptr<term2> r = subst(b.right, repl);
		return l == b.left && r == b.right
		     ? e : make2t(bop2 { b.op, move(l), move(r) });
	},
	[&](const uop2 &u) {
		sptr<term2> a = subst(u.operand, repl);
		return a == u.operand ? e : make2t(uop2 { u.op, move(a) });
	},
	[&](const cnst2 &) { return e; },
	[&](const ite2 &i) {
		sptr<form2> c = subst(i.cond, repl);
		sptr<term2> y = subst(i.yes, repl);
		sptr<term2> n = subst(i.no, repl);
		return c == i.cond && y == i.yes && n == i.no
		     ? e : make2t(ite2 { move(c), move(y), move(n) });
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

bool smlp::is_ground(const sptr<term2> &e)
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

sptr<form2> smlp::cnst_fold(const sptr<form2> &f, const hmap<str,sptr<term2>> &repl)
{
	return f->match(
	[&](const prop2 &p) {
		sptr<term2> l = cnst_fold(p.left, repl);
		sptr<term2> r = cnst_fold(p.right, repl);
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

sptr<term2> smlp::cnst_fold(const sptr<term2> &e, const hmap<str,sptr<term2>> &repl)
{
	return e->match(
	[&](const name &n) {
		auto it = repl.find(n.id);
		return it == repl.end() ? e : cnst_fold(it->second, repl);
	},
	[&](const cnst2 &) { return e; },
	[&](const bop2 &b) {
		sptr<term2> l = cnst_fold(b.left, repl);
		sptr<term2> r = cnst_fold(b.right, repl);
		const cnst2 *lc = l->get<cnst2>();
		const cnst2 *rc = r->get<cnst2>();
		if (!lc || !rc) {
			if (l == b.left && r == b.right)
				return e;
			return make2t(bop2 { b.op, move(l), move(r) });
		}
		kay::Q q;
		switch (b.op) {
		case bop::ADD: q = to_Q(lc->value) + to_Q(rc->value); break;
		case bop::SUB: q = to_Q(lc->value) - to_Q(rc->value); break;
		case bop::MUL: q = to_Q(lc->value) * to_Q(rc->value); break;
		}
		return make2t(cnst2 { move(q) });
	},
	[&](const uop2 &u) {
		sptr<term2> o = cnst_fold(u.operand, repl);
		const cnst2 *c = o->get<cnst2>();
		if (!c)
			return o == u.operand ? e : make2t(uop2 { u.op, move(o) });
		kay::Q q;
		switch (u.op) {
		case uop::UADD: q = +to_Q(c->value); break;
		case uop::USUB: q = -to_Q(c->value); break;
		}
		return make2t(cnst2 { move(q) });
	},
	[&](const ite2 &i) {
		sptr<form2> c = cnst_fold(i.cond, repl);
		sptr<term2> y = cnst_fold(i.yes, repl);
		sptr<term2> n = cnst_fold(i.no, repl);
		if (*c == *true2)
			return y;
		if (*c == *false2)
			return n;
		if (c == i.cond && y == i.yes && n == i.no)
			return e;
		return make2t(ite2 { move(c), move(y), move(n) });
	}
	);
}

bool smlp::is_nonlinear(const sptr<form2> &f)
{
	return f->match(
	[](const prop2 &p) {
		return is_nonlinear(p.left) || is_nonlinear(p.right);
	},
	[](const lbop2 &b) {
		for (const sptr<form2> &f : b.args)
			if (is_nonlinear(f))
				return true;
		return false;
	},
	[](const lneg2 &n) { return is_nonlinear(n.arg); }
	);
}

bool smlp::is_nonlinear(const sptr<term2> &e)
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

static void collect_free_vars(const sptr<term2> &t, hset<str> &s);

static void collect_free_vars(const sptr<form2> &f, hset<str> &s)
{
	f->match(
	[&](const prop2 &p) {
		collect_free_vars(p.left, s);
		collect_free_vars(p.right, s);
	},
	[&](const lbop2 &b) {
		for (const sptr<form2> &a : b.args)
			collect_free_vars(a, s);
	},
	[&](const lneg2 &n) { collect_free_vars(n.arg, s); }
	);
}

static void collect_free_vars(const sptr<term2> &t, hset<str> &s)
{
	t->match(
	[&](const name &n) { s.emplace(n.id); },
	[](const cnst2 &) {},
	[&](const bop2 &b) {
		collect_free_vars(b.left, s);
		collect_free_vars(b.right, s);
	},
	[&](const uop2 &u) { collect_free_vars(u.operand, s); },
	[&](const ite2 &i) {
		collect_free_vars(i.cond, s);
		collect_free_vars(i.yes, s);
		collect_free_vars(i.no, s);
	}
	);
}

hset<str> smlp::free_vars(const sptr<term2> &f)
{
	hset<str> s;
	collect_free_vars(f, s);
	return s;
}

hset<str> smlp::free_vars(const sptr<form2> &f)
{
	hset<str> s;
	collect_free_vars(f, s);
	return s;
}

sptr<term2> smlp::derivative(const sptr<term2> &t, const str &var)
{
	return t->match(
	[&](const name &n) { return n.id == var ? one : zero; },
	[&](const cnst2 &) { return zero; },
	[&](const bop2 &b) {
		switch (b.op) {
		case bop::ADD:
		case bop::SUB:
			return make2t(bop2 { b.op,
				derivative(b.left, var),
				derivative(b.right, var)
			});
		case bop::MUL:
			return make2t(bop2 { bop::ADD,
				make2t(bop2 { bop::MUL,
					derivative(b.left, var),
					b.right,
				}),
				make2t(bop2 { bop::MUL,
					b.left,
					derivative(b.right, var),
				}),
			});
		}
		unreachable();
	},
	[&](const uop2 &u) {
		return make2t(uop2 { u.op, derivative(u.operand, var) });
	},
	[&](const ite2 &) -> sptr<term2> { return nullptr; }
	);
}

template <typename T>
static sptr<T> simplify(const sptr<T> &t, hmap<void *,expr2s> &m)
{
	auto it = m.find(t.get());
	if (it == end(m))
		it = m.emplace(t.get(), t->match(
		[&](const name &) -> expr2s { return t; },
		[&](const cnst2 &c) -> expr2s {
			if (c.value.get<kay::Z>())
				return t;
			const kay::Q *q = c.value.get<kay::Q>();
			kay::Q r = *q;
			using namespace kay;
			canonicalize(r);
			if (r.get_den() == 1)
				return make2t(cnst2 { move(r.get_num()) });
			if (r.get_num() == q->get_num() &&
			    r.get_den() == q->get_den())
				return t;
			return make2t(cnst2 { move(r) });
		},
		[&](const bop2 &b) -> expr2s {
			sptr<term2> l = simplify(b.left, m);
			sptr<term2> r = simplify(b.right, m);
			switch (b.op) {
			case bop::ADD:
			case bop::SUB:
				if (*l == *zero)
					return r;
				if (*r == *zero)
					return l;
				if (*l == *r)
					return b.op == bop::SUB
					     ? zero
					     : make2t(bop2 { bop::MUL,
						make2t(cnst2 { kay::Z(2) }),
						l
					});
				break;
			case bop::MUL:
				if (*l == *zero)
					return zero;
				if (*r == *zero)
					return zero;
				if (*l == *one)
					return r;
				if (*r == *one)
					return l;
				break;
			}
			if (l == b.left && r == b.right)
				return t;
			return make2t(bop2 { b.op, move(l), move(r) });
		},
		[&](const uop2 &u) -> expr2s {
			sptr<term2> o = simplify(u.operand, m);
			if (*o == *zero)
				return zero;
			if (u.op == uop::UADD)
				return o;
			if (const cnst2 *c = o->get<cnst2>())
				return c->value.match(
				[](const auto &v) { return make2t(cnst2 { kay::Q(-v) }); }
				);
			if (o == u.operand)
				return t;
			return make2t(uop2 { u.op, move(o) });
		},
		[&](const ite2 &i) -> expr2s {
			sptr<form2> c = simplify(i.cond, m);
			sptr<term2> y = simplify(i.yes, m);
			sptr<term2> n = simplify(i.no, m);
			if (*c == *true2)
				return y;
			if (*c == *false2)
				return n;
			if (*y == *n)
				return y;
			return make2t(ite2 { move(c), move(y), move(n), });
		},
		[&](const prop2 &p) -> expr2s {
			sptr<term2> l = simplify(p.left, m);
			sptr<term2> r = simplify(p.right, m);
			if (const cnst2 *lc = l->get<cnst2>())
			if (const cnst2 *rc = r->get<cnst2>())
				return do_cmp(to_Q(lc->value), p.cmp,
				              to_Q(rc->value)) ? true2 : false2;
			if (l == p.left && r == p.right)
				return t;
			return make2f(prop2 { p.cmp, move(l), move(r) });
		},
		[&](const lbop2 &b) -> expr2s {
			vec<sptr<form2>> a;
			for (const sptr<form2> &f : b.args) {
				sptr<form2> g = simplify(f, m);
				if (*g == *true2) {
					switch (b.op) {
					case lbop2::AND: continue;
					case lbop2::OR: return true2;
					}
					unreachable();
				}
				if (*g == *false2) {
					switch (b.op) {
					case lbop2::AND: return false2;
					case lbop2::OR: continue;
					}
					unreachable();
				}
				a.emplace_back(move(g));
			}
			if (a == b.args)
				return t;
			return make2f(lbop2 { b.op, move(a) });
		},
		[&](const lneg2 &n) -> expr2s {
			sptr<form2> o = simplify(n.arg, m);
			if (*o == *true2)
				return false2;
			if (*o == *false2)
				return true2;
			if (o == n.arg)
				return t;
			return make2f(lneg2 { move(o) });
		}
		)).first;
	return *it->second.template get<sptr<T>>();
}

sptr<term2> smlp::simplify(const sptr<term2> &t)
{
	hmap<void *,expr2s> m;
	return ::simplify(t, m);
}

sptr<form2> smlp::simplify(const sptr<form2> &f)
{
	hmap<void *,expr2s> m;
	return ::simplify(f, m);
}
