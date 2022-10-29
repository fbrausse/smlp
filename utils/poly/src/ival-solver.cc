
#include "ival-solver.hh"

#include <iv/ival.hh>

using namespace smlp;

using Ival = iv::ival;

using Ty = vec<iv::ival>; /* finite_union */

static iv::ival hull(const Ty &t)
{
	assert(!empty(t));
	iv::ival r = t.front();
	for (size_t i=1; i<size(t); i++)
		r = convex_hull(r, t[i]);
	return r;
}

static Ty eval(const hmap<str,Ty> &dom, const sptr<term2> &t, hmap<void *,Ty> &m);

static Ty eval(const hmap<str,Ty> &dom, const term2 &t, hmap<void *,Ty> &m)
{
	return t.match(
	[](const cnst2 &c) {
		return Ty { c.value.match(
		[](const auto &v) { return Ival(v); }
		) };
	},
	[&](const name &n) {
		auto it = dom.find(n.id);
		assert(it != dom.end());
		return it->second;
	},
	[&](const uop2 &u) {
		Ty r;
		for (iv::ival i : eval(dom, u.operand, m)) {
			switch (u.op) {
			case uop::UADD: break;
			case uop::USUB: neg(i); break;
			}
			r.push_back(i);
		}
		return r;
	},
	[&](const bop2 &b) {
		Ty ret;
		Ty l = eval(dom, b.left, m);
		Ty r = eval(dom, b.right, m);
		if (size(l) > size(r))
			for (const iv::ival &i : l) {
				Ty k;
				for (iv::ival j : r) {
					switch (b.op) {
					case bop::ADD: j += i; break;
					case bop::SUB: j = i - j; break;
					case bop::MUL: j *= i; break;
					}
					k.push_back(j);
				}
				ret.push_back(hull(k));
			}
		else
			for (const iv::ival &j : r) {
				Ty k;
				for (iv::ival i : l) {
					switch (b.op) {
					case bop::ADD: i += j; break;
					case bop::SUB: i -= j; break;
					case bop::MUL: i *= j; break;
					}
					k.push_back(i);
				}
				ret.push_back(hull(k));
			}
		return ret;
	},
	[](const ite2 &) -> Ty { abort(); }
	);
}

#include <iostream>

namespace {
static const struct res {

	enum { NO, YES, MAYBE } v;

	friend bool operator==(const res &a, const res &b)
	{
		return a.v == b.v;
	}

	friend res operator!(const res &a)
	{
		return { a.v == YES ? NO : a.v == NO ? YES : MAYBE };
	}

	friend res operator&&(const res &a, const res &b)
	{
		if (a.v == NO || b.v == NO)
			return { NO };
		if (a.v == YES && b.v == YES)
			return { YES };
		return { MAYBE };
	}

	friend res & operator&=(res &a, const res &b)
	{
		return a = a && b;
	}

	friend res operator||(const res &a, const res &b)
	{
		if (a.v == YES || b.v == YES)
			return { YES };
		if (a.v == NO && b.v == NO)
			return { NO };
		return { MAYBE };
	}

	friend res & operator|=(res &a, const res &b)
	{
		return a = a || b;
	}
} YES = { res::YES }
, NO = { res::NO }
, MAYBE = { res::MAYBE };
}

static res eval(const hmap<str,Ty> &dom, const form2 &f, hmap<void *,Ty> &m)
{
	return f.match(
	[&](const prop2 &p) {
		Ty r = eval(dom, bop2 { bop::SUB, p.left, p.right }, m);
		for (const iv::ival &i : r)
			std::cerr << "eval: " << i << "\n";
		iv::ival v = hull(r);
		std::cerr << "hull: " << v << "\n";
		switch (p.cmp) {
		case LT: return hi(v) < 0 ? YES : lo(v) >= 0 ? NO : MAYBE;
		case LE: return hi(v) <= 0 ? YES : lo(v) > 0 ? NO : MAYBE;
		case GT: return lo(v) > 0 ? YES : hi(v) <= 0 ? NO : MAYBE;
		case GE: return lo(v) >= 0 ? YES : hi(v) < 0 ? NO : MAYBE;
		case EQ: return sgn(v) == iv::ZERO ? YES : sgn(v) == iv::OV_ZERO ? MAYBE : NO;
		case NE: return sgn(v) == iv::ZERO ? NO : sgn(v) == iv::OV_ZERO ? MAYBE : YES;
		}
		unreachable();
	},
	[&](const lbop2 &b) {
		res r = b.op == lbop2::AND ? YES : NO;
		for (const sptr<form2> &a : b.args)
			switch (b.op) {
			case lbop2::AND: r &= eval(dom, *a, m); break;
			case lbop2::OR : r |= eval(dom, *a, m); break;
			}
		return r;
	},
	[&](const lneg2 &n) { return !eval(dom, *n.arg, m); }
	);
}

static Ty eval(const hmap<str,Ty> &dom, const sptr<term2> &t, hmap<void *,Ty> &m)
{
	auto it = m.find(t.get());
	if (it == m.end())
		it = m.emplace(t.get(), eval(dom, *t, m)).first;
	return it->second;
}

result ival_solver::check()
{
	iv::rounding_mode rnd(FE_DOWNWARD);
	hmap<str,Ty> d;
	for (const auto &[var,c] : dom) {
		Ty rng;
		c.range.match(
		[](const entire &) { abort(); },
		[&](const list &l) {
			for (const kay::Q &v : l.values)
				rng.emplace_back(v);
		},
		[&](const ival &i) {
			rng.emplace_back(iv::endpts {
				lo(iv::ival(i.lo)),
				hi(iv::ival(i.hi)),
			});
		}
		);
		d.emplace(var, move(rng));
	}
	hmap<void *,Ty> m;
	res r = YES;
	for (const form2 &f : asserts)
		r &= eval(d, f, m);
	if (r == YES) {
		hmap<str,sptr<term2>> model;
		for (const auto &[var,c] : dom)
			model.emplace(var, make2t(c.range.match(
				[](const entire &) { return cnst2 { kay::Z(0) }; },
				[](const list &l) { return cnst2 { l.values.front() }; },
				[](const ival &i) { return cnst2 { mid(i) }; }
			)));
		return sat { move(model) };
	}
	if (r == NO)
		return unsat {};
	return unknown { "overlap" };
}
