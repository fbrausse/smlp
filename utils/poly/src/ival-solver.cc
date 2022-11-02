
#include "ival-solver.hh"

#include <iv/ival.hh>

using namespace smlp;

using Ty = vec<iv::ival>; /* finite_union */

static iv::ival hull(const Ty &t)
{
	assert(!empty(t));
	iv::ival r = t.front();
	for (size_t i=1; i<size(t); i++)
		r = convex_hull(r, t[i]);
	return r;
}

static iv::ival eval(const hmap<str,iv::ival> &dom, const sptr<term2> &t, hmap<void *,iv::ival> &m);

static iv::ival eval(const hmap<str,iv::ival> &dom, const term2 &t, hmap<void *,iv::ival> &m)
{
	return t.match(
	[](const cnst2 &c) {
		return c.value.match(
		[](const auto &v) { return iv::ival(v); }
		);
	},
	[&](const name &n) {
		auto it = dom.find(n.id);
		assert(it != dom.end());
		return it->second;
	},
	[&](const uop2 &u) {
		iv::ival i = eval(dom, u.operand, m);
		switch (u.op) {
		case uop::UADD: break;
		case uop::USUB: neg(i); break;
		}
		return i;
	},
	[&](const bop2 &b) {
		iv::ival l = eval(dom, b.left, m);
		if (b.op == bop::MUL && *b.left == *b.right)
			return square(l);
		iv::ival r = eval(dom, b.right, m);
		switch (b.op) {
		case bop::ADD: l += r; break;
		case bop::SUB: l -= r; break;
		case bop::MUL: l *= r; break;
		}
		return l;
	},
	[](const ite2 &) -> iv::ival { abort(); }
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

	friend std::ostream & operator<<(std::ostream &s, const res &r)
	{
		switch (r.v) {
		case YES: s << "YES"; break;
		case NO: s << "NO"; break;
		case MAYBE: s << "MAYBE"; break;
		}
		return s;
	}
} YES = { res::YES }
, NO = { res::NO }
, MAYBE = { res::MAYBE };
}

static res eval(const hmap<str,iv::ival> &dom, const form2 &f, hmap<void *,iv::ival> &m)
{
	return f.match(
	[&](const prop2 &p) {
		iv::ival v = eval(dom, bop2 { bop::SUB, p.left, p.right }, m);
		std::cerr << "eval: " << v;
		res r;
		switch (p.cmp) {
		case LT: r = hi(v) < 0 ? YES : lo(v) >= 0 ? NO : MAYBE; break;
		case LE: r = hi(v) <= 0 ? YES : lo(v) > 0 ? NO : MAYBE; break;
		case GT: r = lo(v) > 0 ? YES : hi(v) <= 0 ? NO : MAYBE; break;
		case GE: r = lo(v) >= 0 ? YES : hi(v) < 0 ? NO : MAYBE; break;
		case EQ: r = sgn(v) == iv::ZERO ? YES : sgn(v) == iv::OV_ZERO ? MAYBE : NO; break;
		case NE: r = sgn(v) == iv::ZERO ? NO : sgn(v) == iv::OV_ZERO ? MAYBE : YES; break;
		default: abort();
		}
		std::cerr << " -> " << r << "\n";
		return r;
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

static iv::ival eval(const hmap<str,iv::ival> &dom, const sptr<term2> &t, hmap<void *,iv::ival> &m)
{
	auto it = m.find(t.get());
	if (it == m.end())
		it = m.emplace(t.get(), eval(dom, *t, m)).first;
	return it->second;
}

static res eval(const hmap<str,iv::ival> &dom, const form2 &f)
{
	hmap<void *,iv::ival> m;
	return eval(dom, f, m);
}

static iv::ival to_ival(const kay::Q &q)
{
	if (q.get_den() == 1)
		return iv::ival(q.get_num());
	return iv::ival(q);
}

template <typename F>
static void forall_products(const vec<pair<str,vec<iv::ival>>> &p,
                            hmap<str,iv::ival> &q, F &&f, size_t i=0)
{
	assert(i <= size(p));
	if (i < size(p)) {
		const auto &[var,l] = p[i];
		for (const iv::ival &r : l) {
			q[var] = r;
			forall_products(p, q, f, i+1);
		}
		q.erase(var);
	} else
		f(q);
}

static vec<iv::ival> split_ival(const iv::ival &v)
{
	double m = mid(v);
	iv::ival a = iv::endpts { lo(v), m };
	vec<iv::ival> r = { a };
	double bl = nextafter(m, INFINITY);
	if (bl <= hi(v))
		r.emplace_back(iv::endpts { bl, hi(v) });
	return r;
}

result ival_solver::check()
{
	/* need directed rounding downward for iv::ival */
	iv::rounding_mode rnd(FE_DOWNWARD);

	/* Replace the domain with intervals, collect discrete vars in d */
	hmap<str,iv::ival> c;
	vec<pair<str,vec<iv::ival>>> d;
	for (const auto &[var,k] : dom)
		k.range.match(
		[&,var=var](const entire &) {
			c.emplace(var, iv::endpts { -INFINITY, INFINITY });
		},
		[&,var=var](const list &l) {
			vec<iv::ival> ivs;
			for (const kay::Q &v : l.values)
				ivs.emplace_back(to_ival(v));
			d.emplace_back(var, move(ivs));
		},
		[&,var=var](const ival &i) {
			c.emplace(var, iv::endpts {
				lo(to_ival(i.lo)),
				hi(to_ival(i.hi)),
			});
		}
		);

	/* For any combination of assignments to discrete vars interval-evaluate
	 * the formula. It is SAT if there is (at least) one combination that
	 * makes it evaluate to YES and otherwise UNKNOWN if there is (at least)
	 * one combination that makes it MAYBE and all others evaluate to NO. */
	res r = NO;
	opt<hmap<str,iv::ival>> sat_model;
	forall_products(d, c, [&](const hmap<str,iv::ival> &dom) {
		if (r == YES)
			return;
		for (const auto &[var,_] : d) {
			assert(ispoint(dom.find(var)->second));
			fprintf(stderr, "%s:%2g ", var.c_str(),
			        lo(dom.find(var)->second));
		}
		res s = eval(dom, conj);
		if (s == MAYBE) {
			/* single sub-division of all domain elements */
			vec<pair<str,vec<iv::ival>>> sp;
			kay::Z n = 1;
			for (const auto &[var,v] : dom) {
				sp.emplace_back(var, split_ival(v));
				n *= size(sp.back().second);
			}
			fprintf(stderr, "checking %s subdivisions...\n", n.get_str().c_str());
			hmap<str,iv::ival> ndom;
			res t = NO;
			forall_products(sp, ndom, [&t,this](const hmap<str,iv::ival> &ndom) {
				t |= eval(ndom, conj);
			});
			std::cerr << "subdivisions produced " << t << "\n";
			s = t;
		}
		if (s == YES && !sat_model)
			sat_model = dom;
		r |= s;
	});

	if (r == YES) {
		/* any value from the intervals will do; they are non-empty by
		 * construction */
		hmap<str,sptr<term2>> model;
		for (const auto &[var,c] : *sat_model)
			model.emplace(var, make2t(cnst2 { kay::Q(mid(c)) }));
		return sat { move(model) };
	}
	if (r == NO) {
		/* no value from at least one interval satisfies the formula */
		return unsat {};
	}
	/* some values do, others do not satisfy the formula */
	return unknown { "overlap" };
}
