/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "ival-solver.hh"
#include "dump-smt2.hh"

#include <kay/dbl-ival.hh>

#include <iostream>

using namespace smlp;
namespace dbl = kay::dbl;

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

	friend str to_string(const res &r)
	{
		switch (r.v) {
		case YES: return "YES";
		case NO: return "NO";
		case MAYBE: return "MAYBE";
		}
		unreachable();
	}

	friend std::ostream & operator<<(std::ostream &s, const res &r)
	{
		return s << to_string(r);
	}
} YES = { res::YES }
, NO = { res::NO }
, MAYBE = { res::MAYBE };
}

static dbl::ival eval(const hmap<str,dbl::ival> &dom, const sptr<term2> &t, hmap<void *,dbl::ival> &m);
static res eval(const hmap<str,dbl::ival> &dom, const form2 &t, hmap<void *,dbl::ival> &m);

static dbl::ival eval(const hmap<str,dbl::ival> &dom, const term2 &t, hmap<void *,dbl::ival> &m)
{
	return t.match(
	[](const cnst2 &c) {
		return c.value.match(
		[](const auto &v) { return dbl::ival(v); }
		);
	},
	[&](const name &n) {
		auto it = dom.find(n.id);
		assert(it != dom.end());
		return it->second;
	},
	[&](const uop2 &u) {
		dbl::ival i = eval(dom, u.operand, m);
		switch (u.op) {
		case uop::UADD: break;
		case uop::USUB: neg(i); break;
		}
		return i;
	},
	[&](const bop2 &b) {
		dbl::ival l = eval(dom, b.left, m);
		if (b.op == bop::MUL && *b.left == *b.right)
			return square(l);
		dbl::ival r = eval(dom, b.right, m);
		switch (b.op) {
		case bop::ADD: l += r; break;
		case bop::SUB: l -= r; break;
		case bop::MUL: l *= r; break;
		}
		return l;
	},
	[&](const ite2 &i) {
		switch (eval(dom, *i.cond, m).v) {
		case res::YES: return eval(dom, i.yes, m);
		case res::NO: return eval(dom, i.no, m);
		case res::MAYBE: return convex_hull(eval(dom, i.yes, m), eval(dom, i.no, m));
		}
		unreachable();
	}
	);
}

static res eval(const hmap<str,dbl::ival> &dom, const form2 &f, hmap<void *,dbl::ival> &m)
{
	return f.match(
	[&](const prop2 &p) {
		dbl::ival v = eval(dom, bop2 { bop::SUB, p.left, p.right }, m);
		// std::cerr << "eval: " << v;
		res r;
		switch (p.cmp) {
		case LT: r = hi(v) < 0 ? YES : lo(v) >= 0 ? NO : MAYBE; break;
		case LE: r = hi(v) <= 0 ? YES : lo(v) > 0 ? NO : MAYBE; break;
		case GT: r = lo(v) > 0 ? YES : hi(v) <= 0 ? NO : MAYBE; break;
		case GE: r = lo(v) >= 0 ? YES : hi(v) < 0 ? NO : MAYBE; break;
		case EQ: r = sgn(v) == dbl::ZERO ? YES : sgn(v) == dbl::OV_ZERO ? MAYBE : NO; break;
		case NE: r = sgn(v) == dbl::ZERO ? NO : sgn(v) == dbl::OV_ZERO ? MAYBE : YES; break;
		default: abort();
		}
		// std::cerr << " -> " << r << "\n";
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

static dbl::ival eval(const hmap<str,dbl::ival> &dom, const sptr<term2> &t, hmap<void *,dbl::ival> &m)
{
	auto it = m.find(t.get());
	if (it == m.end())
		it = m.emplace(t.get(), eval(dom, *t, m)).first;
	return it->second;
}

template <typename T, typename F>
static void forall_products(const vec<pair<str,vec<T>>> &p,
                            hmap<str,T> &q, F &&f, size_t i=0)
{
	assert(i <= size(p));
	if (i < size(p)) {
		const auto &[var,l] = p[i];
		for (const T &r : l) {
			q[var] = r;
			forall_products(p, q, f, i+1);
		}
		q.erase(var);
	} else
		f(q);
}

static dbl::ival to_ival(const kay::Q &q)
{
	if (q.get_den() == 1)
		return dbl::ival(q.get_num());
	return dbl::ival(q);
}

opt<pair<double,double>>
smlp::dbl_interval_eval(const domain &dom, const sptr<term2> &t)
{
	/* need directed rounding downward for dbl::ival */
	dbl::rounding_mode rnd(FE_DOWNWARD);

	/* Replace the domain with intervals, collect discrete vars in d */
	hmap<str,dbl::ival> c;
	vec<pair<str,vec<dbl::ival>>> d;
	for (const auto &[var,k] : dom)
		k.range.match(
		[&,var=var](const entire &) {
			c.emplace(var, dbl::endpts { -INFINITY, INFINITY });
		},
		[&,var=var](const list &l) {
			vec<dbl::ival> ivs;
			for (const kay::Q &v : l.values)
				ivs.emplace_back(to_ival(v));
			d.emplace_back(var, move(ivs));
		},
		[&,var=var](const ival &i) {
			c.emplace(var, dbl::endpts {
				lo(to_ival(i.lo)),
				hi(to_ival(i.hi)),
			});
		}
		);

	opt<dbl::ival> r;
	forall_products(d, c, [&r,&t](const hmap<str,dbl::ival> &dom) {
		hmap<void *,dbl::ival> m;
		dbl::ival s = eval(dom, t, m);
		r = r ? convex_hull(*r, s) : s;
	});
	if (r)
		return pair { lo(*r), hi(*r) };
	return {};
}

static res eval(const hmap<str,dbl::ival> &dom, const form2 &f)
{
	hmap<void *,dbl::ival> m;
	return eval(dom, f, m);
}

static res eval_products(const vec<pair<str,vec<dbl::ival>>> &p,
                         hmap<str,dbl::ival> &q,
                         opt<hmap<str,dbl::ival>> &sat_model,
                         vec<hmap<str,dbl::ival>> &maybes,
                         vec<hmap<str,dbl::ival>> &nos,
                         const form2 &conj)
{
	res r = NO;
	forall_products(p, q, [&](const hmap<str,dbl::ival> &dom) {
		if (r == YES)
			return;
		res s = eval(dom, conj);
		switch (s.v) {
		case res::YES:
			if (!sat_model)
				sat_model = dom;
			break;
		case res::NO: nos.push_back(dom); break;
		case res::MAYBE: maybes.push_back(dom); break;
		}
		r |= s;
	});
	return r;
}

static vec<dbl::ival> split_ival(const dbl::ival &v)
{
	double m = mid(v);
	dbl::ival a = dbl::endpts { lo(v), m };
	vec<dbl::ival> r = { a };
	double bl = nextafter(m, INFINITY);
	if (bl <= hi(v))
		r.emplace_back(dbl::endpts { bl, hi(v) });
	return r;
}

static sptr<form2> in_domain(const hmap<str,dbl::ival> &dom)
{
	vec<sptr<form2>> c;
	for (const auto &[var,k] : dom) {
		sptr<term2> v = make2t(name { var });
		c.emplace_back(make2f(prop2 { GE, v, make2t(cnst2 { kay::Q(lo(k)) }) }));
		c.emplace_back(make2f(prop2 { LE, v, make2t(cnst2 { kay::Q(hi(k)) }) }));
	}
	return conj(move(c));
}

static result check_critical_points(const domain &dom, const sptr<form2> &orig)
{
	/* Check whether domain is bounded and if so, generate the list of its
	 * corner points */
	bool bounded_domain = true;
	vec<pair<str,vec<sptr<term2>>>> corners;
	size_t n_corners = 1;
	for (const auto &[var,k] : dom) {
		if (k.range.get<entire>()) {
			bounded_domain = false;
			break;
		}
		vec<sptr<term2>> values;
		if (const list *l = k.range.get<list>()) {
			for (const kay::Q &v : l->values)
				values.emplace_back(make2t(cnst2 { v }));
		} else {
			const ival *i = k.range.get<ival>();
			assert(i);
			values.emplace_back(make2t(cnst2 { i->lo }));
			values.emplace_back(make2t(cnst2 { i->hi }));
		}
		n_corners *= size(values);
		corners.emplace_back(var, move(values));
	}
	fprintf(stderr, "bounded domain: %d, #corners: %zu\n",
	        bounded_domain, n_corners);
	if (!bounded_domain)
		return unknown { "unbounded domain" };

	/* Check whether the partial derivatives are defined everywhere and if
	 * so, produce a formula stating that all should be equal to zero */
	struct {
		const domain &dom;
		lbop2 grad_eq_0 = { lbop2::AND, {} };
		bool deriv_exists = true;

		void operator()(const sptr<form2> &f)
		{
			if (!deriv_exists)
				return;
			f->match(
			[&](const prop2 &p) {
				sptr<term2> diff = make2t(bop2 {
					bop::SUB,
					p.left,
					p.right,
				});
				for (const auto &[var,_] : dom) {
					sptr<term2> d = derivative(diff, var);
					if (!d) {
						deriv_exists = false;
						return;
					}
					grad_eq_0.args.push_back(make2f(prop2 {
						EQ,
						d,
						zero
					}));
				}
			},
			[this](const lbop2 &l) {
				for (const sptr<form2> &g : l.args)
					(*this)(g);
			},
			[this](const lneg2 &n) {
				(*this)(n.arg);
			}
			);
		}
	} check { dom, };
	check(orig);
	fprintf(stderr, "derivatives exist: %d\n", check.deriv_exists);
	if (!check.deriv_exists)
		return unknown { "derivative not defined everywhere" };

	/* find all critical points of all functions in the problem */
	sptr<form2> f = make2f(move(check.grad_eq_0));
	/* restrict domain to only used variables, required for
	 * all_solutions() to terminate */
	domain sdom;
	hset<str> vars = free_vars(f);
	vec<pair<str,sptr<term2>>> remaining_vars;
	for (const auto &[var,k] : dom)
		if (vars.contains(var))
			sdom.emplace_back(var, k);
		else {
			/* pick any value in the domain */
			remaining_vars.emplace_back(var, k.range.match(
			[](const entire &) { return zero; },
			[](const list &l) { return make2t(cnst2 { l.values.front() }); },
			[](const ival &i) { return make2t(cnst2 { mid(i) }); }
			));
		}

	opt<hmap<str,sptr<term2>>> sat_model;
	auto eval = [&](const hmap<str,sptr<term2>> &dom) {
		if (sat_model)
			return;
		sptr<form2> f = cnst_fold(orig, dom);/*
		fprintf(stderr, "crit eval: ");
		dump_smt2(stderr, *f);
		fprintf(stderr, "\n");*/
		assert(*f == *true2 || *f == *false2);
		if (*f == *true2)
			sat_model = dom;
	};

	timing t0;
	size_t n = 0;
	for (hmap<str,sptr<term2>> crit : all_solutions(sdom, f)) {
		crit.insert(begin(remaining_vars), end(remaining_vars));
		eval(crit);
		n++;
	}
	timing t1;
	fprintf(stderr, "eval on %zu critical points: %d in %.3fs\n",
	        n, sat_model ? true : false, (double)(t1 - t0));
	hmap<str,sptr<term2>> empty;
	forall_products(corners, empty, eval);
	timing t2;
	fprintf(stderr, "monotonicity result: %d in %.3fs\n",
	        sat_model ? true : false, (double)(t2 - t1));
	if (sat_model)
		return sat { move(*sat_model) };
	return unsat {};
}

result ival_solver::check()
{
	opt<hmap<str,dbl::ival>> sat_model;
	vec<hmap<str,dbl::ival>> maybes, nos;
	res r;

	{
	/* need directed rounding downward for dbl::ival */
	dbl::rounding_mode rnd(FE_DOWNWARD);

	/* Replace the domain with intervals, collect discrete vars in d */
	hmap<str,dbl::ival> c;
	vec<pair<str,vec<dbl::ival>>> d;
	for (const auto &[var,k] : dom)
		k.range.match(
		[&,var=var](const entire &) {
			c.emplace(var, dbl::endpts { -INFINITY, INFINITY });
		},
		[&,var=var](const list &l) {
			vec<dbl::ival> ivs;
			for (const kay::Q &v : l.values)
				ivs.emplace_back(to_ival(v));
			d.emplace_back(var, move(ivs));
		},
		[&,var=var](const ival &i) {
			c.emplace(var, dbl::endpts {
				lo(to_ival(i.lo)),
				hi(to_ival(i.hi)),
			});
		}
		);

	/* For any combination of assignments to discrete vars interval-evaluate
	 * the formula. It is SAT if there is (at least) one combination that
	 * makes it evaluate to YES and otherwise UNKNOWN if there is (at least)
	 * one combination that makes it MAYBE and all others evaluate to NO. */
	r = eval_products(d, c, sat_model, maybes, nos, conj);
	fprintf(stderr, "lvl -1 it +%zun%zu\n",
	        size(maybes), size(nos));

	for (size_t i=0, j; r == MAYBE && i < max_subdivs; i++) {
		vec<hmap<str,dbl::ival>> maybes2;
		r = NO;
		j = 0;
		vec<pair<str,vec<dbl::ival>>> sp;
		timing t0;
		kay::Z N = 0;
		for (const hmap<str,dbl::ival> &dom : maybes) {
			if (r == YES)
				break;
			/* single sub-division of all domain elements */
			kay::Z n = 1;
			sp.clear();
			for (const auto &[var,v] : dom) {
				sp.emplace_back(var, split_ival(v));
				n *= size(sp.back().second);
			}
			N += n;
			fprintf(stderr, "lvl %zu it %zu/%zu+%zun%zu: checking %s subdivisions...",
			        i, j++, size(maybes), size(maybes2), size(nos),
			        n.get_str().c_str());
			fflush(stderr);
			hmap<str,dbl::ival> ndom;
			size_t old_m = size(maybes2);
			size_t old_n = size(nos);
			res s = eval_products(sp, ndom, sat_model, maybes2, nos, conj);
			if (s == NO) {
				nos.erase(begin(nos) + old_n, end(nos));
				nos.push_back(dom);
			}
			std::cerr << " -> " << s;
			if (s == MAYBE)
				std::cerr << " * " << (size(maybes2) - old_m);
			std::cerr << "\n";
			r |= s;
		}
		timing t1;
		fprintf(stderr, "checked %s subdivisions in %.3gs\n",
		        N.get_str().c_str(), (double)(t1 - t0));
		maybes = move(maybes2);
	}
	}

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

	sptr<form2> orig = to_nnf(simplify(make2f(conj)));
	if (result r = check_critical_points(dom, orig); !r.get<unknown>())
		return r;

	const char *the_logic = logic ? logic->c_str() : nullptr;
#if 0
	for (const auto &reg : maybes) {
		uptr<solver> s = mk_solver0(false, the_logic);
		s->declare(dom);
		s->add(orig);
		s->add(in_domain(reg));
		result r = s->check();
		if (!r.get<unsat>())
			return r;
	}
	return unsat {};
#else
	uptr<solver> s = mk_solver0(false, the_logic);
	s->declare(dom);
	s->add(orig);
	if (1) {
		vec<sptr<form2>> d;
		for (const auto &dom : maybes)
			d.emplace_back(in_domain(dom));
		s->add(disj(move(d)));
	}
	if (0)
		for (const auto &dom : nos)
			s->add(neg(in_domain(dom)));
	return s->check();
#endif
	/* some values do, others do not satisfy the formula */
	return unknown { "overlap" };
}
