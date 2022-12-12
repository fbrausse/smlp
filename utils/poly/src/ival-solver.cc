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
		[](const kay::Z &v) { return dbl::ival(v); },
		[](const kay::Q &v) { return dbl::ival(v); },
		[](const Ap &v) {
			return dbl::ival { dbl::endpts {
				lo(dbl::ival(lo(v))),
				hi(dbl::ival(hi(v))),
			} };
		});
	},
	[&](const name &n) {
		auto it = dom.find(n.id);
		assert(it != dom.end());
		return it->second;
	},
	[&](const uop2 &u) {
		dbl::ival i = eval(dom, u.operand, m);
		switch (u.op) {
		case uop2::UADD: break;
		case uop2::USUB: neg(i); break;
		}
		return i;
	},
	[&](const bop2 &b) {
		dbl::ival l = eval(dom, b.left, m);
		if (b.op == bop2::MUL && *b.left == *b.right)
			return square(l);
		dbl::ival r = eval(dom, b.right, m);
		switch (b.op) {
		case bop2::ADD: l += r; break;
		case bop2::SUB: l -= r; break;
		case bop2::MUL: l *= r; break;
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
		dbl::ival v = eval(dom, bop2 { bop2::SUB, p.left, p.right }, m);/*
		size_t hh = size(dom);
		for (const auto &[n,v] : dom)
			hh = (hh << 1) ^ std::hash<str>{}(n);
		std::cerr << "eval " << hh << cmp_s[p.cmp] << ": " << v;*/
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

static bool is_bounded(const hmap<str,dbl::ival> &dom)
{
	for (const auto &[n,v] : dom)
		if (!isbounded(v))
			return false;
	return true;
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
		res s = eval(dom, conj);/*
		std::cerr << "subdiv on " << (is_bounded(dom) ? "bounded" : "unbounded")
		          << " domain -> " << s << "\n";*/
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
	if (m < hi(v))
		r.emplace_back(dbl::endpts { m, hi(v) });
	return r;
}

static sptr<form2> in_domain(const hmap<str,dbl::ival> &dom)
{
	vec<sptr<form2>> c;
	for (const auto &[var,k] : dom) {
		sptr<term2> v = make2t(name { var });
		assert(!std::isnan(lo(k)));
		assert(!std::isnan(hi(k)));
		if (std::isfinite(lo(k)))
			c.emplace_back(make2f(prop2 { GE, v, make2t(cnst2 { kay::Q(lo(k)) }) }));
		if (std::isfinite(hi(k)))
			c.emplace_back(make2f(prop2 { LE, v, make2t(cnst2 { kay::Q(hi(k)) }) }));
	}
	return conj(move(c));
}

static bool contains_ite(const sptr<term2> &t)
{
	return t->match(
	[](const name &) { return false; },
	[](const cnst2 &) { return false; },
	[](const bop2 &b) { return contains_ite(b.left) || contains_ite(b.right); },
	[](const uop2 &u) { return contains_ite(u.operand); },
	[](const ite2 &) { return true; }
	);
}

result crit_solver::check(const domain &dom, const sptr<form2> &orig)
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
	note(mod_crit,"bounded domain: %d, #corners: %zu\n", bounded_domain, n_corners);
	if (!bounded_domain)
		return unknown { "unbounded domain" };

	/* Check whether the partial derivatives are defined everywhere and if
	 * so, produce a formula stating that all should be equal to zero */
	struct {
		const domain &dom;
		vec<sptr<form2>> grad_eq_0 = {};
		bool deriv_exists = true;
		bool only_order_props = true;
		bool known_continuous = true;
		bool nonlinear = false;

		void operator()(const sptr<form2> &f)
		{
			if (0 && (!deriv_exists || !only_order_props))
				return;
			f->match(
			[&](const prop2 &p) {
				only_order_props &= is_order(p.cmp);
				if (0 && !only_order_props)
					return;
				sptr<term2> diff = simplify(make2t(bop2 {
					bop2::SUB,
					p.left,
					p.right,
				}));
				dbg(mod_prob,"partial derivatives of: ") && (
					dump_smt2(stderr, *diff),
					fprintf(stderr, "\n"));
				known_continuous &= !contains_ite(diff);
				for (const auto &[var,_] : dom) {
					sptr<term2> d = derivative(diff, var);
					if (!d) {
						deriv_exists = false;
						return;
					}
					d = simplify(d);
					if (is_nonlinear(d)) {
						nonlinear = true;
						dbg(mod_prob,"partial derivate is non-linear: ") && (
							dump_smt2(stderr, *d),
							fprintf(stderr, "\n"));
					}
					grad_eq_0.push_back(make2f(prop2 {
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
	note(mod_crit,"derivatives exist: %d, only ordered comparisons: %d, "
	              "known_continuous: %d, nonlinear: %d\n", check.deriv_exists,
	              check.only_order_props, check.known_continuous, check.nonlinear);
	if (!check.deriv_exists)
		return unknown { "derivative may not be defined everywhere" };
	if (!check.only_order_props)
		return unknown { "critical points cannot solve (dis-)equality constraints" };
	if (check.nonlinear)
		return unknown { "cannot reason about critical points of functions with non-linear partial derivatives" };

	/* find all critical points of all functions in the problem */
	sptr<form2> f = conj(move(check.grad_eq_0));
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
/*
	fprintf(stderr, "partial derivatives equal zero: ");
	dump_smt2(stderr, *conj({ domain_constraints(sdom), f }), false);
	fprintf(stderr, "\n");
*/
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

	info(mod_crit,"solving on critical points and %zu domain corners...\n",
	     n_corners);

#if 0
	sptr<form2> tgt = disj({ neg(f), orig });
	uptr<solver> slv = mk_solver0(false, smt2_logic_str(dom, tgt).c_str());
	slv->declare(dom);
	slv->add(tgt);
	result r = slv->check();
	if (r.get<sat>()) {
		info(mod_crit,"critical -> target is SAT\n");
		return r;
	} else {
		info(mod_crit,"critical -> target is not SAT\n");
	}
#endif

	timing t0;
	size_t n = 0;
	for (hmap<str,sptr<term2>> crit : all_solutions(sdom, f)) {
		crit.insert(begin(remaining_vars), end(remaining_vars));
		if (note(mod_crit,"critical point:")) {
			for (const auto &[v,c] : crit)
				fprintf(stderr, " %s=%s", v.c_str(),
					to_string(c->get<cnst2>()->value).c_str());
			fprintf(stderr, "\n");
		}
		eval(crit);
		n++;
	}
	timing t1;
	note(mod_crit,"eval on %zu critical points: %d in %.3fs\n",
	              n, sat_model ? true : false, (double)(t1 - t0));
	hmap<str,sptr<term2>> empty;
	forall_products(corners, empty, eval);
	timing t2;
	note(mod_crit,"monotonicity result: %d in %.3fs\n",
	              sat_model ? true : false, (double)(t2 - t1));
	if (sat_model)
		return sat { move(*sat_model) };
	return unsat {};
}

result ival_solver::check() const
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
		[&,var=var,k=k](const ival &i) {
			switch (k.type) {
			case type::INT: {
				vec<dbl::ival> ivs;
				using namespace kay;
				for (kay::Z z = ceil(i.lo), u = floor(i.hi); z <= u; z++)
					ivs.emplace_back(z);
				d.emplace_back(var, move(ivs));
				break;
			}
			case type::REAL:
				c.emplace(var, dbl::endpts {
					lo(to_ival(i.lo)),
					hi(to_ival(i.hi)),
				});
				break;
			}
		}
		);

	info(mod_ival,"solving...\n");

	/* For any combination of assignments to discrete vars interval-evaluate
	 * the formula. It is SAT if there is (at least) one combination that
	 * makes it evaluate to YES and otherwise UNKNOWN if there is (at least)
	 * one combination that makes it MAYBE and all others evaluate to NO. */
	r = eval_products(d, c, sat_model, maybes, nos, asserts);
	note(mod_ival,"lvl -1 it +%zun%zu\n", size(maybes), size(nos));

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
			bool logs = note(mod_ival,
				"lvl %zu it %zu/%zu+%zun%zu: checking %s subdivisions...",
				i, j++, size(maybes), size(maybes2), size(nos),
				n.get_str().c_str());
			fflush(stderr);
			hmap<str,dbl::ival> ndom;
			size_t old_m = size(maybes2);
			size_t old_n = size(nos);
			res s = eval_products(sp, ndom, sat_model, maybes2, nos, asserts);
			if (s == NO) {
				nos.erase(begin(nos) + old_n, end(nos));
				nos.push_back(dom);
			}
			if (logs) {
				std::cerr << " -> " << s;
				if (s == MAYBE)
					std::cerr << " * " << (size(maybes2) - old_m);
				std::cerr << "\n";
			}
			r |= s;
		}
		timing t1;
		note(mod_ival,"checked %s subdivisions in %.3gs\n",
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

	sptr<form2> orig = to_nnf(simplify(make2f(asserts)));
#if 0
	if (result r = check_critical_points(dom, orig); !r.get<unknown>())
		return r;
#endif
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
#elif 0
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
