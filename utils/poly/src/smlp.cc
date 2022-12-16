/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "common.hh"
#include "expr.hh"
#include "infix.hh"
#include "expr2.hh"
#include "domain.hh"
#include "dump-smt2.hh"
#include "ext-solver.hh"
#include "ival-solver.hh"
#include "nn.hh"
#include "poly.hh"

#ifdef SMLP_ENABLE_Z3_API
# include "z3-solver.hh"
# include <z3_version.h>
#endif

#ifdef SMLP_ENABLE_KERAS_NN
# include <H5public.h>
# include <kjson.h>
#endif

#include <cmath>	/* isfinite() */
#include <filesystem>	/* ...::current_path() */
#include <set>

#include <signal.h>

using namespace smlp;
using std::isfinite;

namespace {

// using domain_ref = typename domain::const_iterator;

/* domain is bounded -> subdivison strategy applicable
 * domain is union -> ival/crit require subdivision strategy
 *
 * domain is product -> ival/crit is applicable
 *
 * let F be set of terms lhs-rhs for all prop2 in p.
 * F differentiable, F continuous, only ordered comparisons -> crit applicable
 * F continuous -> ivt applicable for == constraints in p
 * F continuous -> p delta-decidable for any delta > 0
 *
 * F contains (ite c y n) with free variables in c -> F not continuous and not differentiable
 *
 *
 *
 * I'm trying to generalize how smlp chooses which solver strategy to apply...
 * For now, I have the following properties that affect applicability of
 * at least one of the strategies "subdivision", "interval",
 * "critical points", "ivt" (intermediate value theorem, maybe impl later),
 * "smt", "bayesian optimization" (later):
 *
 * - domain is union (of products of subspaces of R)
 * - domain is product of unions (of subspaces of R)
 * - domain is product of intervals
 * - domain is bounded
 * - function is continuous
 * - function is differentiable
 * - function is rational
 * - function is polynomial
 * - predicate is order
 * - predicate is equality
 *
 * - all ite-s in function are Relu -> continuous
 */

struct classification {

	class {
		unsigned any_empty   : 1 = false;
		unsigned all_single  : 1 = true;
		unsigned any_union   : 1 = false;
		unsigned all_bounded : 1 = true;

	public:
		unsigned has_R       : 1 = false;
		unsigned has_Z       : 1 = false;

		void prod(const component &c)
		{
			switch (c.type) {
			case type::INT: has_Z = 1; break;
			case type::REAL: has_R = 1; break;
			}
			c.range.match(
			[&](const entire &) {
				all_single = false;
				all_bounded = false;
			},
			[&](const list &l) {
				any_empty |= size(l.values) == 0;
				all_single &= size(l.values) == 1;
				any_union |= size(l.values) > 1;
			},
			[&](const ival &i) {
				using namespace kay;
				switch (c.type) {
				case type::INT:
					any_empty |= ceil(i.lo) > floor(i.hi);
					all_single &= ceil(i.lo) == floor(i.hi);
					break;
				case type::REAL:
					all_single &= i.lo == i.hi;
					break;
				}
			});
		}

		friend bool is_empty(const auto &dom)
		{
			return dom.any_empty;
		}

		/* product of intervals otherwise */
		friend bool is_union(const auto &dom)
		{
			return dom.any_empty || dom.any_union;
		}

		friend bool is_point(const auto &dom)
		{
			return dom.all_single;
		}

		friend bool is_bounded(const auto &dom)
		{
			return dom.all_bounded;
		}
	} dom; /* properties of the unique topological space represented by
	          domain, atm. this is a product space */

	explicit classification(const domain &d)
	{
		for (const auto &[n,c] : d)
			dom.prod(c);
	}

	static constexpr unsigned
		FUN_IS_CONTINUOUS = 1 << 0,
		FUN_IS_DIFFERENTIABLE = 1 << 1 | FUN_IS_CONTINUOUS,
		FUN_IS_RATIONAL = 1 << 2,
		FUN_IS_POLYNOMIAL = 1 << 3 | FUN_IS_DIFFERENTIABLE | FUN_IS_RATIONAL,
		FUN_IS_PW_LINEAR = 1 << 4 | FUN_IS_RATIONAL,
		FUN_IS_CONSTANT = 1 << 5 | FUN_IS_POLYNOMIAL,
		FUN_IS_ABS_RELU = 1 << 6 | FUN_IS_PW_LINEAR | FUN_IS_CONTINUOUS;
	static constexpr unsigned
		PRED_IS_ORDER = 1 << 0,
		PRED_IS_EQUALITY = 2 << 0;

	void add(const sptr<form2> &f)
	{
		/* TODO */
	}

	unsigned all_fun : 7 = 0x7f; /* properties all functions share */
	// unsigned any_fun : 7 = 0; /* properties any function has */
	unsigned all_pred : 2 = 0x3; /* properties all predicates share */
	// unsigned any_pred : 2 = 0; /* properties any predicate has */
};

struct problem {

	domain dom;
	sptr<form2> p;
};

static classification classify(const problem &p)
{
	classification cl(p.dom);
	cl.add(p.p);
	return cl;
}

static bool enable_ival_on_unbounded_dom = false;

static uptr<solver> mk_solver1(const problem &p)
{
	classification cl = classify(p);
	if (is_union(cl.dom)) {
		/* apply algebraic subdivision strategy, not unbounded;
		 * this is in contrast to the potentially unbounded numeric
		 * subdivision on double-intervals the interval solver applies */
	} else {
		if (is_point(cl.dom)) {
			/* apply "eval" strategy */
		} else if (is_bounded(cl.dom) ||
		           enable_ival_on_unbounded_dom) {
			/* apply "ival" or "crit" strategy */
		} else {
			/* apply "smt" or "crit" */
		}
	}
	unreachable();
}

/* max T, s.t.
 * E x. eta x /\
 * A y. eta y -> theta x y -> alpha y -> beta y /\ obj y >= T
 */
struct maxprob {

	domain dom;
	term2  obj;
	sptr<form2> theta;
	sptr<form2> alpha = true2;
	sptr<form2> beta  = true2;
};
/*
struct pareto {

	domain dom;
	vec<term2> objs;
	sptr<form2> alpha = true2;
	sptr<form2> beta  = true2;
};
*/
}

static result solve_exists(const domain &dom,
                           const sptr<form2> &f,
                           const char *logic = nullptr)
{
	uptr<solver> s = mk_solver(false,
		logic ? logic : smt2_logic_str(dom, f).c_str());
	s->declare(dom);
	s->add(f);
	return s->check();
}

namespace {
enum class res { MAYBE = -1, NO, YES };

struct search_base {

	bool done;

	search_base(bool done) : done(done) {}
	virtual ~search_base() = default;
	virtual bool has_next() const = 0;
	virtual kay::Q query() const = 0;
	virtual void reply(int order) = 0;
	virtual const kay::Q * lo() const = 0;
	virtual const kay::Q * hi() const = 0;

	virtual uptr<search_base> clone() const = 0;
};

struct search_ival : search_base {

	ival v;

	explicit search_ival(ival v)
	: search_base { v.lo > v.hi }
	, v(move(v))
	{}

	bool has_next() const override
	{
		return !done && v.lo <= v.hi;
	}

	kay::Q query() const override { return mid(v); }

	void reply(int order) override
	{
		done |= !order || v.lo == v.hi;
		if (order >= 0)
			v.lo = mid(v);
		if (order <= 0)
			v.hi = mid(v);
	}

	const kay::Q * lo() const override { return &v.lo; }
	const kay::Q * hi() const override { return &v.hi; }

	uptr<search_base> clone() const override { return std::make_unique<search_ival>(*this); }
};

struct bounded_search_ival : search_ival {

	kay::Q prec;
	bool any;

	explicit bounded_search_ival(ival v, kay::Q prec)
	: search_ival { move(v) }
	, prec(move(prec))
	, any(false)
	{ assert(this->prec > 0); }

	bool has_next() const override
	{
		return !done && (!any || length(v) > prec);
	}

	void reply(int order) override
	{
		any = true;
		search_ival::reply(order);
		done |= length(v) <= prec;
	}

	uptr<search_base> clone() const override { return std::make_unique<bounded_search_ival>(*this); }
};

struct search_list : search_base {

	vec<kay::Q> values;
	ssize_t l, r, m;

	explicit search_list(vec<kay::Q> values)
	: search_base { empty(values) }
	, values(move(values))
	, l(0)
	, r(size(this->values) - 1)
	, m(l + (r - l) / 2)
	{
		assert(std::is_sorted(begin(this->values), end(this->values)));
	}

	bool has_next() const override { return !done; }
	kay::Q query() const override { return values[m]; }

	void reply(int order) override
	{
		m += order;
		if (order >= 0)
			l = m;
		if (order <= 0)
			r = m;
		done |= !order || r < l;
		m = l + (r - l) / 2;
	}

	const kay::Q * lo() const override
	{
		return empty(values) ? nullptr
		     : &values[std::clamp<ssize_t>(done ? r : l, 0, size(values)-1)];
	}

	const kay::Q * hi() const override
	{
		return empty(values) ? nullptr
		     : &values[std::clamp<ssize_t>(r, 0, size(values)-1)];
	}

	uptr<search_base> clone() const override { return std::make_unique<search_list>(*this); }
};

struct smlp_result_base {
	kay::Q threshold;
	hmap<str,sptr<term2>> point;

	kay::Q center_value(const sptr<term2> &obj) const
	{
		return to_Q(cnst_fold(obj, point)->get<cnst2>()->value);
	}
};

struct smlp_result : smlp_result_base {
	uptr<solver> slv;
	uptr<search_base> obj_range;

	smlp_result(kay::Q threshold, uptr<solver> slv, hmap<str,sptr<term2>> pt,
	            uptr<search_base> obj_range)
	: smlp_result_base { move(threshold), move(pt) }
	, slv(move(slv))
	, obj_range(move(obj_range))
	{}
};

}

static void trace_result(FILE *f, const char *lbl, const result &r,
                         const kay::Q &T, double t)
{
	const char *state = nullptr;
	vec<str> info;
	r.match(
	[&](const sat &s) {
		state = "sat";
		vec<const decltype(s.model)::value_type *> v;
		for (const auto &p : s.model)
			v.push_back(&p);
		sort(begin(v), end(v), [](const auto *a, const auto *b) {
			return a->first < b->first;
		});
		for (const auto *p : v) {
			info.emplace_back(p->first);
			info.emplace_back(to_string(p->second->get<cnst2>()->value));
		}
	},
	[&](const unsat &) { state = "unsat"; },
	[&](const unknown &u) {
		state = "unknown";
		info.emplace_back(u.reason);
	}
	);
	assert(state);
	fprintf(f, "%s,%s,%g,%5.3f", lbl, state, T.get_d(), t);
	for (const str &s : info)
		fprintf(f, ",%s", s.c_str());
	fprintf(f, "\n");
}

typedef fun<sptr<form2>(opt<kay::Q> delta, const hmap<str,sptr<term2>> &)> theta_t;

static vec<smlp_result>
optimize_EA(cmp_t direction,
            const domain &dom,
            const sptr<term2> &objective,
            const sptr<form2> &alpha,
            const sptr<form2> &beta,
            const sptr<form2> &eta,
            const kay::Q &delta,
            search_base &obj_range,
            const theta_t &theta,
            const char *logic = nullptr)
{
	assert(is_order(direction));

	/* optimize T in obj_range such that (assuming direction is >=):
	 *
	 * E x . eta x /\
	 * A y . theta x y -> alpha y -> (beta y /\ obj y >= T)
	 *
	 * domain constraints from 'dom' have to hold for x and y.
	 */
/*
	fprintf(stderr, "dom: ");
	dump_smt2(stderr, dom);
	fprintf(stderr, "alpha: ");
	dump_smt2(stderr, *alpha);
	fprintf(stderr, "\nbeta: ");
	dump_smt2(stderr, *beta);
	fprintf(stderr, "\n");
*/
	vec<smlp_result> results;

	while (obj_range.has_next()) {
		kay::Q T = obj_range.query();
		printf("r,%s,%s,%s\n",
		       obj_range.lo()->get_str().c_str(),
		       obj_range.hi()->get_str().c_str(),
		       T.get_str().c_str());
		sptr<term2> threshold = make2t(cnst2 { T });

		/* eta x /\ alpha x /\ (beta x /\ obj x >= T) */
		sptr<form2> target = conj({
			beta,
			make2f(prop2 { direction, objective, threshold })
		});
		uptr<solver> exists = mk_solver(true, logic);
		exists->declare(dom);
		exists->add(eta);
		exists->add(alpha);
		exists->add(target);

		for (vec<smlp_result> counter_examples;;) {
			note(mod_cand,"searching candidate %s T ~ %g...\n",
			     cmp_s[direction],T.get_d());
			timing e0;
			result e = exists->check();
			trace_result(stdout, "a", e, T, timing() - e0);
			if (unknown *u = e.get<unknown>())
				MDIE(mod_smlp,2,"exists is unknown: %s\n",
				     u->reason.c_str());
			if (e.get<unsat>()) {
				obj_range.reply(is_less(!direction) ? -1 : +1);
				break;
			}
			auto &candidate = e.get<sat>()->model;

			uptr<solver> forall = mk_solver(false, logic);
			forall->declare(dom);
			/* ! ( theta x y -> alpha y -> beta y /\ obj y >= T ) =
			 * ! ( ! theta x y \/ ! alpha y \/ beta y /\ obj y >= T ) =
			 * theta x y /\ alpha y /\ ! ( beta y /\ obj y >= T) */
			forall->add(theta({}, candidate));
			forall->add(alpha);
			forall->add(neg(target));
			/*
			file test("ce.smt2", "w");
			smlp::dump_smt2(test, dom);
			fprintf(test, "(assert ");
			smlp::dump_smt2(test, *theta(true, candidate));
			fprintf(test, ")\n");
			fprintf(test, "(assert ");
			smlp::dump_smt2(test, *alpha);
			fprintf(test, ")\n");
			fprintf(test, "(assert ");
			smlp::dump_smt2(test, lbop2 { lbop2::OR, {
				make2f(lneg2 { beta }),
				make2f(prop2 { ~direction, objective, threshold })
			} });
			fprintf(test, ")\n");
			*/

			note(mod_coex,"searching counterexample %s T ~ %g...\n",
			     cmp_s[!direction], T.get_d());
			timing a0;
			result a = forall->check();
			trace_result(stdout, "b", a, T, timing() - a0);
			if (unknown *u = a.get<unknown>())
				MDIE(mod_smlp,2,"forall is unknown: %s\n",
				     u->reason.c_str());
			if (a.get<unsat>()) {
				// fprintf(file("ce-z3.smt2", "w"), "%s\n", forall.slv.to_smt2().c_str());
				info(mod_prob,"found solution %s T ~ %g\n",
				     cmp_s[direction], T.get_d());
				results.emplace_back(T, move(exists), candidate,
				                     obj_range.clone());
				obj_range.reply(is_less(direction) ? -1 : +1);
				break;
			}
			auto &counter_example = a.get<sat>()->model;
			/* let's not keep the forall solver around */
			counter_examples.emplace_back(T, nullptr, counter_example, nullptr);
			exists->add(neg(theta({ delta }, counter_example)));
		}
	}
	auto Q_str = [](const kay::Q *l) { return l ? l->get_str() : ""; };
	printf("u,%s,%s,%d\n",
	       Q_str(obj_range.lo()).c_str(), Q_str(obj_range.hi()).c_str(),
	       obj_range.done);

	return results;
}

struct infty { bool pos; };

template <typename T>
struct extended : sumtype<T,infty> {
	using sumtype<T,infty>::sumtype;

	friend bool is_infty(const extended &e, bool *pos = nullptr)
	{
		if (const infty *i = e.template get<infty>()) {
			if (pos)
				*pos = i->pos;
			return true;
		}
		return false;
	}
};

struct Pareto {

	domain dom;
	vec<sptr<term2>> objs; /* assume normalized: search range is [0,1] */
	cmp_t direction;
	sptr<form2> alpha, beta, eta;
	theta_t theta;
	kay::Q eps;
	vec<opt<smlp_result_base>> s;
	hset<size_t> K; /* bounds that can still be raised by at least eps */
	hset<size_t> K_prev;

	Pareto(domain dom, vec<sptr<term2>> objs, cmp_t direction,
	       sptr<form2> alpha, sptr<form2> beta, sptr<form2> eta,
	       theta_t theta, kay::Q eps)
	: dom(move(dom))
	, objs(move(objs))
	, direction(move(direction))
	, alpha(move(alpha))
	, beta(move(beta))
	, eta(move(eta))
	, theta(move(theta))
	, eps(move(eps))
	, s(k())
	, K{}
	{
		/* K = {1,...,k} \ dom s */
		for (size_t i=0; i<k(); i++)
			if (!s[i])
				K.insert(i);
	}

	size_t k() const { return size(objs); }

	/* Greatest lower bound on all objectives o_j without bounds
	 * (j \notin\dom t) under those constraints F_t for the bounds t
	 * defines.
	 *
	 * b_t := max { min_{j=1,...,k, j not in \dom t} o_j(x) : x \in F_T }
	 * b_t \in \bar R = R \cup {\pm\infty}
	 *
	 * where
	 *
	 * F_t = { x \in D : \bigland_{j\in\dom t} o_j(x) >= t(j) }
	 */
	extended<smlp_result_base> b(const vec<opt<smlp_result_base>> &t, kay::Q delta = 0) const
	{
		vec<sptr<form2>> eta_F_t_conj = { eta };
		sptr<term2> min_objs = nullptr;
		for (size_t j=0; j<k(); j++) {
			sptr<term2> obj = objs[j];
			if (t[j]) {
				eta_F_t_conj.push_back(make2f(prop2 {
					direction, obj, make2t(cnst2 { t[j]->threshold })
				}));
			} else {
				min_objs = min_objs ? make2t(ite2 {
					make2f(prop2 { LT, obj, min_objs }),
					obj,
					min_objs,
				}) : obj;
			}
		}
		sptr<form2> eta_F_t = conj(move(eta_F_t_conj));
		if (!min_objs)
			return infty { is_less(direction) ? false : true };

		bounded_search_ival range(ival { 0, 1 }, eps);
		str logic = smt2_logic_str(dom, conj({
			alpha, beta, eta_F_t,
			make2f(prop2 { LT, min_objs, zero }),
		}));
		vec<smlp_result> r = optimize_EA(direction, dom,
		                                 min_objs, alpha,
		                                 beta, eta_F_t, delta,
		                                 range, theta, logic.c_str());
		if (empty(r))
			return infty { is_less(direction) ? true : false };

		const smlp_result &opt = r.back();
		/*
		if (info(mod_prob,
		         "%s of objective in theta in [%s, %s] ~ [%f, %f] around:\n",
		         is_less((cmp_t)c) ? "min max" : "max min",
		         opt.threshold.get_str().c_str(),
		         obj_range->hi()->get_str().c_str(),
		         opt.threshold.get_d(),
		         obj_range->hi()->get_d()))
			print_model(stderr, opt.point, 2);*/

		return opt;
	}

	const smlp_result_base & last_point() const
	{
		assert(!empty(K_prev));
		const opt<smlp_result_base> &v = s[*begin(K_prev)];
		assert(v);
		return *v;
	}

	bool done() const { return empty(K); }

	void step()
	{
		assert(!is_less(direction)); /* not implemented, see + eps below */

		/* s <- {(j,s(j)) : j=1,...,k, j not in K} */
		vec<opt<smlp_result_base>> s2(k());
		for (size_t j=0; j<k(); j++)
			if (!K.contains(j)) {
				assert(s[j]);
				s2[j] = s[j];
			}
		s = move(s2);

		/* s <- s \cup {(j,b_s) : j in K} */
		extended<smlp_result_base> bs = b(s);
		assert(!is_infty(bs));
		for (size_t j : K)
			s[j] = *bs.get<smlp_result_base>();

		hset<size_t> KN = K;
		for (size_t j : K) {
			/* K' <- K \ {j} */
			hset<size_t> K2 = KN;
			K2.erase(j);

			/* t <- s|_{K'} */
			vec<opt<smlp_result_base>> t(k());
			for (size_t i : K2)
				t[i] = s[i];

			extended<smlp_result_base> bt = b(t);
			assert(!is_infty(bt));
			if (!do_cmp<kay::Q>(bt.get<smlp_result_base>()->threshold, direction, s[j]->threshold + eps)) {
				/* cannot increase bound on o_j simultaneously
				 * by epsilon */
				warn(mod_prob,"fixing objective %zu on threshold %s ~ %g\n",
				     j, s[j]->threshold.get_str().c_str(), s[j]->threshold.get_d());
				KN.erase(j);
			}
		}
		K_prev = std::exchange(K, move(KN));
	}
};
/*
static void
optimize_pareto_C(const domain &dom,
                  const vec<sptr<term2>> &objs,
                  const kay::Q &eps,
{
	assert(eps > 0);
	
}
*/

static void print_model(FILE *f, const hmap<str,sptr<term2>> &model, int indent)
{
	size_t k = 0;
	vec<const pair<const str,sptr<term2>> *> v;
	for (const auto &p : model) {
		k = max(k, p.first.length());
		v.emplace_back(&p);
	}
	std::ranges::sort(v, {}, [](const auto *a) -> strview { return a->first; });
	for (const auto *p : v)
		fprintf(f, "%*s%*s = %s\n", indent, "", -(int)k, p->first.c_str(),
		        to_string(p->second->get<cnst2>()->value).c_str());
}

template <typename T>
static bool from_string(const char *s, T &v)
{
	using std::from_chars;
	using kay::from_chars;
	auto [end,ec] = from_chars(s, s + strlen(s), v);
	return !*end && ec == std::errc {};
}

static void opt_single(const domain &dom, const sptr<term2> &lhs,
                       const sptr<form2> &alpha, const sptr<form2> &beta,
                       const sptr<form2> &eta, cmp_t c,
                       uptr<search_base> obj_range, bool dump_smt2, int timeout,
                       bool solve,
                       const char *delta_s, const vec<str> &args, int N,
                       const fun<sptr<form2>(opt<kay::Q>,const hmap<str,sptr<term2>> &)> &theta)
{
	if (!solve)
		return;

	/* hint for the solver: (non-)linear real arithmetic, potentially also
	 * with integers */
	str logic = smt2_logic_str(dom, conj({ alpha, beta, eta, make2f(prop2 { LT, lhs, zero }) }));

	kay::Q delta;
	if (!from_string(delta_s, delta) || delta < 0)
		MDIE(mod_smlp,1,"error: cannot parse DELTA as a positive "
		                "rational constant: '%s'\n", delta_s);

	printf("d,%s\n", std::filesystem::current_path().c_str());
	printf("c,%zu,", size(args));
	for (const str &s : args)
		fwrite(s.c_str(), 1, s.length() + 1, stdout);
	printf("\n");

	vec<smlp_result> r = optimize_EA(c, dom, lhs, alpha, beta, eta, delta,
	                                 *obj_range, theta, logic.c_str());
	if (empty(r)) {
		info(mod_prob,
			"no solution for objective in theta in "
			"[%s, %s] ~ [%f, %f]\n",
			obj_range->lo()->get_str().c_str(),
			obj_range->hi()->get_str().c_str(),
			obj_range->lo()->get_d(),
			obj_range->hi()->get_d());
	} else {
		const smlp_result &opt = r.back();
		if (info(mod_prob,
		         "%s of objective in theta in [%s, %s] ~ [%f, %f] around:\n",
		         is_less((cmp_t)c) ? "min max" : "max min",
		         opt.threshold.get_str().c_str(),
		         obj_range->hi()->get_str().c_str(),
		         opt.threshold.get_d(),
		         obj_range->hi()->get_d()))
			print_model(stderr, opt.point, 2);
		for (const auto &s : r) {
			kay::Q c = s.center_value(lhs);
			note(mod_prob,
			     "T: %s ~ %f -> center: %s ~ %f, search range: [%s,%s]\n",
			     s.threshold.get_str().c_str(),
			     s.threshold.get_d(),
			     c.get_str().c_str(), c.get_d(),
			     s.obj_range->lo() ? s.obj_range->lo()->get_str().c_str() : nullptr,
			     s.obj_range->hi() ? s.obj_range->hi()->get_str().c_str() : nullptr);
		}
	}
	for (int n=1; n<N; n++)
		;
}

static void alarm_handler(int sig)
{
	if (interruptible *p = interruptible::is_active)
		p->interrupt();
	signal(sig, alarm_handler);
}

static void dump_smt2(FILE *f, const char *logic, const problem &p)
{
	fprintf(f, "(set-logic %s)\n", logic);
	dump_smt2(f, p.dom);
	fprintf(f, "(assert ");
	dump_smt2(f, *p.p);
	fprintf(f, ")\n");
	fprintf(f, "(check-sat)\n");
	fprintf(f, "(get-model)\n");
}

static void slv_single(const domain &dom, const sptr<term2> &lhs,
                       const sptr<form2> &alpha, const sptr<form2> &beta,
                       const sptr<form2> &eta, cmp_t c, const char *cnst_s,
                       bool dump_smt2, int timeout, bool solve, int N)
{
	if (*beta != *true2)
		MDIE(mod_smlp,1,"-b BETA is not supported when CNST is given\n");

	/* interpret the CNST on the right hand side */
	kay::Q cnst;
	if (!from_string(cnst_s, cnst))
		MDIE(mod_smlp,1,"CNST must be a rational constant\n");
	dbg(mod_prob,"cnst: %s\n", cnst.get_str().c_str());
	sptr<term2> rhs = make2t(cnst2 { move(cnst) });

	/* the problem consists of domain and the eta, alpha and
	 * (EXPR OP CNST) constraints */
	problem p = {
		move(dom),
		conj({ eta, alpha, make2f(prop2 { c, lhs, rhs, }) }),
	};

	/* optionally dump the smt2 representation of the problem */
	if (dump_smt2)
		::dump_smt2(stdout, smt2_logic_str(dom, p.p).c_str(), p);

	if (timeout > 0) {
		signal(SIGALRM, alarm_handler);
		alarm(timeout);
	}

	/* optionally solve the problem */
	if (solve)
		solve_exists(p.dom, p.p).match(
		[&,lhs=lhs](const sat &s) {
			kay::Q q = to_Q(cnst_fold(lhs, s.model)->get<cnst2>()->value);
			info(mod_prob,"sat, lhs value: %s ~ %g, model:\n",
			        q.get_str().c_str(), q.get_d());
			print_model(stderr, s.model, 2);
			for (const auto &[n,c] : s.model) {
				kay::Q q = to_Q(c->get<cnst2>()->value);
				assert(p.dom[n]->contains(q));
			}
		},
		[](const unsat &) { info(mod_prob,"unsat\n"); },
		[](const unknown &u) {
			info(mod_prob,"unknown: %s\n", u.reason.c_str());
		}
		);
}

#ifdef SMLP_ENABLE_KERAS_NN
# define USAGE "{ DOMAIN EXPR | H5-NN SPEC GEN IO-BOUNDS }"
#else
# define USAGE "DOMAIN EXPR"
#endif

#define DEF_DELTA	"0"
#define DEF_MAX_PREC	"0.05"

[[noreturn]]
static void usage(const char *program_name, int exit_code)
{
	FILE *f = exit_code ? stderr : stdout;
	fprintf(f, "\
usage: %s [-OPTS] [--] " USAGE " OP [CNST]\n\
", program_name);
	if (!exit_code) {
		fprintf(f,"\
\n\
Options [defaults]:\n\
  -a ALPHA     additional ALPHA constraints restricting candidates *and*\n\
               counter-examples (only points in regions satisfying ALPHA\n\
               are considered counter-examples to safety); can be given multiple\n\
               times, the conjunction of all is used [true]\n\
  -b BETA      additional BETA constraints restricting candidates and safe\n\
               regions (all points in safe regions satisfy BETA); can be given\n\
               multiple times, the conjunction of all is used [true]\n\
  -c COLOR     control colored output: COLOR can be one of: on, off, auto [auto]\n\
  -C COMPAT    use a compatibility layer, can be given multiple times; supported\n\
               values for COMPAT:\n\
               - python: reinterpret floating point constants as python would\n\
                         print them\n\
               - bnds-dom: the IO-BOUNDS are domain constraints, not just ALPHA\n\
               - clamp: clamp inputs (only meaningful for NNs) [no]\n\
               - gen-obj: use single objective from GEN instead of all H5-NN\n\
                          outputs [no]\n\
  -d DELTA     increase radius around counter-examples by factor (1+DELTA) or by\n\
               the constant DELTA if the radius is zero [" DEF_DELTA "]\n\
  -e ETA       additional ETA constraints restricting only candidates, can be\n\
               given multiple times, the conjunction of all is used [true]\n\
  -E EPSILON   constant for step-wise increase of Pareto thresholds [none]\n\
  -F IFORMAT   determines the format of the EXPR file; can be one of: 'infix',\n\
               'prefix' (only EXPR) [infix]\n\
  -h           displays this help message\n\
  -i SUBDIVS   use interval evaluation with SUBDIVS subdivisions and fall back\n\
               to the critical points solver before solving symbolically [no]\n\
  -I EXT-INC   optional external incremental SMT solver [value for -S]\n\
  -n           dry run, do not solve the problem [no]\n\
  -o OBJ-SPEC  specify objective explicitely (only meaningful for NNs), an\n\
               expression using the labels from SPEC or 'Pareto(E1,E2,...)'\n\
               where E1,E2,... are such expressions [EXPR]\n\
  -O OBJ-BNDS  scale objective(s) according to min-max output bounds (only\n\
               meaningful for NNs, either .csv or .json) [none]\n\
  -p           dump the expression in Polish notation to stdout (only EXPR) [no]\n\
  -P PREC      maximum precision to obtain the optimization result for [" DEF_MAX_PREC "]\n\
  -Q QUERY     answer a query about the problem; supported QUERY:\n\
               - vars: list all variables\n\
               - out : list all defined outputs\n\
  -r           re-cast bounded integer variables as reals with equality\n\
               constraints (requires -C bnds-dom); cvc5 >= 1.0.1 requires this\n\
               option when integer variables are present\n\
  -R LO,HI     optimize threshold in the interval [LO,HI] [interval-evaluation\n\
               of the LHS]\n\
  -s           dump the problem in SMT-LIB2 format to stdout [no]\n\
  -S EXT-CMD   invoke external SMT solver instead of the built-in one via\n\
               'SHELL -c EXT-CMD' where SHELL is taken from the environment or\n\
               'sh' if that variable is not set []\n\
  -t TIMEOUT   set the solver timeout in seconds, 0 to disable [0]\n\
  -T THRESHS   instead of on an interval perform binary search among the\n\
               thresholds in the list given in THRESHS; overrides -R and -P;\n\
               THRESHS is either a triple LO:INC:HI of rationals with INC > 0 or\n\
               a comma-separated list of rationals\n\
  -v[LOGLVL]   increases the verbosity of all modules or sets it as specified in\n\
               LOGLVL: comma-separated list of entries of the form [MODULE=]LVL\n\
               where LVL is one of none, error, warn, info, note, debug [note];\n\
               see below for values of the optional MODULE to restrict the level\n\
               to; if LOGLVL is given there must not be space between it and -v\n\
  -V           display version information\n\
\n\
The DOMAIN is a text file containing the bounds for all variables in the\n\
form 'NAME -- RANGE' where NAME is the name of the variable and RANGE is either\n\
an interval of the form '[a,b]' or a list of specific values '{a,b,c,d,...}'.\n\
Empty lines are skipped.\n\
\n\
The EXPR file contains a polynomial expression in the variables specified by the\n\
DOMAIN-FILE. The format is either an infix notation or the prefix notation also\n\
known as Polish notation. The expected format can be specified through the -F\n\
switch.\n\
\n\
The problem to be solved is specified by the two parameters OP CNST where OP is\n\
one of '<=', '<', '>=', '>', '==' and '!='. Remember quoting the OP on the shell\n\
to avoid unwanted redirections. CNST is a rational constant in the same format\n\
as those in the EXPR file (if any).\n\
\n\
For log detail setting -v, MODULE can be one of:\n\
");
	vec<strview> mods(size(Module::modules));
	using namespace std::ranges;
	transform(Module::modules, begin(mods), [](const auto &p) { return p.first; });
	sort(mods);
	for (size_t i=0; i<size(mods); i++)
		fprintf(f, "%s%.*s", i ? ", " : "  ",
		        (int)mods[i].length(), mods[i].data());
	fprintf(f,"\n\
\n\
Options are first read from the environment variable SMLP_OPTS, if set.\n\
\n\
Exit codes are as follows:\n\
  0: normal operation\n\
  1: invalid user input\n\
  2: unexpected SMT solver output (e.g., 'unknown' on interruption)\n\
  3: unhandled SMT solver result (e.g., non-rational assignments)\n\
  4: partial function applicable outside of its domain (e.g., 'Match(expr, .)')\n\
\n\
Developed by Franz Brausse <franz.brausse@manchester.ac.uk>.\n\
License: Apache 2.0; part of SMLP.\n\
");
	}
	exit(exit_code);
}

static void version_info()
{
	printf("SMLP version %d.%d.%d\n", SMLP_VERSION_MAJOR,
	       SMLP_VERSION_MINOR, SMLP_VERSION_PATCH);
	printf("Built with features:"
#ifdef KAY_USE_FLINT
	       " flint"
#endif
#ifdef SMLP_ENABLE_KERAS_NN
	       " keras-nn"
#endif
#ifdef SMLP_ENABLE_Z3_API
	       " z3"
#endif
	       "\n");
	printf("Libraries:\n");
	printf("  GMP version %d.%d.%d linked %s\n",
	       __GNU_MP_VERSION, __GNU_MP_VERSION_MINOR,
	       __GNU_MP_VERSION_PATCHLEVEL, __gmp_version);
#ifdef KAY_USE_FLINT
	printf("  Flint version %s linked %s\n",
	       FLINT_VERSION, flint_version);
	/*
	printf("  MPFR version %s linked %s\n",
	       MPFR_VERSION_STRING, mpfr_get_version());*/
#endif
	unsigned maj, min, pat, rev;
#ifdef SMLP_ENABLE_Z3_API
	Z3_get_version(&maj, &min, &pat, &rev);
	printf("  Z3 version %d.%d.%d linked %d.%d.%d\n",
	       Z3_MAJOR_VERSION, Z3_MINOR_VERSION,
	       Z3_BUILD_NUMBER, maj, min, pat);
#endif
#ifdef SMLP_ENABLE_KERAS_NN
	uint32_t kjson_v = kjson_version();
	printf("  kjson version %d.%d.%d linked %d.%d.%d\n",
	       KJSON_VERSION >> 16, (KJSON_VERSION >> 8) & 0xff,
	       KJSON_VERSION & 0xff, kjson_v >> 16,
	       (kjson_v >> 8) & 0xff, kjson_v & 0xff);
	H5get_libversion(&maj, &min, &pat);
	printf("  HDF5 version %d.%d.%d linked %d.%d.%d\n",
	       H5_VERS_MAJOR, H5_VERS_MINOR, H5_VERS_RELEASE,
	       maj, min, pat);
#endif
	(void)maj;
	(void)min;
	(void)pat;
	(void)rev;
}

static sptr<form2> parse_infix_form2(const char *s)
{
	static const unroll_funs_t logic = {
		{"And", unroll_and},
		{"Or", unroll_or},
		{"Not", unroll_not},
		{"+", unroll_add},
		{"-", unroll_sub},
		{"*", unroll_mul},
	};
	return *unroll(parse_infix(s, false), logic, unroll_cnst_None).get<sptr<form2>>();
}

template <typename T>
static void note_smt2_line(const char *pre, const sptr<T> &g, const char *post = "")
{
	if (note(mod_prob, "%s", pre)) {
		smlp::dump_smt2(stderr, *g);
		fprintf(stderr, "%s\n", post);
	}
}

static ival get_obj_range(const char *obj_range_s,
                          const domain &dom, const sptr<term2> &obj)
{
	ival obj_range;
	if (obj_range_s) {
		using kay::from_chars;
		const char *end = obj_range_s + strlen(obj_range_s);
		auto r = from_chars(obj_range_s, end, obj_range.lo);
		bool ok = r.ec == std::errc {} && *r.ptr == ',';
		if (ok) {
			r = from_chars(r.ptr+1, end, obj_range.hi);
			ok &= r.ec == std::errc {} && !*r.ptr;
		}
		if (!ok)
			MDIE(mod_smlp,1,"cannot parse argument '%s' to '-R' as "
			                "a pair of rational numbers\n",
			     obj_range_s);
		note(mod_prob,"got objective range from -R: [%s,%s]\n",
		        obj_range.lo.get_str().c_str(),
		        obj_range.hi.get_str().c_str());
		if (obj_range.lo > obj_range.hi)
			warn(mod_prob,"empty objective range\n");
	} else {
		auto lh = dbl_interval_eval(dom, obj);
		if (!lh)
			MDIE(mod_prob,1,"domain is empty\n");
		info(mod_prob,"approximated objective range: [%g,%g], "
		              "use -R to specify it manually\n",
		        lh->first, lh->second);
		if (!isfinite(lh->first) || !isfinite(lh->second))
			MDIE(mod_prob,1,"optimization over an unbounded range "
			                "is not supported\n");
		obj_range.lo = kay::Q(lh->first);
		obj_range.hi = kay::Q(lh->second);
	}
	return obj_range;
}

static cmp_t parse_op(const std::string_view &s)
{
	for (size_t c=0; c<ARRAY_SIZE(cmp_s); c++)
		if (cmp_s[c] == s)
			return (cmp_t)c;
	MDIE(mod_smlp,1,"OP '%.*s' unknown\n",(int)s.length(), s.data());
}

static uptr<search_base>
parse_search_range(char *threshs_s, const char *max_prec_s,
                   const char *obj_range_s, const domain &dom,
                   const sptr<term2> &lhs)
{
	if (threshs_s) {
		if (obj_range_s)
			warn(mod_prob,"option -R %s is unused, -T overrides it\n",
			     obj_range_s);
		vec<kay::Q> vs;
		if (strchr(threshs_s, ':')) {
			kay::Q v[3];
			const char *beg = threshs_s;
			const char *end = beg + strlen(beg);
			for (size_t i=0; i<3; i++) {
				using kay::from_chars;
				auto r = from_chars(beg, end, v[i]);
				if (r.ec != std::errc {})
					MDIE(mod_smlp,1,"cannot parse '%s' in "
					     "THRESHS as a rational constant\n",
					     threshs_s);
				if (*r.ptr != (i < 2 ? ':' : '\0'))
					MDIE(mod_smlp,1,"expected three "
					     "':'-delimited rational numbers in "
					     "THRESHS, here: '%s'\n", r.ptr);
				beg = r.ptr + 1;
			}
			if (v[1] <= 0)
				MDIE(mod_smlp,1,"INC must be positive\n");
			for (kay::Q q = v[0]; q <= v[2]; q += v[1]) {
				dbg(mod_smlp,"got '%s' for -T\n",q.get_str().c_str());
				vs.push_back(q);
			}
		} else {
			for (char *s = NULL, *t = strtok_r(threshs_s, ",", &s);
			     t; t = strtok_r(NULL, ",", &s)) {
				kay::Q v;
				if (!from_string(t, v))
					MDIE(mod_smlp,1,"cannot parse '%s' in THRESHS "
					                "as a rational constant\n", t);
				vs.emplace_back(move(v));
			}
			std::sort(begin(vs), end(vs));
			vs.erase(std::unique(begin(vs), end(vs)), end(vs));
		}
		if (empty(vs))
			MDIE(mod_prob,1,"list THRESHS cannot be empty\n");
		return std::make_unique<search_list>(move(vs));
	} else {
		kay::Q max_prec;
		if (!from_string(max_prec_s, max_prec) || max_prec < 0)
			MDIE(mod_smlp,1,"cannot parse MAX_PREC as a non-negative "
			                "rational constant: '%s'\n", max_prec_s);

		ival range = get_obj_range(obj_range_s, dom, lhs);
		return max_prec
		     ? std::make_unique<bounded_search_ival>(range, max_prec)
		     : std::make_unique<search_ival>(range);
	}
}

static void set_loglvl(char *arg)
{
	if (!arg) {
		for (const auto &[n,m] : Module::modules)
			m->lvl = (loglvl)((int)m->lvl + 1);
		return;
	}
	hmap<strview,loglvl> values = {
		{ "none" , QUIET },
		{ "error", ERROR },
		{ "warn" , WARN },
		{ "info" , INFO },
		{ "note" , NOTE },
		{ "debug", DEBUG },
	};
	for (char *s = NULL, *t = strtok_r(arg, ",", &s); t;
	     t = strtok_r(NULL, ",", &s)) {
		char *ss, *mod = strtok_r(t, "=", &ss);
		assert(mod);
		char *lvl = strtok_r(NULL, "=", &ss);
		if (!lvl)
			swap(mod, lvl);
		if (mod && lvl)
			dbg(mod_prob,"setting log-level of '%s' to '%s'\n",
			             mod, lvl);
		else
			dbg(mod_prob,"setting log-level to '%s'\n", lvl);
		auto jt = values.find(lvl);
		if (jt == end(values))
			MDIE(mod_smlp,1,"unknown log level '%s' given in LOGLVL\n",
			     lvl);
		if (mod) {
			auto it = Module::modules.find(mod);
			if (it == end(Module::modules))
				MDIE(mod_smlp,1,"unknown module '%s' given in "
				                "LOGLVL\n",mod);
			it->second->lvl = jt->second;
		} else
			for (const auto &[n,m] : Module::modules)
				m->lvl = jt->second;
	}
}

static bool is_constant(domain dom, const sptr<term2> &t, const hset<str> &wrt)
{
	hmap<str,sptr<term2>> rename;
	for (const str &n : wrt) {
		str r = fresh(dom, n);
		rename.emplace(n, make2t(name { r }));
		component *c = dom[n];
		assert(c);
		dom.emplace_back(move(r), *c);
	}
	return solve_exists(dom, make2f(prop2 { NE, t, subst(t, rename) })).match(
	[](const sat &) { return false; },
	[](const unsat &) { return true; },
	[](const unknown &u) -> bool { MDIE(mod_prob,2,"is_constant is unknown: %s\n",u.reason.c_str()); }
	);
}

static bool is_constant(domain dom, const sptr<term2> &t)
{
	return is_constant(move(dom), t, free_vars(t));
}

static bool is_constant_on(domain dom, const sptr<term2> &t)
{
	hset<str> wrt = free_vars(t);
	for (auto it = begin(wrt); it != end(wrt);)
		if (!dom[*it])
			it = wrt.erase(it);
		else
			++it;
	return is_constant(move(dom), t, wrt);
}

static void parse_obj_spec(const char *obj_spec, const domain &dom,
                           const hmap<str,sptr<term2>> &funs, sptr<term2> &lhs,
                           vec<sptr<term2>> &pareto)
{
	auto proc = [&](const expr &f) {
		sptr<term2> t = *unroll(f, {
			{"+", unroll_add},
			{"-", unroll_sub},
			{"*", unroll_mul},
		}, unroll_cnst_ZQ).get<sptr<term2>>();
		for (const str &s : free_vars(t))
			if (!dom[s] && !funs.contains(s))
				MDIE(mod_prob,1,"free variable '%s' in "
				     "OBJ-SPEC is neither in domain nor "
				     "a defined output\n",s.c_str());
		return simplify(t);
	};
	expr e = parse_infix(obj_spec, false);
	if (const call *c = e.get<call>()) {
		if (c->func->get<name>()->id != "Pareto")
			MDIE(mod_smlp,1,"cannot interpret OBJ-SPEC '%s'\n",
			     obj_spec);
		std::set<term2> objs;
		for (const expr &f : c->args) {
			sptr<term2> t = proc(f);
			auto [it,ins] = objs.emplace(*t);
			if (!ins || is_ground(t)) {
				err(mod_smlp,"%s Pareto objective expression: ",
				    !ins ? "duplicate" : "ground") &&
				(smlp::dump_smt2(stderr, *t),
				 fprintf(stderr, "\n"));
				DIE(1,"");
			}
			pareto.emplace_back(move(t));
		}
		if (size(pareto) < 2)
			MDIE(mod_smlp,1,
			     "%s objective for Pareto optimization\n",
			     empty(pareto) ? "no" : "only single");
		if (note(mod_prob,"Pareto objectives:\n"))
			for (const sptr<term2> &t : pareto) {
				fprintf(stderr, "  ");
				smlp::dump_smt2(stderr, *t);
				fprintf(stderr, "\n");
			}
	} else {
		lhs = proc(e);
		note_smt2_line("objective: ", lhs);
	}
}

int main(int argc, char **argv)
{
	if (const char *opts_c = getenv("SMLP_OPTS")) {
		str opts = opts_c;
		unsetenv("SMLP_OPTS");
		char *shell = getenv("SHELL");
		char sh[] = "sh";
		if (!shell)
			shell = sh;
		char c[] = "-c";
		str cmd = "exec \"$0\" " + opts + " \"$@\"";
		vec<char *> args = { shell, c, cmd.data(), };
		for (int i=0; i<=argc; i++)
			args.push_back(argv[i]);
		execvp(shell, args.data());
		err(mod_smlp,"could not interpret envvar SMLP_OPTS (%s), "
		             "ignoring...\n", strerror(errno));
		setenv("SMLP_OPTS", opts.c_str(), 0);
	}

	/* these determine the mode of operation of this program */
	bool             single_obj    = false;
	bool             solve         = true;
	bool             dump_pe       = false;
	bool             dump_smt2     = false;
	bool             infix         = true;
	bool             python_compat = false;
	bool             inject_reals  = false;
	bool             io_bnds_dom   = false;
	int              timeout       = 0;
	bool             clamp_inputs  = false;
	const char      *obj_spec      = nullptr;
	const char      *obj_bounds    = nullptr;
	const char      *max_prec_s    = nullptr;
	vec<sptr<form2>> alpha_conj    = {};
	vec<sptr<form2>> beta_conj     = {};
	vec<sptr<form2>> eta_conj      = {};
	const char      *delta_s       = nullptr;
	const char      *obj_range_s   = nullptr;
	const char      *eps_s         = nullptr;
	char            *threshs_s     = nullptr;
	vec<strview>     queries;
	int              N             = 1;

	/* record args (before potential reordering) to log to trace later */
	vec<str> args;
	for (int i=0; i<argc; i++)
		args.emplace_back(argv[i]);

	/* parse options from the command-line */
	const char *opts = ":a:b:c:C:d:e:E:F:hi:I:nN:o:O:pP:Q:rR:sS:t:T:v::V";
	for (int opt; (opt = getopt(argc, argv, opts)) != -1;)
		switch (opt) {
		case 'a': alpha_conj.emplace_back(parse_infix_form2(optarg)); break;
		case 'b': beta_conj.emplace_back(parse_infix_form2(optarg)); break;
		case 'c':
			if (optarg == "on"sv)
				Module::log_color = true;
			else if (optarg == "off"sv)
				Module::log_color = false;
			else if (optarg == "auto"sv)
				Module::log_color = isatty(STDERR_FILENO);
			else
				MDIE(mod_smlp,1,"option '-c' only supports 'on', "
				                "'off', 'auto'\n");
			break;
		case 'C':
			if (optarg == "python"sv)
				python_compat = true;
			else if (optarg == "bnds-dom"sv)
				io_bnds_dom = true;
			else if (optarg == "clamp"sv)
				clamp_inputs = true;
			else if (optarg == "gen-obj"sv)
				single_obj = true;
			else
				MDIE(mod_smlp,1,"option '-C' only supports "
				     "'python', 'bnds-dom', 'clamp', 'gen-obj'\n");
			break;
		case 'd': delta_s = optarg; break;
		case 'e': eta_conj.emplace_back(parse_infix_form2(optarg)); break;
		case 'E': eps_s = optarg; break;
		case 'F':
			if (optarg == "infix"sv)
				infix = true;
			else if (optarg == "prefix"sv)
				infix = false;
			else
				MDIE(mod_smlp,1,"option '-F' only supports "
				                "'infix' and 'prefix'\n");
			break;
		case 'h': usage(argv[0], 0);
		case 'i': {
			if (from_string(optarg, intervals))
				break;
			MDIE(mod_smlp,1,"SUBDIVS argument to '-i' must be numeric\n");
		}
		case 'I': inc_solver_cmd = optarg; break;
		case 'n': solve = false; break;
		case 'N': N = atoi(optarg); break;
		case 'p': dump_pe = true; break;
		case 'P': max_prec_s = optarg; break;
		case 'Q': queries.push_back(optarg); break;
		case 'r': inject_reals = true; break;
		case 'R': obj_range_s = optarg; break;
		case 'o': obj_spec = optarg; break;
		case 'O': obj_bounds = optarg; break;
		case 's': dump_smt2 = true; break;
		case 'S': ext_solver_cmd = optarg; break;
		case 't':
			if (from_string(optarg, timeout))
				break;
			MDIE(mod_smlp,1,"TIMEOUT argument to '-t' must be numeric\n");
		case 'T': threshs_s = optarg; break;
		case 'v': set_loglvl(optarg); break;
		case 'V': version_info(); exit(0);
		case ':': MDIE(mod_smlp,1,"option '-%c' requires an argument\n",
		               optopt);
		case '?': MDIE(mod_smlp,1,"unknown option '-%c'\n",optopt);
		}

	pre_problem pp;

	/* ------------------------------------------------------------------
	 * Obtain the pre_problem
	 * ------------------------------------------------------------------ */

#ifdef SMLP_ENABLE_KERAS_NN
	if (argc - optind >= 5) {
		/* Solve NN problem */
		const char *hdf5_path = argv[optind];
		const char *spec_path = argv[optind+1];
		const char *gen_path = argv[optind+2];
		const char *io_bounds = argv[optind+3];
		pp = parse_nn(gen_path, hdf5_path, spec_path, io_bounds,
		              obj_bounds, clamp_inputs, single_obj);
		optind += 4;
	} else
#else
	/* these are unused w/o NN support */
	(void)single_obj;
	(void)clamp_inputs;
	(void)obj_bounds;
#endif
	if (argc - optind >= 3) {
		/* Solve polynomial problem */
		pp = parse_poly_problem(argv[optind], argv[optind+1],
		                        python_compat, dump_pe, infix);
		optind += 2;
	} else
		usage(argv[0], 1);

	/* find out about the OP comparison operation */
	if (argc - optind < 1)
		usage(argv[0], 1);
	cmp_t c = parse_op(argv[optind++]);

	/* ------------------------------------------------------------------
	 * Preprocess pre_problem
	 * ------------------------------------------------------------------ */

	if (inject_reals && !(io_bnds_dom || empty(pp.input_bounds)))
		MDIE(mod_smlp,1,"\
error: -r requires -C bnds-dom: re-casting integers as reals based on IO-BOUNDS\n\
implies that IO-BOUNDS are regarded as domain constraints instead of ALPHA.\n");
	sptr<form2> alpha = pp.interpret_input_bounds(io_bnds_dom, inject_reals);

	auto &[dom,lhs,funs,f_bnds,in_bnds,eta,pc,theta] = pp;
	for (const auto &[n,t] : funs)
		(dbg(mod_prob,"defined output '%s': ", n.c_str()) &&
		 (smlp::dump_smt2(stderr, *t), fprintf(stderr, "\n"))) ||
		note(mod_prob,"defined output '%s'\n", n.c_str());

	/* ------------------------------------------------------------------
	 * If bounds on named outputs are given, scale them.
	 * ------------------------------------------------------------------ */

	hmap<str,sptr<term2>> funs_org = funs;
	for (const auto &[n,range] : f_bnds) {
		auto it = funs.find(n);
		if (it == end(funs))
			MDIE(mod_smlp,1,"normalizing undefined output '%s'\n",
			     n.c_str());
		kay::Q len = length(range);
		if (len <= 0)
			MDIE(mod_smlp,1,"normalization range of '%s' does not "
			                "have length > 0: [%s,%s]\n",
			     n.c_str(),
			     range.lo.get_str().c_str(),
			     range.hi.get_str().c_str());
		dbg(mod_prob,"scaling '%s' from range [%s,%s] ~ [%g,%g] to [0,1]\n",
		     n.c_str(),
		     range.lo.get_str().c_str(), range.hi.get_str().c_str(),
		     range.lo.get_d(), range.hi.get_d()) ||
		info(mod_prob,"scaling '%s' from range ~ [%g,%g] to [0,1]\n",
		     n.c_str(), range.lo.get_d(), range.hi.get_d());
		sptr<term2> &f = it->second;
		using namespace kay;
		f = make2t(bop2 { bop2::MUL,
			make2t(bop2 { bop2::SUB, f, make2t(cnst2{ range.lo }) }),
			make2t(cnst2 { inv(len) }),
		});
	}

	/* ------------------------------------------------------------------
	 * Parse the -o OBJ-SPEC option, if it was there
	 * ------------------------------------------------------------------ */

	if (obj_spec && lhs)
		MDIE(mod_smlp,1,"cannot use both, -o OBJ-SPEC and an anonymous "
		                "objective function given via EXPR or -C gen-obj\n");

	if (!obj_spec && !lhs)
		MDIE(mod_smlp,1,"no objective specified; please use either "
		                "-o OBJ-SPEC or -C gen-obj\n");

	vec<sptr<term2>> pareto, pareto_org;
	if (obj_spec) {
		parse_obj_spec(obj_spec, dom, funs, lhs, pareto);
		pareto_org = pareto;
	}

	/* ------------------------------------------------------------------
	 * Complete the definitions of alpha, beta, eta
	 * ------------------------------------------------------------------ */

	alpha_conj.emplace_back(move(alpha));
	alpha = simplify(conj(move(alpha_conj)));
	note_smt2_line("alpha: ", alpha);

	sptr<form2> beta = simplify(conj(move(beta_conj)));
	note_smt2_line("beta : ", beta);

	eta_conj.emplace_back(move(eta));
	eta = simplify(conj(move(eta_conj)));
	note_smt2_line("eta  : ", eta);

	if (note(mod_prob,"domain:\n"))
		smlp::dump_smt2(stderr, dom);

	/* ------------------------------------------------------------------
	 * Answer any -Q QUERY
	 * ------------------------------------------------------------------ */

	for (const strview &q : queries) {
		bool o = info(mod_smlp,"query '%.*s':\n", (int)q.size(),q.data());
		auto print_vars = [o,dom=&dom](vec<strview> v, const char *unbound) {
			sort(begin(v), end(v));
			for (const strview &id : v)
				o && fprintf(stderr, "  '%.*s': %s\n",
				             (int)id.length(), id.data(),
				             (*dom)[id] ? "bound" : unbound);
		};
		if (q == "vars") {
			hset<str> h = free_vars(lhs);
			print_vars({ begin(h), end(h) }, "free");
		} else if (q == "out") {
			vec<strview> v;
			for (const auto &[s,t] : funs)
				v.emplace_back(s);
			print_vars(move(v), "defined");
		} else
			MDIE(mod_smlp,1,"unknown query '%.*s'\n",(int)q.size(),q.data());
	}

	/* ------------------------------------------------------------------
	 * Substitute defined terms in alpha, beta, eta and lhs/pareto
	 * ------------------------------------------------------------------ */

	auto check_nonconst = [dom=&dom](const sptr<term2> &t, const sptr<term2> &p = nullptr){
		if (is_constant(*dom, t)) {
			err(mod_prob,"objective is constant: ") &&
			(smlp::dump_smt2(stderr, p ? *p : *t), fprintf(stderr, "\n"));
			DIE(1,"");
		}
		return t;
	};

	info(mod_prob,"checking whether objective%s constant...\n",
	     lhs ? " is" : "s are");

	alpha = subst(alpha, funs);
	beta = subst(beta, funs);
	eta = subst(eta, funs);
	if (lhs)
		lhs = check_nonconst(subst(lhs, funs), lhs);
	else
		for (sptr<term2> &o : pareto)
			o = check_nonconst(subst(o, funs), o);

	/* ------------------------------------------------------------------
	 * Check that the constraints from partial function evaluation are met
	 * on the domain.
	 * ------------------------------------------------------------------ */

	note_smt2_line("checking for out-of-domain application of partial "
	               "functions: (and alpha (not ", pc, "))...");
	result ood = solve_exists(dom, conj({ alpha, neg(pc) }));
	if (const sat *s = ood.get<sat>()) {
		err(mod_prob,"ALPHA and DOMAIN constraints do not imply that "
		             "all function parameters are inside the "
		             "respective function's domain, e.g.:\n");
		print_model(stderr, s->model, 2);
		exit(4);
	} else if (const unknown *u = ood.get<unknown>())
		MDIE(mod_prob,2,"deciding out-of-domain condition: %s\n",
		     u->reason.c_str());
	note(mod_prob,"out-of-domain condition is false\n");

	/* ------------------------------------------------------------------
	 * Finally, determine right problem to solve and solve it
	 * ------------------------------------------------------------------ */

	const char *cnst_s = nullptr;
	if (argc - optind == 1)
		cnst_s = argv[optind++];

	if (argc - optind > 0) {
		if (err(mod_smlp,"unrecognized trailing options:")) {
			for (int i=optind; i<argc; i++)
				fprintf(stderr, " '%s'", argv[i]);
			fprintf(stderr, "\n");
		}
		DIE(1,"");
	}

	if (lhs && cnst_s) {
		if (obj_range_s)
			warn(mod_prob,"objective range specification "
			              "-R is unused when CNST is given\n");
		if (threshs_s)
			warn(mod_prob,"objective thresholds -T are unused when "
			              "CNST is given\n");
		if (max_prec_s)
			warn(mod_prob,"precision PREC is unused when CNST is "
			              "given\n");
		if (delta_s)
			warn(mod_prob,"DELTA is unused when CNST is given\n");

		slv_single(dom, lhs, alpha, beta, eta, c, cnst_s,
		           dump_smt2, timeout, solve, N);
		return EXIT_SUCCESS;
	}

	if (lhs) {
		if (!max_prec_s)
			max_prec_s = DEF_MAX_PREC;
		if (!delta_s)
			delta_s = DEF_DELTA;

		uptr<search_base> obj_range = parse_search_range(threshs_s,
		                                                 max_prec_s,
		                                                 obj_range_s,
		                                                 dom, lhs);
		opt_single(dom, lhs, alpha, beta, eta, c, move(obj_range),
		           dump_smt2, timeout, solve, delta_s, args, N, theta);
		return EXIT_SUCCESS;
	}

	assert(!empty(pareto));
	if (cnst_s)
		MDIE(mod_smlp,1,"comparison of multiple objectives against a "
		                "constant is not implemented\n");

	if (threshs_s)
		MDIE(mod_smlp,1,"-T THRESHS is not supported for Pareto optimization\n");
	if (max_prec_s)
		MDIE(mod_smlp,1,"-P PREC is not supported for Pareto optimization\n");
	if (obj_range_s)
		MDIE(mod_smlp,1,"-R LO,HI is not supported for Pareto optimization\n");

	if (!eps_s)
		MDIE(mod_smlp,1,"-E EPSILON is not set for Pareto optimization\n");
	kay::Q eps;
	if (!from_string(eps_s, eps))
		MDIE(mod_smlp,1,"cannot interpret argument '%s' to -E as a "
		                "rational number\n", eps_s);
	Pareto pi(dom, pareto, c, alpha, beta, eta, theta, eps);
	assert(!pi.done());
	while (!pi.done())
		pi.step();
	if (info(mod_prob,"Pareto optimization done with thresholds:\n"))
		for (size_t i=0; i<pi.k(); i++) {
			assert(pi.s[i]);
			fprintf(stderr, "  (%s ", cmp_smt2[c]);
			smlp::dump_smt2(stderr, *pareto_org[i]);
			fprintf(stderr, " %s)\n", kay::to_string(pi.s[i]->threshold).c_str());
		}
	if (info(mod_prob,"computed point %s-close to Pareto front:\n", eps.get_str().c_str()))
		print_model(stderr, pi.last_point().point, 2);
	if (info(mod_prob,"objectives' values at computed point:\n"))
		for (size_t i=0; i<pi.k(); i++) {
			fprintf(stderr, "  ");
			smlp::dump_smt2(stderr, *pareto_org[i]);
			kay::Q nq = to_Q(cnst_fold(pareto[i], pi.last_point().point)->get<cnst2>()->value);
			kay::Q oq = to_Q(cnst_fold(subst(pareto_org[i], funs_org),
			                                 pi.last_point().point)->get<cnst2>()->value);
			assert(c);
			fprintf(stderr, " normalized ~ %g, original ~ %g\n",
			        nq.get_d(), oq.get_d());
		}
}
