/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "domain.hh"

namespace smlp {

struct sat { hmap<str,sptr<term2>> model; };
struct unsat {};
struct unknown { str reason; };

typedef sumtype<sat,unsat,unknown> result;

struct solver {

	virtual ~solver() = default;
	virtual void declare(const domain &d) = 0;
	virtual void add(const sptr<form2> &f) = 0;
	virtual result check() = 0;

	class all_solutions_iter {

		friend all_solutions_iter all_solutions(solver &s);
		friend class iterator;

		solver *slv;
		result r;

		void next()
		{
			r = slv->check();
			if (const sat *s = r.get<sat>()) {
				vec<sptr<form2>> ne;
				for (const auto &[var,t] : s->model)
					ne.emplace_back(make2f(prop2 { NE, make2t(name { var }), t }));
				slv->add(disj(move(ne)));
			}
		}

	protected:
		explicit all_solutions_iter(solver &slv) : slv(&slv) { next(); }

		friend void swap(all_solutions_iter &a, all_solutions_iter &b)
		{
			swap(a.slv, b.slv);
			swap(a.r, b.r);
		}

		all_solutions_iter & operator=(all_solutions_iter o)
		{
			swap(*this, o);
			return *this;
		}

		all_solutions_iter(all_solutions_iter &&o)
		: slv(o.slv)
		, r(move(o.r))
		{ o.slv = nullptr; }

	public:
		struct sentinel {};
		class iterator {
			friend class all_solutions_iter;

			all_solutions_iter *as;
			explicit iterator(all_solutions_iter *as) : as(as) {}

		public:
			using difference_type = int; /* doesn't make sense, but ok... */
			using value_type = hmap<str,sptr<term2>>;

			iterator(iterator &&o) : as(o.as) { o.as = nullptr; }
			friend void swap(iterator &a, iterator &b) { swap(a.as, b.as); }
			iterator & operator=(iterator o) { swap(*this, o); return *this; }

			value_type & operator*() const { return as->r.get<sat>()->model; }
			iterator & operator++() { as->next(); return *this; }
			void operator++(int) { ++*this; }
			bool operator==(const sentinel &) const { return as->empty(); }
		};

		iterator begin() { return iterator { this }; }
		sentinel end() { return {}; }

		friend iterator begin(all_solutions_iter &s) { return s.begin(); }
		friend sentinel end(all_solutions_iter &s) { return s.end(); }

		bool empty() const { return !r.get<sat>(); }
	};

	struct all_solutions_iter_owned : all_solutions_iter
	                                , std::ranges::view_interface<all_solutions_iter_owned> {
		uptr<solver> s;

		explicit all_solutions_iter_owned(uptr<solver> s)
		: all_solutions_iter(*s)
		, s(move(s))
		{}
	};

	static_assert(std::ranges::input_range<solver::all_solutions_iter>);
	static_assert(std::ranges::view<solver::all_solutions_iter_owned>);

	friend solver::all_solutions_iter all_solutions(solver &s)
	{
		return solver::all_solutions_iter(s);
	}
};

struct acc_solver : solver {

	void declare(const domain &d) override { assert(empty(dom)); dom = d; }
	void add(const sptr<form2> &f) override { asserts.args.push_back(f); }
	result check() override { return static_cast<const acc_solver *>(this)->check(); }
	virtual result check() const = 0;

protected:
	domain dom;
	lbop2 asserts = { lbop2::AND, {} };
};

uptr<solver> mk_solver0(bool incremental, const char *logic);
solver::all_solutions_iter_owned all_solutions(const domain &dom, const sptr<form2> &f);

struct solver_seq : solver {

	const vec<uptr<solver>> solvers;

	explicit solver_seq(vec<uptr<solver>> solvers)
	: solvers(move(solvers))
	{ assert(!empty(this->solvers)); }

	void declare(const domain &d) override
	{
		for (const uptr<solver> &s : solvers)
			s->declare(d);
	}

	void add(const sptr<form2> &f) override
	{
		for (const uptr<solver> &s : solvers)
			s->add(f);
	}

	result check() override
	{
		result r = unknown { "solver sequence is empty" };
		for (const uptr<solver> &s : solvers)
			if (!(r = s->check()).get<unknown>())
				return r;
		return r;
	}
};

struct interruptible {

	virtual ~interruptible() = default;
	virtual void interrupt() = 0;

	static interruptible *is_active;
};

}
