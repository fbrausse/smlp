
#pragma once

#include "expr.hh"
#include "reals.hh"

namespace smlp {

struct term2;

/* Closed interval with rational endpoints */
struct ival {

	kay::Q lo, hi;

	ival(kay::Q v = 0)
	: ival(v, v)
	{}

	ival(kay::Q lo, kay::Q hi)
	: lo(move(lo))
	, hi(move(hi))
	{
		assert(this->lo <= this->hi);
	}

	friend kay::Q length(const ival &i)
	{
		return i.hi - i.lo;
	}

	friend kay::Q mid(const ival &i)
	{
		return (i.lo + i.hi) / 2;
	}

	bool contains(const kay::Z &v) const
	{
		return lo <= v && v <= hi;
	}

	bool contains(const kay::Q &v) const
	{
		return lo <= v && v <= hi;
	}

	friend bool disjoint(const ival &a, const ival &b)
	{
		return a.hi < b.lo || a.lo > b.hi;
	}

	bool intersects(const ival &o) const
	{
		return !disjoint(*this, o);
	}

	bool operator==(const ival &) const = default;
};

/* A "butchered" algebraic real number in the sense that we can't refine its
 * root_bounds */
struct Ap { /* Ap stands for A'. */
	ival root_bounds;

	friend const kay::Q & lo(const Ap &a) { return a.root_bounds.lo; }
	friend const kay::Q & hi(const Ap &a) { return a.root_bounds.hi; }
	friend bool known_Q(const Ap &a) { return lo(a) == hi(a); }

	/* only when known_Q(a) is true */
	friend const kay::Q & to_Q(const Ap &a)
	{
		assert(known_Q(a));
		return lo(a);
	}

	friend bool known_R(const Ap &a) { return known_Q(a); }

	friend const reals::eager::R to_R(const Ap &a)
	{
		assert(known_R(a));
		return { to_Q(a) };
	}
};

hmap<size_t,kay::Q> parse_upoly(const sptr<term2> &t, const strview &var);

template <typename T>
concept integer = std::same_as<T,kay::Z>;

template <typename T>
concept rational = integer<T> || std::same_as<T,kay::Q>;

template <rational T>
struct upoly {

	hmap<size_t,T> coeffs;

	upoly(hmap<size_t,T> coeffs)
	: coeffs(move(coeffs))
	{
		erase_if(this->coeffs, [](const auto &p){ return p.second == 0; });
	}

	kay::Q operator()(const kay::Q &x) const;

	expr to_expr(const expr &x) const;
	sptr<term2> to_term2(const sptr<term2> &x) const;

	friend sptr<term2> to_term2(const upoly &p, const sptr<term2> &x)
	{ return p.to_term2(x); }

	friend ssize_t degree(const upoly &p)
	{
		using std::max;
		ssize_t n = -1;
		for (const auto &[d,c] : p.coeffs)
			n = max<ssize_t>(n, d);
		return n;
	}

	friend bool operator==(const upoly &, const upoly &) = default;

	void assert_minimal() const;

	str get_str(str var = "x") const;

	friend str to_string(const upoly &p, str var = "x")
	{
		return p.get_str(move(var));
	}
};

template <typename T> upoly(hmap<size_t,T>) -> upoly<T>;

extern template struct upoly<kay::Z>;
extern template struct upoly<kay::Q>;

/* First brings p = cn*x^n + ... + c0 into a monic form by dividing all
 * coefficients ci by cn and then multiplies all coefficients ci by the least
 * common multiple of all denominators. This "normalization" preserves the roots
 * of p and brings, e.g., 2x^2-4 into the form x^2-2 and x^2-1/2 into the form
 * 2x^2-1. */
upoly<kay::Z> reduce_monic(upoly<kay::Q> p);

struct A : Ap { /* algebraic reals */
	str var; /* empty: unset */
	upoly<kay::Z> p;
	size_t root_idx; /* zero: unset */

	A(ival root_bounds, str var, upoly<kay::Z> p, size_t root_idx = 0)
	: Ap { move(root_bounds) }
	, var(move(var))
	, p(move(p))
	, root_idx(root_idx)
	{
		assert(known_R(*this));
		this->p.assert_minimal();

		// assert(unique_root(this->p, this->root_bounds));
	}

	A(ival root_bounds, str var, upoly<kay::Q> p, size_t root_idx = 0)
	: A(move(root_bounds), move(var), reduce_monic(move(p)), root_idx)
	{
		dbg(mod_prob, "normalized upoly in A: %s\n", this->p.get_str().c_str());
	}

	friend bool known_R(const A &a)
	{
		return known_Q(a) || (!empty(a.var) && degree(a.p) >= 0);
	}

	reals::eager::R to_R() const; /* only when known_R() is true */
	friend reals::eager::R to_R(const A &a) { return a.to_R(); }

	bool operator==(const A &b) const
	{
		/* minimal polynomial (and thus also the reduce_monic() version
		 * of it) are unique. */
		return p == b.p && root_bounds.intersects(b.root_bounds);
	}
	std::strong_ordering operator<=>(const A &b) const;

	str get_str() const;
	friend str to_string(const A &a) { return a.get_str(); }
};

}
