/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "common.hh"

#include <kay/numbers.hh>

namespace smlp {

namespace reals {

using namespace kay;

namespace eager {

struct R;

static inline R limit(fun<R(const Z &)> s);

struct R {

	fun<Q(const Z &)> apx;

	R(Q c = 0)
	: apx([c=move(c)](const Z &) { return c; })
	{}

	explicit R(fun<Q(const Z &)> f)
	: apx(move(f))
	{}

	friend Q approx(const R &v, const Z &prec)
	{
		return v.apx(prec);
	}

	friend R & operator+=(R &a, R b) { a = move(a) + move(b); return a; }
	friend R operator+(R a, R b)
	{
		return R([a=move(a),b=move(b)](const Z &p){
			return approx(a, p-1) + approx(b, p-1);
		});
	}

	friend R operator+(R a) { return a; }

	friend R operator-(R a)
	{
		return R([a=move(a)](const Z &p){ return -approx(a, p); });
	}

	friend R & operator-=(R &a, R b) { a += -move(b); return a; }
	friend R operator-(R a, R b) { return a + (-move(b)); }

	friend Z ubound_log2(const R &x)
	{
		return sizeinbase(ceil(abs(approx(x, 0))), 2);
	}

	friend R & operator*=(R &a, R b) { a = move(a) * move(b); return a; }
	friend R operator*(R a, R b)
	{
		/*    |a*b - p*q|
		 *  = |a*b - p*q + a*q - a*q|
		 *  = |a*(b-q) + (a-p)*q |
		 * <= |a|*|b-q| + |a-p|*|q|
		 * <= 2^as*|b-q| + |a-p|*2^qs
		 * <= 2^n */
		Z as = ubound_log2(a);
		return R([as=move(as),a=move(a),b=move(b)](const Z &n){
			/* |b-q| <= 2^(n-as-1) -> 2^as*|b-q| <= 2^(n-1) */
			Q q = approx(b, n-as-1);
			Z qs = sizeinbase(ceil(abs(q)), 2);
			/* |q| <= 2^qs -> |a-p| <= 2^(n-qs-1) -> |a-p|*|q| <= 2^(n-1) */
			Q p = approx(a, n-qs-1);
			return p*q;
		});
	}

	friend Z lbound_log2(const R &x)
	{
		Z n = 0;
		Q p = 1; /* 2^n */
		while (p >= abs(approx(x, n-1))) {
			p /= 2;
			n--;
		}
		return n-1;
	}

	friend R inv(R b)
	{
		Z lb = lbound_log2(b);
		return R([lb=move(lb),b=move(b)](const Z &n){
			using std::min;
			return inv(approx(b, min(lb, Z(n-1+2*lb))));
		});
	}

	friend R & operator/=(R &a, R b) { a *= inv(move(b)); return a; }
	friend R operator/(R a, R b)
	{
		return move(a) * inv(move(b));
	}

	friend R abs(R a) {
		return R([a=move(a)](const Z &p){ return abs(approx(a, p)); });
	}

	friend R sqrt(R z)
	{
		return limit([z=move(z)](const Z &n){
			R a = Q(1);
			Q p = scale(Q(1), to_mpz_class(n-1).get_si());
			while (abs(approx(a-z/a, n-1)) >= p)
				a = (a+z/a) / Q(2);
			return a;
		});
	}
};

static inline R limit(fun<R(const Z &)> s)
{
	return R([s=move(s)](const Z &n){ return approx(s(n-1), n-1); });
}
} // end namespace eager
} // end namespace reals

}
