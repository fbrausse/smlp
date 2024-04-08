/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "algebraics.hh"
#include "expr2.hh"

using namespace smlp;
using namespace reals::eager;

template <rational T>
kay::Q upoly<T>::operator()(const kay::Q &x) const
{
	using namespace kay;
	Q r = 0;
	for (const auto &[d,c] : coeffs)
		fma(r, c, pow(x, d));
	return r;
}

template <rational T>
expr upoly<T>::to_expr(const expr &x) const
{
	using kay::to_string;
	using std::to_string;
	sptr<expr> mul = make1e(name { "*" });
	sptr<expr> pow = make1e(name { "^" });
	vec<expr> sum;
	for (const auto &[d,c] : coeffs)
		sum.emplace_back(call { mul, {
			cnst { to_string(c) },
			call { pow, { x, cnst { to_string(d) } } },
		} });
	return call { make1e(name { "+" }), move(sum) };
}

template <rational T>
sptr<term2> upoly<T>::to_term2(const sptr<term2> &x) const
{
	sptr<term2> r = zero;
	for (const auto &[d,c] : coeffs) {
		sptr<term2> y = make2t(cnst2 { c });
		for (size_t i=0; i<d; i++) /* TODO: inefficient */
			y = make2t(bop2 { bop2::MUL, move(y), x });
		r = make2t(bop2 { bop2::ADD, move(r), y });
	}
	return simplify(r);
}

upoly<kay::Z> smlp::reduce_monic(upoly<kay::Q> p)
{
	using kay::Q;
	using kay::Z;

	assert(!empty(p.coeffs));
	const pair<const size_t,Q> *m = nullptr;
	for (const auto &dc : p.coeffs)
		if (!m || dc.first > m->first)
			m = &dc;
	if (m->second != 1) {
		Q mc = m->second;
		for (auto &[d,c] : p.coeffs)
			c /= mc;
	}
	/* now p is monic */
	struct {
		Z v = 1;
		void add(const Z &den)
		{
			Q tmp = v;
			tmp /= den;
			v = tmp.get_num() * den;
		}
	} lcm_denom;
	for (const auto &dc : p.coeffs)
		lcm_denom.add(dc.second.get_den());
	hmap<size_t,Z> q;
	for (const auto &[d,c] : p.coeffs) {
		Q cc = c * lcm_denom.v;
		assert(cc.get_den() == 1);
		q[d] = cc.get_num();
	}
	return upoly(move(q));
}

template <rational T>
void upoly<T>::assert_minimal() const
{
#ifndef NDEBUG
	/* minimal polynomial: check irreducibility over Q,
	 * e.g. by the rational roots test: for any rational root p/q of
	 * cn*x^n + ... + c0 where ci are integers, p divides c0 and
	 * q divides cn.
	 *
	 * Here, all coefficients of the poly are assumed to be
	 * integers, i.e., for all factors p of |c0| and all factors q
	 * of |cn| we need to check that
	 *   p co-prime q -> this->p(+/-p/q) != 0. */
	using kay::Z;
	using kay::Q;
	/*
	Z ifact = 1;
	auto lcm = [](Z a, Z b) -> Z { b = abs(b); return Q(abs(a),b).get_num() * b; };
	for (const auto &[d,c] : this->p.coeffs)
		ifact = lcm(ifact, c.get_den());
	for (const auto &[d,c] : this->p.coeffs)
		ifact = Q(ifact, c.get_den()).get_num() * c.get_den();
	if (ifact != 1)
		MDIE(mod_prob,2,"upoly of algebraic requires multiplication "
		                "with %s to be an integer polynomial\n",
		     ifact.get_str().c_str());*/
	if constexpr (std::same_as<T,kay::Q>)
		for (const auto &[d,c] : coeffs)
			assert(c.get_den() == 1);

	assert(size(coeffs) >= 2);
	const auto *m = &*begin(coeffs);
	for (const auto &dc : coeffs)
		if (dc.first > m->first)
			m = &dc;

	auto c0it = coeffs.find(0);
	Q c0 = c0it == end(coeffs) ? Q(0) : c0it->second;
	const Q &cn = m->second;
	dbg(mod_prob,"upoly with c0: %s, c%zd: %s\n",
	    c0.get_str().c_str(), m->first,
	    cn.get_str().c_str());
	for (Z f0 = 1, c0n = abs(c0.get_num()); f0 <= c0n; f0++) {
		if (c0n % f0 != 0)
			continue;
		for (Z fn = 1, cnn = abs(cn.get_num()); fn <= cnn; fn++) {
			if (cnn % fn != 0)
				continue;
			Q x(f0,fn);
#ifdef KAY_USE_GMPXX
			kay::canonicalize(x);
#endif
			if (x.get_num() != f0)
				continue;
			Q p = (*this)(x);
			Q n = (*this)(-x);
			dbg(mod_prob, "upoly(+/-%s) = %s, %s\n",
			    x.get_str().c_str(),
			    p.get_str().c_str(), n.get_str().c_str());
			assert(p);
			assert(n);
		}
	}
#endif
}

template <rational T>
str upoly<T>::get_str(str var) const
{
	return to_string(to_term2(make2t(name { move(var) })));
}

namespace smlp {
template struct upoly<kay::Z>;
template struct upoly<kay::Q>;
}

str A::get_str() const
{
	return "(root-of " + to_string(p, var) + " (" +
	       kay::to_string(lo(*this)) + " " +
	       kay::to_string(hi(*this)) + "))";
}

R A::to_R() const
{
	using namespace kay;
	if (known_Q(*this))
		return R(to_Q(*this));
	assert(known_R(*this));
	return R([a=*this](const kay::Z &n) mutable {
		/* bisection */
		ival &i = a.root_bounds;
		int lsgn = sgn(a.p(i.lo)), rsgn = sgn(a.p(i.hi));
		if (!lsgn)
			return i.lo;
		if (!rsgn)
			return i.hi;
		assert(lsgn != rsgn);
		Q tgt_p = kay::scale(1, kay::to_mpz_class(n).get_si());
		while (true) {
			Q m = mid(i);
			if (length(i) < tgt_p)
				return m;
			int s = sgn(a.p(m));
			if (!s || s == lsgn)
				i.lo = m;
			if (!s || s == rsgn)
				i.hi = m;
		}
	});
}

std::strong_ordering A::operator<=>(const A &b) const
{
	if (*this == b)
		return std::strong_ordering::equal;
	if (known_Q(*this) && known_Q(b))
		return to_Q(*this) <=> to_Q(b);
	if (root_idx && b.root_idx && p == b.p)
		return root_idx <=> b.root_idx;
	using namespace reals::eager;
	using namespace kay;
	R x = to_R() - b.to_R();
	for (Z p = 0;; p--) {
		Q d = approx(x, p);
		if (abs(d) > kay::scale(1, p.get_si()))
			return sgn(d) < 0 ? std::strong_ordering::less
			                  : std::strong_ordering::greater;
	}
}

hmap<size_t,kay::Q> smlp::parse_upoly(const sptr<term2> &t, const strview &var)
{
	using namespace kay;
	using kay::neg;
	using poly = hmap<size_t,Q>;

	poly ret;
	t->match(
	[&](const name &n) { assert(n.id == var); ret[1] = 1; },
	[&](const bop2 &b) {
		poly l = parse_upoly(b.left, var);
		poly r = parse_upoly(b.right, var);
		switch (b.op) {
		case bop2::SUB:
			for (auto &[e,c] : r)
				neg(c);
			/* fall through */
		case bop2::ADD:
			ret = move(l);
			for (const auto &[e,c] : r)
				ret[e] += c;
			break;
		case bop2::MUL:
			for (const auto &[e,c] : l)
				for (const auto &[f,d] : r)
					ret[e+f] += c * d;
			break;
		}
	},
	[&](const uop2 &u) {
		poly o = parse_upoly(u.operand, var);
		switch (u.op) {
		case uop2::USUB:
			for (auto &[e,c] : o)
				neg(c);
			/* fall through */
		case uop2::UADD:
			ret = move(o);
			break;
		}
	},
	[&](const cnst2 &c) { ret[0] = to_Q(c.value); },
	[&](const ite2 &) { assert(0); }
	);
	erase_if(ret, [](const auto &p){ return p.second == 0; });
	return ret;
}
