/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "expr.hh"

#include <kay/numbers.hh>

namespace smlp {

/* Symbolic representation of a order/equality predicate over ints/reals */

enum cmp_t { LE, LT, GE, GT, EQ, NE, };

inline const char *cmp_s[] = { "<=", "<", ">=", ">", "==", "!=" };
inline const char *cmp_smt2[] = { "<=", "<", ">=", ">", "=", "distinct" };

/* Some properties of cmp_t */
static inline bool is_strict(cmp_t c) { return (unsigned)c & 0x1; }
static inline bool is_order(cmp_t c) { return !((unsigned)c >> 2); }

/* Flip a cmp_t */
static inline cmp_t operator-(cmp_t c)
{
	switch (c) {
	case LE: return GE;
	case LT: return GT;
	case GE: return LE;
	case GT: return LT;
	case EQ:
	case NE:
		return c;
	}
	unreachable();
}

/* Negate a cmp_t */
static inline cmp_t operator~(cmp_t c)
{
	switch (c) {
	case LE: return GT;
	case LT: return GE;
	case GE: return LT;
	case GT: return LE;
	case EQ: return NE;
	case NE: return EQ;
	}
	unreachable();
}

/* Definition of 'expr2' terms and 'form2' formulas. */

struct expr2;
struct form2;

/* A 'form2' formula is either:
 * - prop2: two expr2 expressions compared using a cmp_t
 * - lbop2: logical binary connective of n form2 values: AND, OR
 * - lneg2: logical negation of a form2
 */

struct prop2 { cmp_t cmp; sptr<expr2> left, right; };
struct lbop2 { enum { AND, OR } op; vec<form2> args; };
struct lneg2 { sptr<form2> arg; };

inline const char *lbop_s[] = { "and", "or" };

struct form2 : sumtype<prop2,lbop2,lneg2>
             , std::enable_shared_from_this<form2> {

	using sumtype<prop2,lbop2,lneg2>::sumtype;
};

/* An 'expr2' expression is either:
 * - name : identifier, same as in 'expr'
 * - bop2 : binary arithmetic operation, same as 'bop' in 'expr'
 * - uop2 : unary arithmetic operation, same as 'uop' in 'expr'
 * - cnst2: disjoint union of integers (kay::Z), rationals (kay::Q) and strings
 * - ite2 : if-then-else: condition (form2) and two expr2 expressions
 *
 * Note that in contrast to 'expr' expressions, there is no 'call' operator.
 * Another difference to 'expr' is 'cnst2': numeric literals have been
 * interpreted (and rationals potentially reduced).
 */

struct ite2 { form2 cond; sptr<expr2> yes, no; };
struct bop2 { decltype(bop::op) op; sptr<expr2> left, right; };
struct uop2 { decltype(uop::op) op; sptr<expr2> operand; };
struct cnst2 { sumtype<kay::Z,kay::Q,str> value; };

struct expr2 : sumtype<name,bop2,uop2,cnst2,ite2>
             , std::enable_shared_from_this<expr2> {

	using sumtype<name,bop2,uop2,cnst2,ite2>::sumtype;
};

template <typename... Ts>
static inline sptr<expr2> make2e(Ts &&... ts)
{
	return std::make_shared<expr2>(std::forward<Ts>(ts)...);
}

template <typename... Ts>
static inline sptr<form2> make2f(Ts &&... ts)
{
	return std::make_shared<form2>(std::forward<Ts>(ts)...);
}

/* Evaluate known function symbols in 'funs' that occur as a 'call' application
 * in the expr 'e'. Results in a expr2 term. */
expr2 unroll(const expr &e, const hmap<str,fun<expr2(vec<expr2>)>> &funs);

}
