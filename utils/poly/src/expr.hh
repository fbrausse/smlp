/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "common.hh"

#include <charconv> /* std::from_chars_result, etc. */

namespace smlp {

/* Definition of 'expr' as either
 * - name: named constant
 * - call: function call
 * - bop : binary (arithmetic) operation
 * - uop : unary (arithmetic) operation
 * - cnst: literal value
 */

struct expr;

/* Symbolic representation of a order/equality predicate over ints/reals */

enum cmp_t { LE, LT, GE, GT, EQ, NE, };

inline const char *cmp_s[] = { "<=", "<", ">=", ">", "==", "!=" };
inline const char *cmp_smt2[] = { "<=", "<", ">=", ">", "=", "distinct" };

/* Some properties of cmp_t */
static inline bool is_less(cmp_t c) { return (unsigned)c <= LT; }
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

template <typename T>
static inline auto do_cmp(const T &l, cmp_t c, const T &r)
{
	switch (c) {
	case LE: return l <  r;
	case LT: return l <= r;
	case GE: return l >  r;
	case GT: return l >= r;
	case EQ: return l == r;
	case NE: return l != r;
	}
	unreachable();
}

static inline std::from_chars_result
from_chars(const char *first, const char *last, cmp_t &v)
{
	for (size_t i=0; i<ARRAY_SIZE(cmp_s); i++) {
		std::string_view s = cmp_s[i];
		if (std::string_view(first, last).starts_with(s)) {
			v = (cmp_t)i;
			return { first + s.length(), std::errc {} };
		}
	}
	return { first, std::errc::invalid_argument };
}

struct name {
	str id;
	bool operator==(const name &b) const;
};
struct call { sptr<expr> func; vec<expr> args; };
struct bop { enum { ADD, SUB, MUL, } op; sptr<expr> left, right; };
struct uop { enum { UADD, USUB, } op; sptr<expr> operand; };
struct cop { cmp_t cmp; sptr<expr> left, right; };
struct cnst { str value; };

inline const char *bop_s[] = { "+", "-", "*" };
inline const char *uop_s[] = { "+", "-" };

struct expr : sumtype<name,call,bop,uop,cop,cnst>
            , std::enable_shared_from_this<expr> {

	using sumtype<name,call,bop,uop,cop,cnst>::sumtype;
};

template <typename... Ts>
static inline sptr<expr> make1e(Ts &&... ts)
{
	return std::make_shared<expr>(std::forward<Ts>(ts)...);
}

} /* end namespace smlp */
