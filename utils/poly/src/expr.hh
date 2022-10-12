/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "common.hh"

namespace smlp {

/* Definition of 'expr' as either
 * - name: named constant
 * - call: function call
 * - bop : binary (arithmetic) operation
 * - uop : unary (arithmetic) operation
 * - cnst: literal value
 */

struct expr;

struct name { str id; };
struct call { sptr<expr> func; vec<expr> args; };
struct bop { enum { ADD, SUB, MUL, } op; sptr<expr> left, right; };
struct uop { enum { UADD, USUB, } op; sptr<expr> operand; };
struct cnst { str value; };

inline const char *bop_s[] = { "+", "-", "*" };
inline const char *uop_s[] = { "+", "-" };

struct expr : sumtype<name,call,bop,uop,cnst>
            , std::enable_shared_from_this<expr> {

	using sumtype<name,call,bop,uop,cnst>::sumtype;
};

template <typename... Ts>
static inline sptr<expr> make1e(Ts &&... ts)
{
	return std::make_shared<expr>(std::forward<Ts>(ts)...);
}

} /* end namespace smlp */
