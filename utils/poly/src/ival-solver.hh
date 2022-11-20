/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "solver.hh"

namespace smlp {

/* Interval evaluation of the term t on the domain dom using double-precision
 * arithmetic with correct rounding. That means the result (if any) is an over-
 * approximation of the true range of t over dom. The result is empty iff the
 * domain is. If it is not empty, the result contains the lower and upper
 * endpoints of an interval enclosure of the true range.
 *
 * Note that the condition of ite2 terms is evaluated using intervals as well
 * and in case it overlaps, the (convex) hull of the values of both branches is
 * taken. This may result in loss of precision additional to that of pure
 * interval arithmetic and that of double-precision arithmetic rounding errors.
 *
 * In case the domain contains 'entire' component ranges, the result may not be
 * bounded. 'list' components are evaluated separately with point intervals and
 * results are combined as the (convex) hull. The above comment about loss of
 * precision applies as well.
 */
opt<pair<double,double>>
dbl_interval_eval(const domain &dom, const sptr<term2> &t);

struct ival_solver : acc_solver {

	ival_solver(size_t max_subdivs = 0, const char *logic = nullptr)
	: max_subdivs(max_subdivs)
	, logic(logic ? opt<str>(str(logic)) : opt<str> {})
	{}

	result check() override;

private:
	size_t max_subdivs;
	opt<str> logic;
};

}
