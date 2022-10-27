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
	virtual void add(const form2 &f) = 0;
	virtual result check() = 0;
};

}
