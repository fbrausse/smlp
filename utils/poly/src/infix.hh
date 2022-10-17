/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "expr.hh"

namespace smlp {

expr parse_infix(FILE *f, bool python_compat);
expr parse_infix(str s, bool python_compat);

}
