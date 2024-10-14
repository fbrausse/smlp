/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "expr.hh"

namespace smlp {

void dump_pe(FILE *f, const expr &e);
expr parse_pe(FILE *f);

}
