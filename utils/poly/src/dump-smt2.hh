/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "domain.hh"

namespace smlp {

/* Functions to print 'form2' formulas, 'expr2' expressions and 'domain'
 * constraints in SMT-LIB2 format.
 */

void dump_smt2(FILE *f, const form2 &e);
void dump_smt2(FILE *f, const expr2 &e);
void dump_smt2(FILE *f, const domain &d);

}
