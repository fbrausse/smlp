/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "nn.hh"

namespace smlp {

pre_problem parse_poly_problem(const char *simple_domain_path,
                               const char *poly_expression_path,
                               bool python_compat,
                               bool dump_pe = false,
                               bool infix = true);

}
