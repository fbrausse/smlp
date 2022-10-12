
#pragma once

#include "expr.hh"

namespace smlp {

void dump_pe(FILE *f, const expr &e);
expr parse_pe(FILE *f);

}
