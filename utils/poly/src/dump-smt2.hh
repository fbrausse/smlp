
#pragma once

#include "domain.hh"

namespace smlp {

void dump_smt2(FILE *f, const form2 &e);
void dump_smt2(FILE *f, const expr2 &e);
void dump_smt2(FILE *f, const domain &d);

}
