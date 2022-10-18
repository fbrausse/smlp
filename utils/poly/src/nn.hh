
#pragma once

#include "domain.hh"

namespace smlp {

struct pre_problem {
	domain dom;
	sptr<term2> obj;
	hmap<str,sptr<term2>> funcs;
	sptr<form2> input_constraints = true2;
	sptr<form2> partial_domain = true2;
	fun<sptr<form2>(bool,const hmap<str,sptr<term2>> &)> theta;
};

pre_problem parse_nn(const char *gen_path, const char *hdf5_path,
                     const char *spec_path, const char *io_bounds,
                     const char *out_bounds, bool clamp_inputs,
                     bool single_obj);

}
