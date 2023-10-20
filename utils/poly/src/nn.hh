
#pragma once

#include "domain.hh"

namespace smlp {

typedef fun<sptr<form2>(opt<kay::Q>, const hmap<str,sptr<term2>> &)> theta_t;

/* optimize T in obj_range such that (assuming direction is >=):
 *
 * E x . eta x /\
 * A y . theta x y -> alpha y -> (beta y /\ obj y >= T)
 *
 * domain constraints from 'dom' have to hold for x and y.
 */

/* eta  : constraints on candidates; for instance: a grid
 * theta: stability region
 * alpha: constraint on whether a particular y is considered eligible as a counter-example
 * beta : additional constraint valid solutions have to satisfy; if not, y is a counter-example
 *
 * delta: is a real constant, that is used to increase the radius around counter-examples
 *
 * epsilon: is a real constant used as modifier(?) for the thresholds T that allows
 *          to prove completeness of the algorithm
 */

struct pre_problem {
	domain dom;
	sptr<term2> obj; /* objective */
	hmap<str,sptr<term2>> funcs; /* named responses / outputs */
	hmap<str,ival> func_bounds;
	hmap<str,ival> input_bounds; /* alpha */
	sptr<form2> eta = true2; /* corresponds to "safe" list in .spec */
	sptr<form2> partial_domain = true2; /* constraints from evaluating partial functions */
	theta_t theta;

	/* alpha, beta are not part of the pre-problem */

	/* Potentially modifies dom and input_bounds based on params and returns
	 * a formula corresponding to the constraints from the remaining
	 * input_bounds. */
	sptr<form2> interpret_input_bounds(bool bnds_dom, bool inject_reals);
};

pre_problem parse_nn(const char *gen_path, const char *hdf5_path,
                     const char *spec_path, const char *io_bounds,
                     const char *obj_bounds, bool clamp_inputs,
                     bool single_obj);

}
