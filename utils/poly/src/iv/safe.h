/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2020 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2020 The University of Manchester
 */

#ifndef IV_SAFE_H
#define IV_SAFE_H

#include <stddef.h>	/* size_t */
#include <unistd.h>	/* ssize_t */
#include <stdint.h>	/* uint32_t */
#include <stdio.h>	/* FILE */
#include <signal.h>	/* sig_atomic_t */

#include <flint/fmpz.h>	/* fmpz */
#include <flint/fmpq.h>	/* fmpq */

#include <smlp/common.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	double lo, hi;
} iv_ival;

#if 0
/* --------------
 * function terms
 * -------------- */

typedef struct iv_fterm iv_fterm;

typedef enum {                  /* A B: any, S: scalar real, V: real vector space */
	IV_FTERM_PW,            /* V->V: vector of scalar functions, apply point-wise */
	IV_FTERM_PW1,           /* V->V: apply scalar function for every comp of a vector */
	IV_FTERM_AFFINE1,       /* S->S: affine scalar function a*x+b */
	IV_FTERM_COMPOSITION,   /* A->B: composition of n (different) functions */
	IV_FTERM_CLAMP01,       /* S->S: clamp argument to [0,1] */
	IV_FTERM_LAYER,         /* V->V: composition of an AFFINE_MATRIX and one of ACT_ID or ACT_RELU */
	IV_FTERM_AFFINE_MATRIX, /* V->V: A*x + b where A is a matrix and b is a vector */
	IV_FTERM_ACT_ID,        /* S->S: identity */
	IV_FTERM_ACT_RELU,      /* S->S: relu(x) = max(x,0) */
	IV_FTERM_OPT,           /* A->A: either identity or another A->A function */
	IV_FTERM_OBJ_PROJ0,     /* V->S: x[0] */
	IV_FTERM_OBJ_SUB01,     /* V->S: x[0]-x[1] */
} iv_fterm_type;

/* neg on error */
iv_fterm_type iv_fterm_get_type(const iv_fterm *fterm);

const iv_fterm * iv_fterm_pw_get(const iv_fterm *pw, size_t n);

const iv_fterm * iv_fterm_pw1_get(const iv_fterm *pw1);

/* neg on error */
int iv_fterm_affine1_getf(const iv_fterm *affine1, float *a, float *b);
int iv_fterm_affine1_getd(const iv_fterm *affine1, double *a, double *b);

/* neg on error */
ssize_t iv_fterm_composition_size(const iv_fterm *composition);
const iv_fterm * iv_fterm_composition_nth(const iv_fterm *composition, size_t n);

const iv_fterm * iv_fterm_layer_get_affine(const iv_fterm *layer);
const iv_fterm * iv_fterm_layer_get_activation(const iv_fterm *layer);

/* neg on error */
int iv_fterm_affine_matrix_get_dims(const iv_fterm *matrix,
                                    size_t *width, size_t *height);
int iv_fterm_affine_matrix_getf_row(const iv_fterm *matrix, size_t i, float *row);
int iv_fterm_affine_matrix_getf_col(const iv_fterm *matrix, size_t j, float *col);

int iv_fterm_opt_get(const iv_fterm *opt, const iv_fterm **res);

ssize_t iv_fterm_input_dim(const iv_fterm *fterm);
ssize_t iv_fterm_output_dim(const iv_fterm *fterm);
int iv_fterm_eval(const iv_fterm *fterm, const iv_ival *x, iv_ival *y);

void iv_fterm_destroy(iv_fterm *fterm);
#endif

/* -----------------
 * modelled function
 * ----------------- */

typedef struct iv_model_fun iv_model_fun;

/* DONE */
void iv_model_fun_destroy(iv_model_fun *);

/* DONE */
iv_model_fun * iv_model_fun_create(const char *gen_path,
                                   const char *keras_hdf5_path,
                                   const char *spec_path,
                                   const char *io_bounds, char **err);

/* DONE */
size_t iv_mf_input_dim(const iv_model_fun *mf);
const char * iv_mf_dom_label(const iv_model_fun *mf, size_t i, size_t *len);

typedef struct {
	const iv_model_fun *mf;
	int clamp_inputs;
	const char *out_bounds;
} iv_target_function;

/* iv_target_function represents a composition of these: */
#if 0
/* AFFINE1 */
const iv_fterm * iv_tf_in_scaler(const iv_target_function *f);
/* OPT ( CLAMP01 ) */
const iv_fterm * iv_tf_clamp01(const iv_target_function *f);
/* COMPOSITION of n LAYERs */
const iv_fterm * iv_tf_model(const iv_target_function *f);
/* AFFINE1 */
const iv_fterm * iv_tf_out_scaler(const iv_target_function *f);
/* OPT_PROJ0 or OPT_SUB01 */
const iv_fterm * iv_tf_objective(const iv_target_function *f);
/* OPT ( PW_AFFINE ) */
const iv_fterm * iv_tf_obj_scaler(const iv_target_function *f);
#endif

/* DONE */
void iv_tf_eval_ival(const iv_target_function *f, const iv_ival *x, iv_ival *y);

/* ----
 * grid
 * ---- */

/* all DONE */
int     iv_model_fun_has_grid(const iv_model_fun *f);
ssize_t iv_model_fun_grid_get_rank(const iv_model_fun *f);
int     iv_model_fun_grid_get_dims(const iv_model_fun *f, size_t *dims);
ssize_t iv_model_fun_grid_size(const iv_model_fun *f);

int     iv_model_fun_grid_sget_ival(const iv_model_fun *f, size_t grid_idx,
                                    iv_ival v[STATIC(iv_model_fun_grid_get_rank(f))]);

typedef union {
	fmpz_t z;
	fmpq_t q;
	size_t idx;
} iv_elem;

int     iv_model_fun_grid_get(const iv_model_fun *f, fmpz_t grid_idx,
                              iv_elem v[STATIC(iv_model_fun_grid_get_rank(f))]);

typedef struct {
	uint32_t x_idx;
	float y_lo;
} iv_grid_xy;

#if 0
/* Assumes iv_model_fun source for f has a grid.
 * Evaluates f on the grid and returns the results as an array of size *n.
 * A pair (x,y) is considered to be a result if
 * - res_only_here == NULL, or
 * - y in *res_only_here.
 * If res_range != NULL, it is set to the interval hull of *res_range and all
 * values evaluated on the grid. At the end, the results are sorted by the
 * value lo(y) in descending order. */
iv_grid_xy * iv_model_fun_eval_grid_32_sorted_lo(const iv_target_function *f,
                                                 size_t *n, iv_ival *res_range,
                                                 const iv_ival *res_only_here);
#endif

/* DONE */
int iv_tf_eval_grid(const iv_target_function *f,
                    void (*handle_next)(double lo, double hi, void *udata),
                    void *udata);

/* DONE */
int iv_tf_eval_dataset(const iv_target_function *f, FILE *dataset,
                       void (*handle_res)(double lo, double hi, void *udata),
                       void *udata);

/* DONE */
int iv_tf_region_eval(const iv_target_function *f,
                      const iv_ival c[STATIC(iv_model_fun_grid_get_rank(f))],
                      bool (*split_dom)(double lo, double hi, void *udata),
                      void *udata);


/* same as above but requires rework of Table

struct table;

int iv_model_fun_eval_table(const iv_target_function *f, const struct table *t,
                            void (*handle_res)(iv_ival res, void *udata),
                            void *udata);
*/

/* ----------------
 * threshold search
 * ---------------- */

/* Computes closer outer (sup) and inner (sub) approximations of the function's
 * range until
 * (*) 'sub->lo - sup->lo <= tgt_width' or 'threshold' not in [sup->lo,sub->lo]
 * starting with an output precision of 'start_prec' decreased exponentially.
 *
 * If 'threshold' is to be ignored, pass INFINITY for this parameter.
 * If 'tgt_width' is to be ignored, pass INFINITY for this parameter.
 * If 'signalled' is given, set *signalled to non-zero to stop the computation.
 *
 * Returns
 *    +1 if  valid && correct
 *     0 if  valid
 *    -1 if !valid
 * where 'valid' is true if sub is a non-empty inner approximation of the
 * function's range and 'correct' is true if the condition (*) above holds.
 *
 * If 'signalled' is NULL or *signalled is never set, this function will never
 * return -1, however, it might not terminate if 'threshold < INFINITY' and is
 * close to the lower boundary of the function's range.
 */
/* DONE */
int iv_tf_search_supsub_min(const iv_target_function *f, const iv_ival *x,
                            double start_prec, double tgt_width,
                            double threshold,
                            const volatile sig_atomic_t *signalled,
                            iv_ival *sup, iv_ival *sub);

#ifdef __cplusplus
}
#endif

#endif
