/*
 * This file is part of smlprover.
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 * See the LICENSE file for terms of distribution.
 */

#ifndef CHECK_DATA_LIB_H
#define CHECK_DATA_LIB_H

#include <smlp/table.h>
#include <smlp/response.h>

#undef strtok_r /* some old glibc provides a macro, which breaks our use-case
                 * and the C standard */

struct check_data_radius {
	float abs, rel; /* NAN: n/a */
};

struct check_data {
	struct smlp_table data;
	struct check_data_radius *radii; /* array of size safe_n_cols */
	size_t *safe2spec;
	ssize_t *safe2data;
	size_t safe_n_cols;
	struct smlp_response resp;
};

#define CHECK_DATA_INIT \
	{ SMLP_TABLE_INIT, NULL, NULL, NULL, 0, SMLP_RESPONSE_INIT, }

enum {
	CHECK_DATA_ERR_SUCCESS = 0,
	CHECK_DATA_ERR_RESPONSE_DATA_MISMATCH,
	CHECK_DATA_ERR_JSON,
	CHECK_DATA_ERR_SPEC_FORMAT,
	CHECK_DATA_ERR_SAFE2SPEC,
	CHECK_DATA_ERR_SAFE2DATA,
};

struct check_data_idxbuf {
	struct check_data_idxbuf_entry {
		size_t idx;
		double obj_val;
	} *data;
	size_t n, sz;
};

#define CHECK_DATA_IDXBUF_INIT { NULL, 0, 0 }

/* returns 0 on success, -errno otherwise */
int check_data_idxbuf_ensure_left(struct check_data_idxbuf *b, size_t n);

/* returns 0 on success, -errno otherwise */
int check_data_idxbuf_append(struct check_data_idxbuf *b, size_t idx,
                             double obj_val);

void check_data_idxbuf_fini(struct check_data_idxbuf *b);

int check_data_init(struct check_data *cd, const char *response,
                    const float norm_resp_minmax[2],
                    FILE *f_spec, FILE *f_data,
                    size_t n, const char *safe_labels[STATIC(n)]);
void check_data_fini(struct check_data *d);

/* size of norm_resp_minmax is 2 times the number of free variables in response */
struct check_data * check_data_create(const char *response,
                                      const char *spec_path,
                                      const char *data_path,
                                      const float norm_resp_minmax[2],
                                      size_t n,
                                      const char *safe_labels[STATIC(n)]);
void check_data_destroy(struct check_data *d);

double check_data_compute_objective(const struct check_data *cd,
                                    const float *data_row);

/* returns how many data points fail the test or a negated CHECK_DATA_ERR_*
 * otherwise. ok_p, fail_idcs and ok_idcs may be NULL */
size_t check_data_check(const struct check_data *cd, size_t *ok_p,
                        const float centers[STATIC(cd->safe_n_cols)],
                        double threshold,
                        struct check_data_idxbuf *fail_idcs,
                        struct check_data_idxbuf *ok_idcs);

#endif
