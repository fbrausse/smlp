/*
 * This file is part of smlprover.
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 * See the LICENSE file for terms of distribution.
 */

#ifndef TABLE_H
#define TABLE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <smlp/common.h>

#include <stdio.h>	/* FILE */

struct smlp_table {
	size_t n_rows, n_cols;
	const char **column_labels;
	char *_column_label_data;
	float *data;
};

#define SMLP_TABLE_INIT	{ 0, 0, NULL, NULL, NULL }

void    smlp_table_fini(struct smlp_table *c);
ssize_t smlp_table_col_idx(const struct smlp_table *c, const char *label);
void    smlp_table_col_idcs(const struct smlp_table *c, size_t n,
                            const char *const labels[STATIC(n)], size_t *idcs);

void    smlp_table_col_get_minmax(const struct smlp_table *c, size_t col,
                                  float *min, float *max);
int     smlp_table_col_norm_minmax(const struct smlp_table *c, size_t col,
                                   float min, float max);

/* CSV I/O */

int     smlp_table_read_header(struct smlp_table *c, FILE *f, int field_sep);
int     smlp_table_read_body(struct smlp_table *c, FILE *f, int field_sep);

static inline int smlp_table_read(struct smlp_table *c, FILE *f,
                                  int read_header)
{
	if (read_header) {
		int r = smlp_table_read_header(c, f, ',');
		if (r < 0)
			return r;
	}
	return smlp_table_read_body(c, f, ',');
}

int smlp_table_csv_write_header(const struct smlp_table *c, FILE *f,
                                int field_sep);

static inline float * smlp_table_data_row(const struct smlp_table *c,
                                          size_t row)
{
	return c->data + c->n_cols * row;
}

#ifdef __cplusplus
}
#endif

#endif
