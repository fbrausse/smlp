/*
 * This file is part of smlprover.
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 * See the LICENSE file for terms of distribution.
 */

#include <smlp/table.h>

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>	/* fmin(), fmax() */
#include <inttypes.h>	/* uintmax_t */

#undef strtok_r /* some old glibc provides a macro, which breaks our use-case
                 * and the C standard */

#define MAX(a,b)	((a) > (b) ? (a) : (b))

static ssize_t getlinec(char **line, size_t *sz, FILE *f)
{
	ssize_t rd = getline(line, sz, f);
	if (rd > 0 && (*line)[rd-1] == '\n')
		(*line)[--rd] = '\0';
	if (rd > 0 && (*line)[rd-1] == '\r')
		(*line)[--rd] = '\0';
	return rd;
}

void smlp_table_fini(struct smlp_table *c)
{
	free(c->column_labels);
	free(c->_column_label_data);
	free(c->data);
	memset(c, 0, sizeof(*c));
}

ssize_t smlp_table_col_idx(const struct smlp_table *c, const char *label)
{
	for (size_t i=0; i<c->n_cols; i++)
		if (!strcmp(c->column_labels[i], label))
			return i;
	fprintf(stderr, "column with label '%s' not found\n", label);
	return -1;
}

void smlp_table_col_idcs(const struct smlp_table *c, size_t n,
                         const char *const labels[static n], size_t *idcs)
{
	for (size_t i=0; i<n; i++) {
		ssize_t idx = smlp_table_col_idx(c, labels[i]);
		if (idx >= 0)
			idcs[idx] = i;
	}
}

int smlp_table_csv_write_header(const struct smlp_table *c, FILE *f,
                                int field_sep)
{
	for (size_t i=0; i<c->n_cols; i++)
		if (fprintf(f, "%s%s", i ? (char[]){ field_sep, '\0' } : "",
		            c->column_labels[i]) < 0)
			return -errno;
	if (fprintf(f, "\n") < 0)
		return -errno;
	return 0;
}

int smlp_table_read_header(struct smlp_table *c, FILE *f, int field_sep)
{
	char *line = NULL;
	size_t sz = 0;
	if (getlinec(&line, &sz, f) < 0)
		return -errno;
	size_t i = 0;
	size_t cl_sz = 0;
	for (char *last = line;;) {
		if (i+1 > cl_sz)
			c->column_labels = realloc(c->column_labels,
			                           sizeof(*c->column_labels) *
			                           MAX(i+1,2*cl_sz));
		c->column_labels[i] = last;
		char *next = strchr(last, field_sep);
		i++;
		if (!next)
			break;
		*next = '\0', last = next+1;
	}
	c->n_cols = i;
	c->_column_label_data = line;
	line = NULL, sz = 0;
	return 0;
}

#if 0
static float my_atof(const char *s)
{
	const char *s0 = s;
	int sgn = 1;
	if (*s == '-')
		sgn = 0, s++;
	else if (*s == '+')
		sgn++;
	uintmax_t r = 0;
	const char SYM[] = "0123456789";
	for (;; s++) {
		const char *p = strchr(SYM, *s);
		if (!p)
			return atof(s0);
		int k = p - SYM;
		if (k == 10)
			break;
		uintmax_t t = r * 10 + k;
		if (t < r)
			return atof(s0);
		r = t;
	}
	return sgn ? r : -r;
}
#else
static float my_atof(const char *s)
{
	const char *s0 = s;
	int sgn = 1;
	if (*s == '-')
		sgn = 0, s++;
	else if (*s == '+')
		sgn++;
	uintmax_t r = 0;
	for (; *s; s++) {
		if (*s < '0' || *s > '9')
			return atof(s0);
		uintmax_t t = r * 10 + (*s - '0');
		if (t < r)
			return atof(s0);
		r = t;
	}
	return sgn ? r : -(float)r;
}
#endif

int smlp_table_read_body(struct smlp_table *c, FILE *f, int field_sep)
{
	char *line = NULL;
	size_t sz = 0;
	size_t data_rows_sz = 0;
	ssize_t rd;
	for (; (rd = getline(&line, &sz, f)) > 0; c->n_rows++) {
		if (c->n_rows + 1 > data_rows_sz)
			c->data = realloc(c->data,
			                  sizeof(*c->data) * c->n_cols *
			                  (data_rows_sz = MAX(c->n_rows+1,
			                                      2*data_rows_sz)));
		char *save;
		size_t i = 0;
		float *d = smlp_table_data_row(c, c->n_rows);
		for (char *tok = strtok_r(line, (char[]){field_sep,'\0'}, &save); tok;
		     i++, tok = strtok_r(NULL, (char[]){field_sep,'\0'}, &save)) {
			assert(i < c->n_cols);
			d[i] = my_atof(tok);
		}
	}
	free(line);
	return rd == -1 ? feof(f) ? 0 : -ferror(f) : 0;
}

void smlp_table_col_get_minmax(const struct smlp_table *c, size_t col,
                               float *min, float *max)
{
	float ymin = c->data[col], ymax = c->data[col];
	for (size_t i=1; i<c->n_rows; i++) {
		float f = smlp_table_data_row(c, i)[col];
		ymin = fmin(ymin, f), ymax = fmax(ymax, f);
	}
	if (min)
		*min = ymin;
	if (max)
		*max = ymax;
}

int smlp_table_col_norm_minmax(const struct smlp_table *c, size_t col,
                               float ymin, float ymax)
{
	if (ymin < ymax) {
		for (size_t i=0; i<c->n_rows; i++) {
			float *y = smlp_table_data_row(c, i) + col;
			*y = (*y - ymin) / (ymax - ymin);
		}
		return 1;
	}
	return 0;
}
