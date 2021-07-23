/*
 * This file is part of smlprover.
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 * See the LICENSE file for terms of distribution.
 */

#include <smlp/check-data-lib.h>
#include <smlp/cbuf.h>

#include <kjson.h>

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <unistd.h>	/* getopt(3p) */
#include <math.h>	/* fabs(3) */

static float parse_rad(int is_int, const struct kjson_value *rad)
{
	if (!rad)
		return NAN;
	if (is_int)
		assert(rad->type == KJSON_VALUE_NUMBER_INTEGER);
	float r = rad->type == KJSON_VALUE_NUMBER_INTEGER ? rad->i
	        : rad->type == KJSON_VALUE_NUMBER_DOUBLE  ? rad->d
	        : INFINITY;
	assert(!isfinite(r) || r >= 0);
	return r;
}

static void spec_cols2safe(size_t *cols, char *is_int,
                           const struct smlp_table *safe,
                           const struct kjson_value *spec,
                           struct check_data_radius *radii)
{
	assert(spec->type == KJSON_VALUE_ARRAY);
	const struct kjson_array *l = &spec->a;
	for (size_t i=0; i<l->n; i++) {
		const struct kjson_value *s = &l->data[i];
		const struct kjson_value *label = NULL, *type = NULL, *rad_rel = NULL, *rad_abs = NULL;
		assert(s->type == KJSON_VALUE_OBJECT);
		for (size_t j=0; j<s->o.n; j++)
			if (!strcmp(s->o.data[j].key.begin, "label"))
				label = &s->o.data[j].value;
			else if (!strcmp(s->o.data[j].key.begin, "type"))
				type = &s->o.data[j].value;
			else if (!strcmp(s->o.data[j].key.begin, "rad-rel"))
				rad_rel = &s->o.data[j].value;
			else if (!strcmp(s->o.data[j].key.begin, "rad-abs"))
				rad_abs = &s->o.data[j].value;
		assert(label);
		assert(label->type == KJSON_VALUE_STRING);
		ssize_t idx = smlp_table_col_idx(safe, label->s.begin);
		assert(type);
		assert(type->type == KJSON_VALUE_STRING);/*
		fprintf(stderr, "label '%s', type '%s' -> safe idx %zd\n",
		        json_object_get_string(label),
		        json_object_get_string(type),
		        idx);*/
		int is_resp = !strcmp("response", type->s.begin);
		int is_cat  = !strcmp("category", type->s.begin);
		if ((idx < 0) != (is_resp || is_cat))
			fprintf(stderr, "error: %s: idx: %zd, is response: %d, is cat: %d\n",
			        label->s.begin, idx, is_resp, is_cat);
		assert((idx < 0) == (is_resp || is_cat));
		if (is_resp || is_cat || idx < 0)
			continue;
		cols[idx] = i;
		is_int[idx] = !strcmp(type->s.begin, "int");
		if (!rad_abs && !rad_rel) {
			fprintf(stderr, "error: %s: neither 'rad-rel' nor 'rad-abs' entry\n",
			        label->s.begin);
			continue;
		}
		radii[idx].abs = parse_rad(is_int[idx], rad_abs);
		radii[idx].rel = parse_rad(is_int[idx], rad_rel);/*
		fprintf(stderr, "%s: i: %zd -> idx: %zu, rad-abs: %g, rad-rel: %g\n",
		        label->s.begin, i, idx, radii[idx].abs, radii[idx].rel);*/
	}
}

void check_data_fini(struct check_data *d)
{
	smlp_table_fini(&d->data);
	free(d->radii);
	free(d->safe2spec);
	free(d->safe2data);
}

static ssize_t parse_response_idx(const char *label, void *udata)
{
	struct smlp_table *t = udata;
	return smlp_table_col_idx(t, label);
}

static int parse_response(struct check_data *cd, const char *response)
{
	int r = smlp_response_parse(&cd->resp, response, parse_response_idx,
	                            &cd->data);
	if (r)
		return CHECK_DATA_ERR_RESPONSE_DATA_MISMATCH;
	return 0;
}

int check_data_init(struct check_data *cd, const char *response,
                    const float resp_norm_minmax[2], FILE *f_spec, FILE *f_data,
                    size_t n, const char *safe_labels[static n])
{
	struct smlp_cbuf specbuf = SMLP_CBUF_INIT;
	struct kjson_value spec = KJSON_VALUE_INIT;
	int r = 0;
	if (!cd || !response || !f_spec || !f_data || !safe_labels)
		return -EINVAL;

	cd->safe_n_cols = n;
	char is_int[n];

	const struct smlp_table safe = {
		.n_rows = 0, .n_cols = n, .column_labels = safe_labels,
	};

	if ((r = smlp_table_read(&cd->data, f_data, 1)))
		goto fail;

	if ((r = parse_response(cd, response)))
		goto fail;

	{
		unsigned n_resps;
		switch (cd->resp.type) {
		case SMLP_RESPONSE_ID: n_resps = 1; break;
		case SMLP_RESPONSE_SUB: n_resps = 2; break;
		default: assert(0);
		}
		float minmax[2*n_resps];
		if (!resp_norm_minmax) {
			for (unsigned i=0; i<n_resps; i++)
				smlp_table_col_get_minmax(&cd->data,
				                          cd->resp.idx[i],
				                          minmax+2*i+0,
				                          minmax+2*i+1);
			resp_norm_minmax = minmax;
		}
		for (unsigned i=0; i<n_resps; i++)
			smlp_table_col_norm_minmax(&cd->data, cd->resp.idx[i],
			                           resp_norm_minmax[2*i+0],
			                           resp_norm_minmax[2*i+1]);
	}

	if ((r = smlp_cbuf_init_file(&specbuf, f_spec)))
		goto fail;

	if (!kjson_parse(&(struct kjson_parser){ specbuf.cstr.buf }, &spec) ||
	    spec.type != KJSON_VALUE_ARRAY) {
		r = CHECK_DATA_ERR_JSON;
		goto fail;
	}

	cd->radii     = malloc(n * sizeof(*cd->radii));
	cd->safe2spec = malloc(n * sizeof(*cd->safe2spec));
	cd->safe2data = malloc(n * sizeof(*cd->safe2data));
	memset(cd->safe2spec, -1, n * sizeof(*cd->safe2spec));
	memset(cd->safe2data, -1, n * sizeof(*cd->safe2data));
	memset(is_int, -1, sizeof(n));
	for (size_t i=0; i<n; i++)
		cd->radii[i].abs = cd->radii[i].rel = NAN;
	spec_cols2safe(cd->safe2spec, is_int, &safe, &spec, cd->radii);
	for (size_t i=0; i<safe.n_cols; i++) {
		// fprintf(stderr, "%zu -> %zd\n", i, spec2safe[i]);
		if (!(cd->safe2spec[i] < (size_t)-1)) {
			r = CHECK_DATA_ERR_SAFE2SPEC;
			goto fail;
		}
	}
	for (size_t i=0; i<n; i++)
		assert(isfinite(cd->radii[i].abs) || isfinite(cd->radii[i].rel));

	smlp_table_col_idcs(&safe, cd->data.n_cols, cd->data.column_labels,
	                    (size_t *)cd->safe2data);

	for (size_t i=0; i<safe.n_cols; i++) {
		// fprintf(stderr, "%zu -> %zd\n", i, spec2safe[i]);
		if (!(cd->safe2data[i] < (size_t)-1)) {
			r = CHECK_DATA_ERR_SAFE2DATA;
			goto fail;
		}
		assert(!strcmp(safe.column_labels[i],
		               cd->data.column_labels[cd->safe2data[i]]));
		if (is_int[i] < 0) {
			r = CHECK_DATA_ERR_SAFE2SPEC;
			goto fail;
		}
		if (is_int[i])
			cd->safe2data[i] = ~cd->safe2data[i];
	}

done:
	kjson_value_fini(&spec);
	smlp_cbuf_fini(&specbuf);
	return r;
fail:
	check_data_fini(cd);
	goto done;
}

void check_data_destroy(struct check_data *d)
{
	check_data_fini(d);
	free(d);
}

struct check_data * check_data_create(const char *response,
                                      const char *spec_path,
                                      const char *data_path,
                                      const float norm_resp_minmax[2],
                                      size_t n,
                                      const char *safe_labels[static n])
{
	int r = 0;
	FILE *f_spec = NULL, *f_data = NULL;
	struct check_data *cd = NULL;

	if (!response || !spec_path || !data_path || !n || !safe_labels) {
		r = -EINVAL;
		goto fail;
	}

	if (!(f_spec = fopen(spec_path, "r")) ||
	    !(f_data = fopen(data_path, "r")) ||
	    !(cd = calloc(1, sizeof(*cd))))
		goto fail_errno;

	static thread_local char BUF[1 << 18];
	setvbuf(f_data, BUF, _IOFBF, sizeof(BUF));
	r = check_data_init(cd, response, norm_resp_minmax, f_spec, f_data, n, safe_labels);
done:
	if (f_spec)
		fclose(f_spec);
	if (f_data)
		fclose(f_data);
	if (r) {
		free(cd);
		cd = NULL;
	}
	return cd;

fail_errno:
	r = -errno;
fail:
	goto done;
}

static void spec_radii(float *radii, const struct check_data *cd,
                       const float c[static cd->safe_n_cols])
{
	size_t n = cd->safe_n_cols;
	for (size_t i=0; i<n; i++) {
		const struct check_data_radius *rad = cd->radii + i;
		float r;
		if (isfinite(rad->rel) && c[i])
			r = fabs(c[i]) * rad->rel;
		else
			r = isfinite(rad->abs) ? rad->abs : rad->rel;
		/*
		fprintf(stderr, "rad around col %zu value %g: %g where abs: %g, rel: %g\n",
		        i, c[i], r, rad->abs, rad->rel);*/
		radii[i] = r;
	}
}

int check_data_idxbuf_ensure_left(struct check_data_idxbuf *b, size_t n)
{
	size_t nn0 = b->n + n;
	if (nn0 <= b->sz)
		return 0;
	size_t nn1 = 2*b->sz, nn = nn0 > nn1 ? nn0 : nn1;
	if (nn < b->sz)
		return -ERANGE;
	void *dat = realloc(b->data, sizeof(*b->data) * nn);
	if (!dat)
		return -errno;
	b->sz = nn;
	b->data = dat;
	return 0;
}

void check_data_idxbuf_fini(struct check_data_idxbuf *b)
{
	free(b->data);
}

int check_data_idxbuf_append(struct check_data_idxbuf *b, size_t idx,
                             double obj_val)
{
	int r = check_data_idxbuf_ensure_left(b, 1);
	if (!r) {
		struct check_data_idxbuf_entry *e = &b->data[b->n++];
		e->idx = idx;
		e->obj_val = obj_val;
	}
	return r;
}

double check_data_compute_objective(const struct check_data *cd,
                                    const float *data_row)
{
	assert(cd->resp.type == SMLP_RESPONSE_ID ||
	       cd->resp.type == SMLP_RESPONSE_SUB);
	double r = data_row[cd->resp.idx[0]];
	if (cd->resp.type == SMLP_RESPONSE_SUB)
		r -= data_row[cd->resp.idx[1]];
	return r;
}

static int in_ball_float(float d, float c, float r)
{
	return fabs(d-c) <= r;
}

static int in_ball_int(int d, int c, float r)
{
	int v = d - c;
	return -r <= v && v < r;
}

static int in_ball(float d, float c, float r, int is_int)
{
	if (is_int) {
		assert((int)d == d);
		assert((int)c == c);
		return in_ball_int(d, c, r);
	} else
		return in_ball_float(d, c, r);
}

size_t check_data_check(const struct check_data *cd, size_t *ok_p,
                        const float c[static cd->safe_n_cols],
                        double threshold,
                        struct check_data_idxbuf *fail_idcs,
                        struct check_data_idxbuf *ok_idcs)
{
	size_t ok = 0, fail = 0;
	size_t M = cd->safe_n_cols;
	float r[M];
	spec_radii(r, cd, c);
	for (size_t j=0; j<cd->data.n_rows; j++) {
		const float *d = smlp_table_data_row(&cd->data, j);
		size_t k;
		for (k=0; k<M; k++) {
			ssize_t idx = cd->safe2data[k];
			int is_int = 0;
			if (idx < 0)
				idx = ~idx, is_int = 1;
			if (!in_ball(d[idx], c[k], r[k], is_int))
				break;
		}
		if (k < M)
			continue;
		double val = check_data_compute_objective(cd, d);
		if (val < threshold) {
			if (fail_idcs)
				check_data_idxbuf_append(fail_idcs, j, val);
			fail++;
		} else {
			if (ok_idcs)
				check_data_idxbuf_append(ok_idcs, j, val);
			ok++;
		}
	}
	if (ok_p)
		*ok_p = ok;
	return fail;
}
