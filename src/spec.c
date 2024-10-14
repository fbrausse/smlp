
#define _POSIX_C_SOURCE	200809L	/* getline */

#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>	/* ssize_t */
#include <inttypes.h>	/* strtoimax */

#include <kjson.h>
#include <smlp/table.h>
#include <smlp/spec.h>


#ifdef _WIN32
# define FMT_SZ	"I64"
#else
# define FMT_SZ "z"
#endif

#ifndef MAX
# define MAX(a,b)	((a) > (b) ? (a) : (b))
#endif

#define ARRAY_SIZE(...)	(sizeof(__VA_ARGS__)/sizeof(*(__VA_ARGS__)))

SMLP_FN_ATTR_PRINTF(1,2)
static void log_error(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}

static int buf_ensure_sz(void **buf, size_t *sz, const size_t *valid, size_t n)
{
	if (*valid + n <= *sz)
		return 0;
	size_t nsz = MAX(2 * *sz, *valid+n);
	void *r = realloc(*buf, nsz);
	if (r) {
		*sz = nsz;
		*buf = r;
	}
	return r ? 0 : -errno;
}

SMLP_FN_ATTR_PRINTF(2,3)
static int bsprintf(char **strp, const char *fmt, ...)
{
	va_list ap, ap2;
	va_start(ap,fmt);
	va_copy(ap2, ap);
	int n = snprintf(NULL, 0, fmt, ap);
	va_end(ap);
	if (n == -1) {
		va_end(ap2);
		return -1;
	}
	*strp = malloc(n+1);
	if (!*strp) {
		va_end(ap2);
		return -1;
	}
	return sprintf(*strp, fmt, ap2);
}

static void * fread_mem(FILE *f, size_t *_sz, size_t *_valid, char **error)
{
	void *content = NULL;
	size_t sz = 0, valid = 0;
	char buf[4096];
	for (size_t rd; (rd = fread(buf, 1, sizeof(buf), f));) {
		int r = buf_ensure_sz(&content, &sz, &valid, rd);
		if (r) {
			free(content);
			bsprintf(error, strerror(-r));
			return NULL;
		}
		memcpy((char *)content + valid, buf, rd);
		valid += rd;
	}
	if (ferror(f)) {
		free(content);
		bsprintf(error, "error reading file\n");
		return NULL;
	}
	if (_sz)
		*_sz = sz;
	if (_valid)
		*_valid = valid;
	return content;
}

#define log_error_and(finally, ...) \
	do { log_error(__VA_ARGS__); finally; } while (0)

int kjson_init_path(struct kjson *sp, const char *path, char **error)
{
	FILE *specf = fopen(path, "r");
	if (!specf)
		return -errno;
	size_t sz, valid;
	char *err = NULL;
	char *spec = fread_mem(specf, &sz, &valid, &err);
	fclose(specf);
	if (!spec) {
		bsprintf(error, "error reading JSON '%s': %s\n", path, err);
		free(err);
		return 1;
	}
	int r = buf_ensure_sz((void *)&spec, &sz, &valid, 1);
	if (r) {
		bsprintf(error, "error allocating buffer: %s\n", strerror(-r));
		return 2;
	}
	spec[valid] = '\0';

	sp->top.type = KJSON_VALUE_NULL;
	sp->src = spec;
	if (kjson_parse(&(struct kjson_parser){ sp->src }, &sp->top))
		return 0;

	bsprintf(error, "error: '%s' does not contain a valid JSON document\n", path);
	free(spec);
	kjson_value_fini(&sp->top);
	return 3;
}

void kjson_fini(struct kjson *sp)
{
	kjson_value_fini(&sp->top);
	free(sp->src);
}

void smlp_spec_entry_fini(struct smlp_spec_entry *e)
{
	free(e->label);
	free(e->cats.concat);
	free(e->cats.data);
	free(e->safe.data);
}

void smlp_spec_fini(struct smlp_spec *table)
{
	for (size_t i=0; i<table->n; i++)
		smlp_spec_entry_fini(table->cols + i);
	free(table->cols);
}

/* assumes e->dtype (and e->cats, if e->dtype == SMLP_DTY_CAT) is initialized */
static bool smlp_value_from_json(union smlp_value *r,
                                 const struct smlp_spec_entry *e,
                                 const struct kjson_value *v,
                                 bool allow_int2dbl)
{
	const char *s;
	switch (v->type) {
	case KJSON_VALUE_STRING:
	case KJSON_VALUE_BOOLEAN:
		if (e->dtype != SMLP_DTY_CAT)
			return false;
		s = v->type == KJSON_VALUE_BOOLEAN ? v->b ? "true" : "false"
		                                   : v->s.begin;
		for (size_t i=0; i<e->cats.n; i++)
			if (!strcmp(e->cats.data[i], s)) {
				r->c = i;
				return true;
			}
		return false;
	case KJSON_VALUE_NUMBER_DOUBLE:
		if (e->dtype != SMLP_DTY_DBL)
			return false;
		r->d = v->d;
		return true;
	case KJSON_VALUE_NUMBER_INTEGER:
		if (e->dtype == SMLP_DTY_INT) {
			r->i = v->i;
			return true;
		} else if (allow_int2dbl && e->dtype == SMLP_DTY_DBL) {
			r->d = v->i;
			return true;
		}
		return false;
	default:
		return false;
	}
}

#define FAIL(code, ...) do { \
		smlp_spec_fini(table); \
		return error && bsprintf(error, __VA_ARGS__) == -1 \
		     ? -errno : code; \
	} while (0)

int smlp_spec_init(struct smlp_spec *table, const struct kjson_value *spec,
                   char **error)
{
	memset(table, 0, sizeof(*table));

	if (spec->type != KJSON_VALUE_ARRAY)
		FAIL(1, "spec format error: top-level is not an array\n");

	table->cols = calloc(spec->a.n, sizeof(*table->cols));
	if (!table->cols)
		FAIL(2, "error allocating table columns: %s", strerror(errno));

	struct smlp_spec_entry *c = table->cols;
	for (size_t i=0; i<spec->a.n; i++, c++) {
		struct kjson_value *e = &spec->a.data[i];
		if (e->type != KJSON_VALUE_OBJECT)
			FAIL(3, "error: element #%" FMT_SZ "u of array is not a "
			        "JSON object\n", i);

		enum smlp_dtype type = -1;
		enum smlp_purpose purp = -1;
		struct kjson_string *label = NULL;
		struct kjson_array *cats = NULL;
		struct kjson_value *rad_rel = NULL, *rad_abs = NULL, *def_val = NULL, *safe = NULL;
		for (size_t j=0; j<e->o.n; j++) {
			struct kjson_object_entry *v = &e->o.data[j];
			if (!strcmp(v->key.begin, "label")) {
				if (v->value.type != KJSON_VALUE_STRING)
					FAIL(4, "error: 'label' value of "
					        "object #%" FMT_SZ "u in array is "
					        "not a JSON string\n", i);

				label = &v->value.s;
			} else if (!strcmp(v->key.begin, "type")) {
				if (v->value.type != KJSON_VALUE_STRING)
					FAIL(5, "error: 'type' value of "
					        "object in element #%" FMT_SZ "u "
					        "of .spec array is not a JSON "
					        "string\n", i);
				if (!strcmp(v->value.s.begin, "knob"))
					purp = SMLP_PUR_CONFIG;
				else if (!strcmp(v->value.s.begin, "input"))
					purp = SMLP_PUR_INPUT;
				else if (!strcmp(v->value.s.begin, "response"))
					purp = SMLP_PUR_RESPONSE;
				else if (!strcmp(v->value.s.begin, "categorical"))
					purp = SMLP_PUR_CONFIG;
				else
					FAIL(6, "error: 'type' value '%s' "
					        "of object #%" FMT_SZ "u in .spec array "
					        "is unknown\n",
					        v->value.s.begin, i);
			} else if (!strcmp(v->key.begin, "range")) {
				switch (v->value.type) {
				case KJSON_VALUE_STRING:
					if (!strcmp(v->value.s.begin, "int"))
						type = SMLP_DTY_INT;
					else if (!strcmp(v->value.s.begin, "float"))
						type = SMLP_DTY_DBL;
					else
						FAIL(7, "error: 'range' "
						        "value '%s' of "
						        "object #%" FMT_SZ "u in "
						        ".spec array is unknown\n",
						        v->value.s.begin, i);
					break;
				case KJSON_VALUE_ARRAY:
					type = SMLP_DTY_CAT;
					cats = &v->value.a;
					break;
				default:
					FAIL(8, "error: 'range' value of "
					        "object in element #%" FMT_SZ "u "
					        "of array is neither a "
					        "JSON string nor an .spec array\n",
					        i);
				}
			} else if (!strcmp(v->key.begin, "rad-rel")) {
				rad_rel = &v->value;
			} else if (!strcmp(v->key.begin, "rad-abs")) {
				rad_abs = &v->value;
			} else if (!strcmp(v->key.begin, "default")) {
				def_val = &v->value;
			} else if (!strcmp(v->key.begin, "safe")) {
				safe = &v->value;
			}
		}

		if ((int)type == -1 || (int)purp == -1 || !label)
			FAIL(9, "error: object #%" FMT_SZ "u in array does not "
			        "specify all of 'label', 'type' and "
			        "'range'\n", i);

		char *concat = NULL;
		const char **cats_data = NULL;
		if (cats) {
			size_t sz = 0, valid = 0;
			struct kjson_value *v = cats->data;
			size_t *cats_offsets = calloc(cats->n, sizeof(*cats_offsets));
			if (!cats_offsets)
				FAIL(10, "error allocating memory: %s\n",
				         strerror(errno));
			for (size_t j=0; j<cats->n; j++, v++) {
				int n;
				switch (v->type) {
				case KJSON_VALUE_BOOLEAN:
					n = v->b ? 4 : 5; /* "true" : "false" */
					break;
				case KJSON_VALUE_NUMBER_INTEGER:
					n = snprintf(NULL, 0, "%jd", v->i);
					break;
				case KJSON_VALUE_STRING:
					n = v->s.len;
					break;
				default:
					free(concat);
					free(cats_offsets);
					FAIL(11, "error: invalid entry type "
					         "for item #%" FMT_SZ "u in "
					         "'range' for object #%" FMT_SZ "u "
					         "in array\n", j, i);
				}
				int r = buf_ensure_sz((void *)&concat, &sz, &valid, n+1);
				if (r) {
					free(concat);
					free(cats_offsets);
					FAIL(12, "error allocating memory: %s\n",
					         strerror(-r));
				}
				cats_offsets[j] = valid;
				switch (v->type) {
				case KJSON_VALUE_BOOLEAN:
					valid += snprintf(concat + valid, sz - valid,
					                  "%s", v->b ? "true" : "false");
					break;
				case KJSON_VALUE_NUMBER_INTEGER:
					valid += snprintf(concat + valid, sz - valid,
					                  "%jd", v->i);
					break;
				case KJSON_VALUE_STRING:
					valid += snprintf(concat + valid, sz - valid,
					                  "%s", v->s.begin);
					break;
				default:
					break;
				}
				valid++; /* include '\0' */
			}

			cats_data = calloc(cats->n, sizeof(*cats_data));
			if (!cats_data)
				FAIL(13, "error allocating memory: %s\n",
				         strerror(errno));
			for (size_t j=0; j<cats->n; j++)
				cats_data[j] = concat + cats_offsets[j];
			free(cats_offsets);
		}

		c->dtype = type;
		c->purpose = purp;
		c->label = strdup(label->begin);
		c->cats.concat = concat;
		c->cats.data = cats_data;
		c->cats.n = cats ? cats->n : 0;
		table->n++;

		if (rad_rel && rad_abs) {
			FAIL(14, "error: 'rad-rel' and "
			         "'rad-abs' specifications in object "
			         "#%" FMT_SZ "u in .spec array\n", i);
		} else if (rad_abs && rad_abs->type == KJSON_VALUE_NUMBER_INTEGER && rad_abs->i == 0) {
			c->radius_type = SMLP_RAD_0;
		} else if (rad_abs) {
			if (c->dtype == SMLP_DTY_CAT)
				FAIL(15, "error: 'rad-abs' != 0 "
				         "in object #%" FMT_SZ "u in .spec "
				         "array of categorical type\n", i);
			if (!smlp_value_from_json(&c->rad.abs, c, rad_abs, false))
				FAIL(16, "error: 'rad-abs' "
				         "value type %d does not match 'range' "
				         "%d in object #%" FMT_SZ "u in .spec "
				         "array\n", rad_abs->type, c->dtype, i);
			c->radius_type = SMLP_RAD_ABS;
		} else if (rad_rel) {
			if (c->dtype == SMLP_DTY_CAT)
				FAIL(17, "error: 'rad-rel' "
				         "in object #%" FMT_SZ "u in .spec "
				         "array of categorical type\n", i);
			switch (rad_rel->type) {
			case KJSON_VALUE_NUMBER_INTEGER:
				c->rad.rel = rad_rel->i;
				break;
			case KJSON_VALUE_NUMBER_DOUBLE:
				c->rad.rel = rad_rel->d;
				break;
			default:
				FAIL(18, "error: 'rad-rel' "
				         "value of object #%" FMT_SZ "u in "
				         ".spec array is not supported\n", i);
			}
			c->radius_type = SMLP_RAD_REL;
		} else
			c->radius_type = SMLP_RAD_NONE;

		if (safe && safe->type != KJSON_VALUE_ARRAY)
			FAIL(19, "error: 'safe' value in "
			         "object #%" FMT_SZ "u in .spec array is "
			         "not an array itself\n", i);
		if (safe) {
			c->safe.data = calloc(safe->a.n, sizeof(*c->safe.data));
			c->safe.n = safe->a.n;
			for (size_t j=0; j<safe->a.n; j++) {
				struct kjson_value *v = &safe->a.data[j];
				if (!smlp_value_from_json(c->safe.data + j, c, v, true))
					FAIL(20, "error: "
					         "cannot interpret value #%"
					         FMT_SZ "u in 'safe' array "
					         "of object #%" FMT_SZ "u "
					         "in its type.\n", j, i);
			}
		}

		if (def_val && !smlp_value_from_json(&c->default_value, c, def_val, true))
			FAIL(21, "error: cannot interpret "
			         "'default' value of object #%" FMT_SZ "u "
			         "in .spec array in its type.\n", i);
	}

	return 0;
}

int smlp_spec_init_path(struct smlp_spec *spec, const char *path, char **error)
{
	struct kjson json;
	int r = kjson_init_path(&json, path, error);
	if (r)
		return r;
	r = smlp_spec_init(spec, &json.top, error);
	kjson_fini(&json);
	return r;
}

static const char *const purpose_strs[] = {
	[SMLP_PUR_CONFIG  ] = "knob",
	[SMLP_PUR_INPUT   ] = "input",
	[SMLP_PUR_RESPONSE] = "response",
};

static const char *smlp_dtype_strs[] = {
	[SMLP_DTY_INT] = "int",
	[SMLP_DTY_DBL] = "float",
	[SMLP_DTY_CAT] = "category",
};

const char * smlp_dtype_str(enum smlp_dtype dty)
{
	return 0 <= dty && dty < ARRAY_SIZE(smlp_dtype_strs)
	     ? smlp_dtype_strs[dty] : NULL;
}

static void entry_print_value_json(FILE *f, const struct smlp_spec_entry *e,
                                   union smlp_value v)
{
	switch (e->dtype) {
	case SMLP_DTY_CAT:
		fprintf(f, "\"%s\"", e->cats.data[v.c]);
		break;
	case SMLP_DTY_INT:
		fprintf(f, "%jd", v.i);
		break;
	case SMLP_DTY_DBL:
		fprintf(f, "%g", v.d);
		break;
	}
}

void smlp_spec_write(const struct smlp_spec *spec, FILE *f)
{
	fprintf(f, "[\n");
	for (size_t i=0; i<spec->n; i++) {
		const struct smlp_spec_entry *e = spec->cols + i;
		const char *type = purpose_strs[e->purpose];
		if (e->dtype == SMLP_DTY_CAT) {
			assert(e->purpose == SMLP_PUR_CONFIG);
			type = "categorical";
		}

		fprintf(f, "\t{");
		fprintf(f, " \"label\": \"%s\"", e->label);
		fprintf(f, ",\"type\": \"%s\"", type);
		fprintf(f, ",\"range\": ");
		if (e->dtype == SMLP_DTY_CAT) {
			fprintf(f, "[");
			for (size_t k=0; k<e->cats.n; k++)
				fprintf(f, "%s%c", e->cats.data[k],
				        k+1<e->cats.n ? ',' : ']');
		} else
			fprintf(f, "\"%s\"", smlp_dtype_strs[e->dtype]);
		if (e->has_default_value) {
			fprintf(f, ",\"default\": ");
			entry_print_value_json(f, e, e->default_value);
		}
		if (e->safe.n) {
			fprintf(f, ",\"safe\": [");
			for (size_t k=0; k<e->safe.n; k++) {
				entry_print_value_json(f, e, e->safe.data[k]);
				fprintf(f, "%c", k+1<e->safe.n ? ',' : ']');
			}
		}
		switch (e->radius_type) {
		case SMLP_RAD_NONE: break;
		case SMLP_RAD_0:
			fprintf(f, ",\"rad-abs\": 0");
			break;
		case SMLP_RAD_ABS:
			fprintf(f, ",\"rad-abs\": ");
			entry_print_value_json(f, e, e->rad.abs);
			break;
		case SMLP_RAD_REL:
			fprintf(f, ",\"rad-rel\": %g", e->rad.rel);
			break;
		}
		fprintf(f, " }%s\n", i+1<spec->n ? "," : "");
	}
	fprintf(f, "]\n");
}

void smlp_array_init(struct smlp_array *a, enum smlp_dtype dty)
{
	a->log_bytes = dty == SMLP_DTY_DBL ? 2 : 0;
	a->dty = dty;
	a->v = NULL;
}

static void smlp_array_set_internal(struct smlp_array *a, size_t i,
                                    union smlp_value v)
{
	switch (a->log_bytes << 2 | a->dty) {
	case 0 << 2 | SMLP_DTY_CAT: a->i8[i] = v.c; break;
	case 0 << 2 | SMLP_DTY_INT: a->i8[i] = v.i; break;

	case 1 << 2 | SMLP_DTY_CAT: a->i16[i] = v.c; break;
	case 1 << 2 | SMLP_DTY_INT: a->i16[i] = v.i; break;

	case 2 << 2 | SMLP_DTY_CAT: a->i32[i] = v.c; break;
	case 2 << 2 | SMLP_DTY_INT: a->i32[i] = v.i; break;
	case 2 << 2 | SMLP_DTY_DBL: a->f32[i] = v.d; break;

	case 3 << 2 | SMLP_DTY_CAT: a->i64[i] = v.c; break;
	case 3 << 2 | SMLP_DTY_INT: a->i64[i] = v.i; break;
	case 3 << 2 | SMLP_DTY_DBL: a->f64[i] = v.d; break;

	default: assert(0);
	}
}

static void smlp_array_convert(struct smlp_array *a, unsigned log_bytes,
                               size_t h, size_t sz)
{
	struct smlp_array b = {
		.log_bytes = log_bytes,
		.dty = a->dty,
		.v = malloc((1 << log_bytes) * sz),
	};
	assert(b.v);

	for (size_t i=0; i<h; i++)
		smlp_array_set_internal(&b, i, smlp_array_get(a, i));

	free(a->v);

	memcpy(a, &b, sizeof(b));
}

void smlp_speced_set(const struct smlp_speced_csv *sp, size_t i, size_t j,
                     union smlp_value v)
{
	struct smlp_array *a = &sp->cols[j];

	int req_logbytes = -1;
	switch (a->dty) {
	case SMLP_DTY_CAT:
		if (v.c == (int8_t)v.c)
			req_logbytes = 0;
		else if (v.c == (int16_t)v.c)
			req_logbytes = 1;
		else if (v.c == (int32_t)v.c)
			req_logbytes = 2;
		else
			req_logbytes = 3;
		break;
	case SMLP_DTY_INT:
		if (v.i == (int8_t)v.i)
			req_logbytes = 0;
		else if (v.i == (int16_t)v.i)
			req_logbytes = 1;
		else if (v.i == (int32_t)v.i)
			req_logbytes = 2;
		else
			req_logbytes = 3;
		break;
	case SMLP_DTY_DBL:
		if (v.d == (float)v.d)
			req_logbytes = 2;
		else
			req_logbytes = 3;
		break;
	}
	assert(req_logbytes != -1);

	if (req_logbytes > a->log_bytes)
		smlp_array_convert(a, req_logbytes, sp->h, sp->sz);

	smlp_array_set_internal(a, i, v);
}

/* speed-up our common case: integers treated as double */
static double my_strtof(const char *s, char **endptr)
{
	const char *s0 = s;
	int sgn = 1;
	if (*s == '-')
		sgn = 0, s++;
	else if (*s == '+')
		s++;
	uintmax_t r = 0;
	for (; *s; s++) {
		if (*s < '0' || *s > '9')
			return strtof(s0, endptr);
		uintmax_t t = r * 10 + (*s - '0');
		if (t < r)
			return strtof(s0, endptr);
		r = t;
	}
	if (endptr)
		*endptr = (char *)s;
	return sgn ? r : -(double)r;
}

void smlp_speced_ensure_size(struct smlp_speced_csv *sp, size_t w, size_t sz)
{
	if (sz > sp->sz) {
		sp->sz = MAX(sz, 2*sp->sz);
		for (size_t i=0; i<w; i++)
			smlp_array_resize(&sp->cols[i], sp->sz);
	}
}

int smlp_speced_init_csv(struct smlp_speced_csv *sp, FILE *csv,
                         const struct smlp_spec *spec)
{
	struct smlp_table c = SMLP_TABLE_INIT;
	smlp_table_read_header(&c, csv, ',');

	if (c.n_cols != spec->n) {
		log_error("error: CSV contains %zu columns != %zu in .spec\n",
		          c.n_cols, spec->n);
		smlp_table_fini(&c);
		return 1;
	}

	for (size_t i=0; i<c.n_cols; i++)
		if (strcmp(c.column_labels[i], spec->cols[i].label)) {
			log_error("error: column %zu in CSV has label '%s' != "
			          "expected '%s' from .spec\n",
			          i, c.column_labels[i], spec->cols[i].label);
			smlp_table_fini(&c);
			return 2;
		}

	smlp_table_fini(&c);

	sp->cols = calloc(spec->n, sizeof(*sp->cols));
	for (size_t i=0; i<spec->n; i++)
		smlp_array_init(sp->cols + i, spec->cols[i].dtype);

	char *line = NULL;
	size_t sz = 0;
	ssize_t rd;
	for (size_t no=2; (rd = getline(&line, &sz, csv)) > 0; no++, sp->h++) {
		smlp_speced_ensure_size(sp, spec->n, sp->h+1);

		if (line[rd-1] == '\n')
			line[--rd] = '\0';

		char *save = NULL;
		size_t i = 0;
		for (char *tok = strtok_r(line, ",", &save); tok;
		     tok = strtok_r(NULL, ",", &save), i++) {
			if (i >= spec->n)
				continue;
			bool r = false;
			char *endp;
			union smlp_value v;
			switch (spec->cols[i].dtype) {
			case SMLP_DTY_CAT:
				for (size_t j=0; j<spec->cols[i].cats.n; j++)
					if (!strcmp(tok, spec->cols[i].cats.data[j])) {
						v.c = j;
						r = true;
					}
				break;
			case SMLP_DTY_DBL:
				v.d = my_strtof(tok, &endp);
				// assert(v->d == strtof(tok, NULL));
				r = !*endp;
				break;
			case SMLP_DTY_INT:
				v.i = strtoimax(tok, &endp, 10);
				r = !*endp;
				break;
			}
			if (!r) {
				log_error("error interpreting '%s' as %s in "
				          "line %zu, column #%zu '%s' of CSV\n",
				          tok, smlp_dtype_str(spec->cols[i].dtype),
				          no, i, spec->cols[i].label);
				free(line);
				free(sp->cols);
				return 4;
			}
			smlp_speced_set(sp, sp->h, i, v);
		}
		if (i != spec->n) {
			log_error("error: line %zu of CSV %zu != %zu columns\n",
			          no, i, spec->n);
			free(line);
			free(sp->cols);
			return 3;
		}
	}
	free(line);

	return 0;
}

void smlp_speced_write_csv_header(const struct smlp_speced_csv *sp,
                                  const struct smlp_spec *spec,
                                  FILE *f)
{
	for (size_t j=0; j<spec->n; j++)
		fprintf(f, "%s%c", spec->cols[j].label, j+1<spec->n ? ',' : '\n');
}

void smlp_speced_write_csv_row(const struct smlp_speced_csv *sp,
                               const struct smlp_spec *spec,
                               size_t row, FILE *f)
{
	for (size_t j=0; j<spec->n; j++) {
		union smlp_value v = smlp_speced_get(sp, row, j);
		char sep = j+1<spec->n ? ',' : '\n';
		switch (spec->cols[j].dtype) {
		case SMLP_DTY_CAT:
			fprintf(f, "%s%c", spec->cols[j].cats.data[v.c], sep);
			break;
		case SMLP_DTY_INT:
			fprintf(f, "%jd%c", v.i, sep);
			break;
		case SMLP_DTY_DBL:
			fprintf(f, "%g%c", v.d, sep);
			break;
		}
	}
}
