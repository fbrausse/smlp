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
#include <stdio.h>
#include <errno.h>
#include <assert.h>
#include <unistd.h>	/* getopt(3p) */
#include <math.h>	/* fabs(3) */
#include <stdarg.h>	/* va_* */
#include <limits.h>	/* INT_MAX */

#include <sys/time.h>

#define STR(x)			#x
#define XSTR(x)			STR(x)
#define ARRAY_SIZE(...)		(sizeof(__VA_ARGS__)/sizeof(*(__VA_ARGS__)))
#define MAX(a,b)		((a) > (b) ? (a) : (b))

#define DIE(code,...) do { fprintf(stderr, __VA_ARGS__); exit(code); } while (0)

struct str {
	char *data;
	size_t sz, len;
};

static void str_vappendf(struct str *s, const char *fmt, va_list ap)
{
	va_list aq;
	va_copy(aq, ap);
	int n = vsnprintf(s->data + s->len, s->sz - s->len, fmt, ap);
	if (n >= s->sz - s->len) {
		s->data = realloc(s->data, s->sz = MAX(2*s->sz, s->len + n+1));
		n = vsnprintf(s->data + s->len, s->sz - s->len, fmt, aq);
		assert(n < s->sz - s->len);
	}
	s->len += n;
	va_end(aq);
}

static void str_appendf(struct str *s, const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	str_vappendf(s, fmt, ap);
	va_end(ap);
}

static thread_local struct str buf = { NULL, 0, 0 };

static void print_csv_header(int verbosity)
{
	str_appendf(&buf, "fail,ok");
	if (verbosity > 0)
		str_appendf(&buf, ",fail-idcs,ok-idcs");
	printf("%s\n", buf.data);
	buf.len = 0;
}

static void str_append_idxbuf(struct str *s, const struct check_data_idxbuf *b,
                              const char *internal_delim, const char *pre_delim)
{
	if (b->n)
		for (size_t i=0; i<b->n; i++)
			str_appendf(s, "%s%zu", i ? internal_delim : pre_delim,
			            b->data[i]);
	else
		str_appendf(s, "%s", pre_delim);
}

static void print_csv_line(size_t fail, size_t ok, struct check_data_idxbuf *fi,
                           struct check_data_idxbuf *fo, int verbosity)
{
	str_appendf(&buf, "%zu,%zu", fail, ok);
	if (verbosity > 0) {
		str_append_idxbuf(&buf, fi, ":", ",");
		str_append_idxbuf(&buf, fo, ":", ",");
	}
	printf("%s\n", buf.data);
	buf.len = 0;
}

static void print_csv_fini(void)
{
	free(buf.data);
	buf.sz = buf.len = 0;
}

static void extract_bounds_json(FILE *f, float minmax[static 2], const char *resp)
{
	struct smlp_cbuf buf;
	int r;
	if ((r = smlp_cbuf_init_file(&buf, f)))
		DIE(1,"error reading JSON BNDS: %s\n",strerror(-r));
	struct kjson_value o;
	if (!kjson_parse(&(struct kjson_parser){ buf.cstr.buf }, &o) ||
	    o.type != KJSON_VALUE_OBJECT)
		DIE(1,"error parsing JSON BNDS\n");

	const struct kjson_value *b = NULL;
	for (size_t i=0; i<o.o.n && !b; i++)
		if (!strcmp(o.o.data[i].key.begin, resp))
			b = &o.o.data[i].value;
	if (!b)
		DIE(1,"error: key '%s' not in BNDS JSON object\n", resp);
	if (b->type != KJSON_VALUE_OBJECT)
		DIE(1,"error: '%s' does not map to a JSON object in BNDS\n",
		    resp);

	const struct kjson_value *min = NULL, *max = NULL;
	for (size_t i=0; i<b->o.n && (!min || !max); i++)
		if (!strcmp(b->o.data[i].key.begin, "min"))
			min = &b->o.data[i].value;
		else if (!strcmp(b->o.data[i].key.begin, "max"))
			max = &b->o.data[i].value;
	if (!min)
		DIE(1,"error: key 'min' not in '%s' JSON object in BNDS\n",
		    resp);
	if (min->type != KJSON_VALUE_NUMBER_DOUBLE)
		DIE(1,"error: 'min' in '%s' does not map to a double value in BNDS\n",
		    resp);
	if (!max)
		DIE(1,"error: key 'max' not in '%s' JSON object in BNDS\n",
		    resp);
	if (min->type != KJSON_VALUE_NUMBER_DOUBLE)
		DIE(1,"error: 'max' in '%s' does not map to a double value in BNDS\n",
		    resp);

	minmax[0] = min->d;
	minmax[1] = max->d;

	kjson_value_fini(&o);
	smlp_cbuf_fini(&buf);
}

static void extract_bounds_csv(FILE *f, float minmax[static 2], const char *resp)
{
	struct smlp_table b = SMLP_TABLE_INIT;
	int r;
	if ((r = smlp_table_read(&b, f, 1)) < 0)
		DIE(1,"error reading BNDS csv: %s\n", strerror(-r));
	ssize_t idx = smlp_table_col_idx(&b, resp);
	if (idx < 0)
		DIE(1,"error: BNDS csv does not contain column '%s'\n", resp);
	if (b.n_rows != 2)
		DIE(1,"error: BNDS csv contains %zu != 2 rows\n", b.n_rows);
	minmax[0] = smlp_table_data_row(&b, 0)[idx];
	minmax[1] = smlp_table_data_row(&b, 1)[idx];
}

static FILE * open_rd(const char *path)
{
	FILE *f = fopen(path, "r");
	if (!f)
		DIE(1,"%s: %s\n",path,strerror(errno));
	return f;
}

static void usage(const char *progname)
{
	printf("\
usage: %s [-v] [ -b BNDS | -B BNDS ] [-f FMT] -r RESPONSE -s SPEC -S SAFE -t THRESHOLD DATA\n",
	       progname);
	printf("\n\
  -b BNDS.json  JSON file describing the data bounds\n\
  -B BNDS.csv   CSV file describing the data bounds\n\
  -f FMT        output format: none, human, [csv]\n\
                'csv' is suitable as input to 'paste -d, SAFE -'\n\
  -h            print this help message\n\
  -r RESPONSE   name of the response variable\n\
  -s SPEC       path to the JSON spec of the input features\n\
  -S SAFE       path to the CSV file containing the safe configurations\n\
  -t THRESHOLD  threshold to restrict output feature to be larger-equal\n\
                [default: " XSTR(DEFAULT_THRESHOLD) "]\n\
  -v            increase verbosity\n\
  DATA          path to the original input data in CSV format\n\
\n\
Exit codes:\n\
  0  all points in SAFE are safe according to DATA\n\
  1  error reading files\n\
  2  error interpreting parameters\n\
  8  a region in SAFE is not safe according to DATA\n\
");
	exit(0);
}

int main(int argc, char **argv)
{
	double threshold = NAN;
	void (*f_bnds_extract)(FILE *, float [static 2], const char *) = NULL;
	FILE *f_spec = NULL, *f_safe = NULL, *f_data = NULL, *f_bnds = NULL;
	const char *resp = NULL;
	int verbosity = 0;
	enum { NONE, HUMAN, CSV } fmt = CSV;
	static const char *const fmt_strs[] = {
		[NONE ] = "none",
		[HUMAN] = "human",
		[CSV  ] = "csv",
	};
next_opt:
	for (int opt; (opt = getopt(argc, argv, ":b:B:f:hr:s:S:t:v")) != -1;)
		switch (opt) {
		case 'b':
		case 'B':
			if (f_bnds)
				fclose(f_bnds);
			f_bnds = open_rd(optarg);
			f_bnds_extract = opt == 'b' ? extract_bounds_json
			                            : extract_bounds_csv;
			break;
		case 'f':
			for (size_t i=0; i<ARRAY_SIZE(fmt_strs); i++)
				if (!strcmp(optarg, fmt_strs[i])) {
					fmt = i;
					goto next_opt;
				}
			DIE(2,"error: unknown FMT '%s' given for option '-f'\n",
			      optarg);
		case 'h': usage(argv[0]);
		case 'r': resp = optarg; break;
		case 's': if (f_spec) fclose(f_spec); f_spec = open_rd(optarg); break;
		case 'S': if (f_safe) fclose(f_safe); f_safe = open_rd(optarg); break;
		case 't': threshold = atof(optarg); break;
		case 'v': verbosity++; break;
		case '?': DIE(2,"error: unknown option '-%c'\n",optopt);
		case ':': DIE(2,"error: option '-%c' requires a parameter\n",
		              optopt);
		}
	if (!resp)
		DIE(2,"error: RESPONSE not specified\n");
	if (!f_spec)
		DIE(2,"error: SPEC not specified\n");
	if (!f_safe)
		DIE(2,"error: SAFE not specified\n");
	if (!isfinite(threshold))
		DIE(2,"error: THRESHOLD not specified\n");

	if (argc - optind < 1)
		DIE(2,"error: DATA not specified\n");
	f_data = open_rd(argv[optind++]);
	if (argc - optind != 0)
		DIE(2,"error: unrecognized trailing parameters\n");

	int r;
	struct smlp_table safe = SMLP_TABLE_INIT;
	if ((r = smlp_table_read(&safe, f_safe, 1)))
		DIE(1,"error reading SAFE: %s\n",
		    r < 0 ? strerror(-r) : strerror(r));
	fclose(f_safe);
	f_safe = NULL;

	float *resp_norm_minmax = NULL;
	float resp_bounds[2];
	if (f_bnds) {
		f_bnds_extract(f_bnds, resp_bounds, resp);
		resp_norm_minmax = resp_bounds;
	}

	struct check_data cd = CHECK_DATA_INIT;
	struct timeval tv, tw;
	gettimeofday(&tv, NULL);
	static char BUF[1 << 18];
	setvbuf(f_data, BUF, _IOFBF, sizeof(BUF));
	r = check_data_init(&cd, resp, resp_norm_minmax, f_spec, f_data,
	                    safe.n_cols, safe.column_labels);
	fclose(f_spec), fclose(f_data);
	f_spec = f_data = NULL;
	if (r < 0)
		DIE(1,"error init'ing: %s\n", strerror(-r));
	switch (r) {
	case CHECK_DATA_ERR_SUCCESS: break;
	case CHECK_DATA_ERR_JSON:
		DIE(1,"error reading SPEC\n");
	case CHECK_DATA_ERR_SPEC_FORMAT:
		DIE(1,"spec JSON is not an array");
	case CHECK_DATA_ERR_RESPONSE_DATA_MISMATCH:
		DIE(1,"error: response '%s' is not part of DATA's labels\n",resp);
	case CHECK_DATA_ERR_SAFE2DATA:
		DIE(1,"error: matching SAFE and DATA labels\n");
	case CHECK_DATA_ERR_SAFE2SPEC:
		DIE(1,"error: matching SAFE and SPEC labels\n");
	}
	gettimeofday(&tw, NULL);
	fprintf(stderr, "reading data took %g sec\n",
	        (tw.tv_sec - tv.tv_sec) + 1e-6 * (tw.tv_usec - tv.tv_usec));

	struct check_data_idxbuf fi = CHECK_DATA_IDXBUF_INIT;
	struct check_data_idxbuf fo = CHECK_DATA_IDXBUF_INIT;

	int any_fail = 0;

	if (fmt == CSV)
		print_csv_header(verbosity);

	gettimeofday(&tv, NULL);
	for (size_t i=0; i<safe.n_rows; i++) {
		size_t ok = 0, fail;
		const float *c = smlp_table_data_row(&safe, i);
		struct check_data_idxbuf *fi_p = verbosity > 0 ? fi.n = 0, &fi : NULL;
		struct check_data_idxbuf *fo_p = verbosity > 0 ? fo.n = 0, &fo : NULL;
		fail = check_data_check(&cd, &ok, c, threshold, fi_p, fo_p);
		any_fail |= fail > 0;
		switch (fmt) {
		case NONE: continue;
		case HUMAN: break;
		case CSV:
			print_csv_line(fail, ok, fi_p, fo_p, verbosity);
			continue;
		}
		if (verbosity == 0) {
			printf("%zu: %zu fail, %zu ok, %zu outside\n",
			       i, fail, ok, cd.data.n_rows - fail - ok);
			continue;
		}
		for (size_t jf=0, jo=0; jf < fi.n || jo < fo.n;) {
			int f = jf < fi.n &&
			        (jo >= fo.n || fi.data[jf].idx < fo.data[jo].idx);
			// size_t j = f ? fi.data[jf++].idx : fo.data[jo++].idx;
			struct check_data_idxbuf_entry *e;
			e = f ? &fi.data[jf++] : &fo.data[jo++];
			// const float *row = smlp_table_data_row(&cd.data, j);
			printf(f ? "data row %zu in ball but threshold "
			           "condition fails: %g < %g"
			         : "data row %zu in ball, OK: %g >= %g",
			       e->idx, e->obj_val,
			       /*check_data_compute_objective(&cd, row),*/
			       threshold);
			printf(", distance: %g\n", e->obj_val - threshold);
		}
	}
	gettimeofday(&tw, NULL);
	fprintf(stderr, "checking data took %g sec\n",
	        (tw.tv_sec - tv.tv_sec) + 1e-6 * (tw.tv_usec - tv.tv_usec));

	print_csv_fini();
	check_data_idxbuf_fini(&fi);
	check_data_idxbuf_fini(&fo);
	check_data_fini(&cd);
	smlp_table_fini(&safe);
	return any_fail ? 8 : 0;
}
