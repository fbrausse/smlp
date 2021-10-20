
#define _POSIX_C_SOURCE 200809L

#include <cstring>
#include <cstdlib>
#include <cinttypes>	/* strtoimax */
#include <cstdarg>
#include <unistd.h>	/* getopt */
#include <libgen.h>	/* basename */
#include <cmath>	/* exp */

#include <sys/time.h>

#include <smlp/spec.hh>

using std::vector, std::move;

#define DIE(code, ...) do { fprintf(stderr,__VA_ARGS__); exit(code); } while (0)
#define DEFAULT_TIMING	"Timing"
#define DEFAULT_DELTA	"delta"

static int verbosity = 1;
static const char *progname;

static void usage(void)
{
	printf("\
usage: %s [-OPTS] -s IN.spec\n\
\n\
Options [defaults]:\n\
  -1 {rx|tx}   prepare single-objective version for RX or TX [multi-objective]\n\
  -D DELTA     name of the 'delta' feature in IN.csv [" DEFAULT_DELTA "]\n\
  -h           display this help message\n\
  -i IN.csv    read dataset from IN.csv [stdin]\n\
  -j N         use N parallel threads to process the dataset [none]\n\
  -m           enable compatibility to gmake's jobserver for parallelism [false]\n\
  -o OUT.csv   write processed dataset to OUT.csv [stdout]\n\
  -q           quiet mode, suppress all non-error outputs [false]\n\
  -s IN.spec   read .spec from IN.spec\n\
  -t OUT.spec  write processed .spec to OUT.spec\n\
  -T TIMING    name of the 'time' feature in IN.csv [" DEFAULT_TIMING "]\n\
  -v           increase verbosity [1]\n\
", progname);
	exit(0);
}

static bool log(int lvl, const char *fmt, ...)
{
	if (lvl >= verbosity)
		return false;
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	return true;
}

#define TIMED(t, ...) do {                                                     \
		struct timeval tv, tw;                                         \
		gettimeofday(&tv, NULL);                                       \
		do { __VA_ARGS__ } while (0);                                  \
		gettimeofday(&tw, NULL);                                       \
		t = (tw.tv_sec - tv.tv_sec) + (tw.tv_usec - tv.tv_usec) * 1e-6;\
	} while (0)

struct shai {

	smlp::speced_csv       sp;
	const size_t           timing_idx, delta_idx;
	std::vector<size_t>    nondup, by_cols;
	std::vector<size_t *>  groups;

	virtual ~shai() = default;

	shai(FILE *csv_in, const char *spec_path,
	     const char *timing_lbl,
	     const char *delta_lbl)
	: sp(csv_in, smlp::spec(spec_path))
	, timing_idx(sp.column_idx(timing_lbl))
	, delta_idx(sp.column_idx(delta_lbl))
	{
		double t;

		TIMED(t, sp.unique_rows(std::back_inserter(nondup)); );
		log(1, "de-dup: %g sec -> %zux%zu\n", t, sp.width(), size(nondup));

		if (ssize_t area_idx = sp.column_idx("Area"); area_idx != -1)
			sp.drop_column(area_idx);

		if (timing_idx == (size_t)-1)
			throw std::runtime_error("Timing column '" +
			                         std::string(timing_lbl) +
			                         "' not in CSV");
		if (sp.get_spec(timing_idx).dtype != SMLP_DTY_INT)
			throw std::runtime_error("Timing column '" +
			                         std::string(timing_lbl) +
			                         "' has type '" +
			                         smlp::to_str(sp.get_spec(timing_idx).dtype) +
			                         "' != " + smlp::to_str(SMLP_DTY_INT));

		if (delta_idx == (size_t)-1)
			throw std::runtime_error("delta column '" +
			                         std::string(timing_lbl) +
			                         "' not in CSV");
		if (sp.get_spec(delta_idx).dtype != SMLP_DTY_DBL)
			throw std::runtime_error("delta column '" +
			                         std::string(delta_lbl) +
			                         "' has type '" +
			                         smlp::to_str(sp.get_spec(delta_idx).dtype) +
			                         "' != " + smlp::to_str(SMLP_DTY_DBL));

		log(1, "group by:");
		for (size_t i=0; i<sp.width(); i++)
			if (i != timing_idx &&
			    sp.get_spec(i).purpose == SMLP_PUR_CONFIG) {
				log(1, " %s", sp.get_spec(i).label);
				by_cols.push_back(i);
			}
		log(1, "\n");

		// size_t eye_n = size(time_window_radii);

		TIMED(t,
			sp.group_by(by_cols, [&](auto &rows, auto &rows_end)
			                     {
				std::sort(rows, rows_end,
				          [&](size_t a, size_t b)
				          { return timing(a) < timing(b); });

				if (log(2, "group %zu of size %5zu:",
				        groups.size(), rows_end - rows)) {
					for (size_t j : by_cols) {
						auto v = sp.get(rows[0], j);
						if (sp.get_spec(j).dtype == SMLP_DTY_DBL)
							log(2, " %g", v.d);
						else
							log(2, " %jd",
							    sp.get_spec(j).dtype == SMLP_DTY_CAT
							    ? v.c : v.i);
					}
					log(2, "\n");
				}

				groups.push_back(&*rows);
			                     }, nondup);
		);
		groups.push_back(nondup.data() + nondup.size());
		log(1, "grouping into %zu groups: %g sec\n", size(groups)-1, t);
	}

	auto timing(size_t i) const { return sp.get(i, timing_idx).i; }
	auto delta (size_t i) const { return sp.get(i,  delta_idx).d; }

	void prepare(intmax_t rad)
	{
		for (size_t i=0; i+1<size(groups); i++) {
			size_t *rows = groups[i];
			size_t rows_n = groups[i+1] - rows;
			for (size_t j=0; j<rows_n; j++)
				_prepare(rad, rows, j, rows_n);
		}
	}

	virtual void objective(intmax_t trad, const size_t *rows, size_t j,
	                       intmax_t width, double min_delta) = 0;

private:
	void _prepare(intmax_t rad, const size_t *rows, size_t j, size_t n)
	{
		using std::min;

		intmax_t t0 = timing(rows[j]);
		double min_delta = delta(rows[j]);
		ssize_t beg, end;
		for (beg=j;; beg--)
			if (beg >= 0 && -rad <= timing(rows[beg]) - t0)
				min_delta = min(min_delta, delta(rows[beg]));
			else {
				beg++;
				break;
			}
		for (end=j;; end++)
			if ((size_t)end < n && timing(rows[end]) - t0 < rad)
				min_delta = min(min_delta, delta(rows[end]));
			else {
				end--;
				break;
			}
		objective(rad, rows, j,
		          timing(rows[end]) - timing(rows[beg]) + 1, min_delta);
	}
};

static const ::smlp_spec_entry
	TRAD = {
		.dtype   = SMLP_DTY_INT,
		.purpose = SMLP_PUR_CONFIG,
		.radius_type = SMLP_RAD_0,
		.label   = strdup("trad"),
	//	.safe    = { time_window_radii.data(), eye_n },
	},
	AREA = {
		.dtype   = SMLP_DTY_DBL,
		.purpose = SMLP_PUR_RESPONSE,
		.label   = strdup("area"),
	},
	EYE_W = {
		.dtype   = SMLP_DTY_INT,
		.purpose = SMLP_PUR_RESPONSE,
		.label   = strdup("eye_w"),
	},
	EYE_H = {
		.dtype   = SMLP_DTY_DBL,
		.purpose = SMLP_PUR_RESPONSE,
		.label   = strdup("eye_h"),
	};

struct shai_v1 : shai {

	const size_t trad_idx, area_idx;
	const bool is_rx;

	shai_v1(FILE *csv_in, const char *spec_path,
	        const char *timing_lbl,
	        const char *delta_lbl,
	        bool is_rx)
	: shai(csv_in, spec_path, timing_lbl, delta_lbl)
	, trad_idx(sp.add_column(TRAD))
	, area_idx(sp.add_column(AREA))
	, is_rx(is_rx)
	{}

	using shai::prepare;

	std::pair<double,double> params() const
	{
		if (is_rx)
			return { 2.346, 4.882 };
		else
			return { 15.6 , 4.882 };
	}

	static double weigh(double x, double x0, double sigma)
	{
		return 1.0 / (1 + exp(-2 * sigma * (x - x0)));
	}

	void objective(intmax_t trad, const size_t *rows, size_t j,
	               intmax_t width, double min_delta) override
	{
		auto [delta_a,time_a] = params();
		double fd = weigh(min_delta, 100, 0.02 / delta_a);
		double ft = weigh(width    , 100, 0.02 / time_a);
		sp.set(rows[j], trad_idx, { .i = trad });
		sp.set(rows[j], area_idx, { .d = fd * ft });
	}
};

struct shai_v2 : shai {

	const size_t trad_idx, eye_w_idx, eye_h_idx;

	shai_v2(FILE *csv_in, const char *spec_path,
	        const char *timing_lbl,
	        const char *delta_lbl)
	: shai(csv_in, spec_path, timing_lbl, delta_lbl)
	, trad_idx(sp.add_column(TRAD))
	, eye_w_idx(sp.add_column(EYE_W))
	, eye_h_idx(sp.add_column(EYE_H))
	{}

	using shai::prepare;

	void objective(intmax_t trad, const size_t *rows, size_t j,
	               intmax_t width, double min_delta
	              ) override
	{
		sp.set(rows[j], trad_idx , { .i = trad });
		sp.set(rows[j], eye_w_idx, { .i = width });
		sp.set(rows[j], eye_h_idx, { .d = min_delta });
	}
};

static void run(shai &&s, const char *spec_out_path, const char *csv_out_path)
{
	if (spec_out_path) {
		FILE *f = fopen(spec_out_path, "w");
		if (!f)
			DIE(1,"%s: %s\n",spec_out_path,strerror(errno));
		smlp_spec_write(&s.sp.get_spec(), f);
		fclose(f);
	}

	FILE *out = csv_out_path ? fopen(csv_out_path, "w") : stdout;
	if (!out)
		DIE(1,"%s: %s\n",csv_out_path,strerror(errno));

	s.sp.write_csv_header(out);
	for (auto diam : { 70, 80, 90, 100, 110, 120, 130 }) {
		s.prepare(diam / 2);
		for (size_t i : s.nondup)
			s.sp.write_csv_row(out, i);
	}
	fclose(out);
}

int main(int argc, char **argv)
{
	const char *csv_in_path = NULL, *csv_out_path = NULL;
	const char *spec_in_path = NULL, *spec_out_path = NULL;
	const char *timing_lbl = DEFAULT_TIMING;
	const char *delta_lbl  = DEFAULT_DELTA;
	bool use_jobserver = false;
	bool quiet = false;
	int jobs = 0;
	const char *v1 = nullptr;

	progname = basename(argv[0]);

	for (int opt; (opt = getopt(argc, argv, ":1:D:hi:j:mo:qs:t:T:v")) != -1;)
		switch (opt) {
		case '1': v1 = optarg; break;
		case 'D': delta_lbl = optarg; break;
		case 'h': usage();
		case 'i': csv_in_path = optarg; break;
		case 'j': jobs = atoi(optarg); break;
		case 'm': use_jobserver = true; break;
		case 'o': csv_out_path = optarg; break;
		case 'q': quiet = true; break;
		case 's': spec_in_path = optarg; break;
		case 't': spec_out_path = optarg; break;
		case 'T': timing_lbl = optarg; break;
		case 'v': verbosity++; break;
		case ':': DIE(1,"error: option '-%c' requires a parameter",optopt);
		case '?': DIE(1,"error: unknown option '-%c'\n",optopt);
		}

	if (quiet)
		verbosity = 0;

	if (optind < argc) {
		fprintf(stderr, "error: unknown trailing options:");
		for (; optind < argc; optind++)
			fprintf(stderr, " %s", argv[optind]);
		DIE(1,"\n");
	}

	if (!spec_in_path)
		DIE(1,"error: SPEC.in not set, use option '-s'\n");

	if (v1 && strcmp(v1, "rx") && strcmp(v1, "tx"))
		DIE(1,"error: parameter '%s' to '-1' is neither 'rx' nor 'tx'\n",v1);

	FILE *csv_in = csv_in_path ? fopen(csv_in_path, "r") : stdin;
	if (!csv_in)
		DIE(1,"%s: %s\n",csv_in_path,strerror(errno));

	try {
		run(v1 ? static_cast<shai &&>(shai_v1(csv_in, spec_in_path,
		                                      timing_lbl, delta_lbl,
		                                      !strcmp(v1, "rx")))
		       : static_cast<shai &&>(shai_v2(csv_in, spec_in_path,
		                                      timing_lbl, delta_lbl)),
		    spec_out_path, csv_out_path);
	} catch (const std::runtime_error &ex) {
		DIE(1,"error: %s\n", ex.what());
	}

	// TODO? use numpy's PyArray_SimpleNewFromData() to wrap each sp.cols[*]

	return 0;
}
