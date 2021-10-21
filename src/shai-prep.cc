
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

SMLP_FN_ATTR_PRINTF(2,3)
static bool log(int lvl, const char *fmt, ...)
{
	if (lvl >= verbosity)
		return false;
	if (fmt) {
		va_list ap;
		va_start(ap, fmt);
		vfprintf(stderr, fmt, ap);
		va_end(ap);
	}
	return true;
}

#define TIMED(t, ...) do {                                                     \
		struct timeval tv, tw;                                         \
		gettimeofday(&tv, NULL);                                       \
		do { __VA_ARGS__ } while (0);                                  \
		gettimeofday(&tw, NULL);                                       \
		t = (tw.tv_sec - tv.tv_sec) + (tw.tv_usec - tv.tv_usec) * 1e-6;\
	} while (0)

struct shai : smlp::speced_csv {

	size_t                 timing_idx, delta_idx;
	std::vector<size_t>    nondup, by_cols;
	std::vector<size_t *>  groups;

	virtual ~shai() = default;

	shai(smlp::speced_csv &&speced,
	     const char *timing_lbl,
	     const char *delta_lbl)
	: speced_csv(move(speced))
	{
		if (ssize_t area_idx = column_idx("Area"); area_idx != -1) {
			log(1, "dropping 'Area' column\n");
			drop_column(area_idx);
		}

		timing_idx = column_idx(timing_lbl);
		if (timing_idx == (size_t)-1)
			throw std::runtime_error("Timing column '" +
			                         std::string(timing_lbl) +
			                         "' not in CSV");
		if (spec(timing_idx).dtype != SMLP_DTY_INT)
			throw std::runtime_error("Timing column '" +
			                         std::string(timing_lbl) +
			                         "' has type '" +
			                         smlp::to_str(spec(timing_idx).dtype) +
			                         "' != " + smlp::to_str(SMLP_DTY_INT));

		delta_idx = column_idx(delta_lbl);
		if (delta_idx == (size_t)-1)
			throw std::runtime_error("delta column '" +
			                         std::string(timing_lbl) +
			                         "' not in CSV");
		if (spec(delta_idx).dtype != SMLP_DTY_DBL)
			throw std::runtime_error("delta column '" +
			                         std::string(delta_lbl) +
			                         "' has type '" +
			                         smlp::to_str(spec(delta_idx).dtype) +
			                         "' != " + smlp::to_str(SMLP_DTY_DBL));

		double t;
		TIMED(t, unique_rows(std::back_inserter(nondup)); );
		log(1, "de-dup: %g sec -> %zux%zu\n", t, width(), nondup.size());

		log(1, "group by:");
		for (size_t i=0; i<width(); i++)
			if (i != timing_idx &&
			    spec(i).purpose == SMLP_PUR_CONFIG) {
				log(1, " %s", spec(i).label);
				by_cols.push_back(i);
			}
		log(1, "\n");

		TIMED(t,
			group_by(by_cols, [&](auto &rows, auto &rows_end)
			                  {
				std::sort(rows, rows_end,
				          [&](size_t a, size_t b)
				          { return timing(a) < timing(b); });

				if (log(2, "group %zu of size %5zu:",
				        groups.size(), rows_end - rows)) {
					for (size_t j : by_cols) {
						auto v = get(rows[0], j);
						if (spec(j).dtype == SMLP_DTY_DBL)
							log(2, " %g", v.d);
						else
							log(2, " %jd",
							    spec(j).dtype == SMLP_DTY_CAT
							    ? v.c : v.i);
					}
					log(2, "\n");
				}

				groups.push_back(&*rows);
			                  }, nondup);
		);
		groups.push_back(nondup.data() + nondup.size());
		log(1, "grouping into %zu groups: %g sec\n", groups.size()-1, t);
	}

	auto timing(size_t i) const { return get(i, timing_idx).i; }
	auto delta (size_t i) const { return get(i,  delta_idx).d; }

	void prepare(intmax_t rad)
	{
		double t;
		TIMED(t,
		for (size_t i=0; i+1<groups.size(); i++) {
			size_t *rows = groups[i];
			size_t rows_n = groups[i+1] - rows;
			for (size_t j=0; j<rows_n; j++)
				_prepare(rad, rows, j, rows_n);
		}
		);
		log(2, "rad %jd prepare took %g sec\n", rad, t);
	}

	virtual void objective(intmax_t trad, size_t row,
	                       intmax_t width, double min_delta) = 0;

private:
	void _prepare(intmax_t trad, const size_t *rows, size_t j, size_t n)
	{
		using std::min;

		intmax_t t0 = timing(rows[j]);
		double min_delta = delta(rows[j]);
		ssize_t beg, end;
		for (beg=j;; beg--)
			if (beg >= 0 && -trad <= timing(rows[beg]) - t0)
				min_delta = min(min_delta, delta(rows[beg]));
			else {
				beg++;
				break;
			}
		for (end=j;; end++)
			if ((size_t)end < n && timing(rows[end]) - t0 < trad)
				min_delta = min(min_delta, delta(rows[end]));
			else {
				end--;
				break;
			}
		objective(trad, rows[j],
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

	shai_v1(speced_csv &&speced,
	        const char *timing_lbl,
	        const char *delta_lbl,
	        bool is_rx)
	: shai(move(speced), timing_lbl, delta_lbl)
	, trad_idx(add_column(TRAD))
	, area_idx(add_column(AREA))
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

	void objective(intmax_t trad, const size_t row,
	               intmax_t width, double min_delta) override
	{
		auto [delta_a,time_a] = params();
		double fd = weigh(min_delta, 100, 0.02 / delta_a);
		double ft = weigh(width    , 100, 0.02 / time_a);
		set(row, trad_idx, { .i = trad });
		set(row, area_idx, { .d = fd * ft });
	}
};

struct shai_v2 : shai {

	const size_t trad_idx, eye_w_idx, eye_h_idx;

	shai_v2(speced_csv &&speced,
	        const char *timing_lbl,
	        const char *delta_lbl)
	: shai(move(speced), timing_lbl, delta_lbl)
	, trad_idx(add_column(TRAD))
	, eye_w_idx(add_column(EYE_W))
	, eye_h_idx(add_column(EYE_H))
	{}

	using shai::prepare;

	void objective(intmax_t trad, const size_t row,
	               intmax_t width, double min_delta
	              ) override
	{
		set(row, trad_idx , { .i = trad });
		set(row, eye_w_idx, { .i = width });
		set(row, eye_h_idx, { .d = min_delta });
	}
};

namespace {
template <typename... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;
}

extern "C" {

#ifdef SMLP_PY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL smlp_ARRAY_API
# include <numpy/arrayobject.h>
static struct init_numpy {
	init_numpy() {
		/* we don't want Python to steal our signal handlers... */
		Py_InitializeEx(0);

		/* ... and neither numpy; see, e.g.,
		 * <https://github.com/numpy/numpy/issues/7545> for
		 * what it uses signal handlers ... pointless */
		PyOS_sighandler_t sighandler = PyOS_getsig(SIGINT);
		/* wrap numpy's init call into a function, because for some
		 * reason it is a macro expanding to 'return' stuff,
		 * very useful for us. There is also _import_array(), but it's
		 * not documented and thus not stable. */
		[]() -> void * { import_array(); return NULL; }();
		PyOS_setsig(SIGINT,sighandler);
	}
} init_np;
#else
typedef struct PyObject PyObject;
#endif

struct smlp_mrc_shai {
	shai *shai;
	char *error;
	PyObject **np_cols;
	PyObject *np_idcs;
};

#define SMLP_MRC_SHAI_PREP_NO_PY	(1 << 0)

struct smlp_mrc_shai_params {
	const char *csv_in_path;
	const char *spec_in_path;
	const char *spec_out_path;
	const char *timing_lbl;
	const char *delta_lbl;
	const char *v1;
};

int smlp_mrc_shai_prep_init(struct smlp_mrc_shai *sh,
                            const struct smlp_mrc_shai_params *par,
                            unsigned flags)
{
	if (!par->spec_in_path)
		DIE(1,"error: IN.spec not set, use option '-s'\n");

	if (par->v1 && strcmp(par->v1, "rx") && strcmp(par->v1, "tx"))
		DIE(1,"error: parameter '%s' to '-1' is neither 'rx' nor 'tx'\n",par->v1);

	FILE *csv_in = par->csv_in_path ? fopen(par->csv_in_path, "r") : stdin;
	if (!csv_in)
		DIE(1,"%s: %s\n",par->csv_in_path,strerror(errno));

	try {
		smlp::speced_csv sp(csv_in, smlp::specification(par->spec_in_path));

		if (log(1,nullptr)) {
			for (size_t i=0; i<sp.width(); i++) {
				const auto &[a,e] = sp.col(i);
				log(1, "  %16s: %u byte(s) %s ~ %.1f MiB\n",
				    e.label, 1 << a.log_bytes, smlp::to_str(a.dty),
				    (sp.height() << a.log_bytes) / 1024.0 / 1024.0);
			}
		}

		if (par->v1)
			sh->shai = new shai_v1(move(sp), par->timing_lbl,
			                       par->delta_lbl,
			                       !strcmp(par->v1, "rx"));
		else
			sh->shai = new shai_v2(move(sp), par->timing_lbl,
			                       par->delta_lbl);

		if (par->spec_out_path) {
			FILE *f = fopen(par->spec_out_path, "w");
			if (!f)
				DIE(1,"%s: %s\n",par->spec_out_path,strerror(errno));
			smlp_spec_write(&sh->shai->spec(), f);
			fclose(f);
		}

	} catch (const std::runtime_error &ex) {
		DIE(1,"error: %s\n", ex.what());
	}

	sh->np_cols = NULL;
	sh->np_idcs = NULL;

	if (~flags & SMLP_MRC_SHAI_PREP_NO_PY) {
#ifdef SMLP_PY
		sh->np_cols = (PyObject **)malloc(sizeof(*sh->np_cols) * sh->shai->width());
		npy_intp dim = sh->shai->height();
		for (size_t j=0; j<sh->shai->width(); j++)
			sh->np_cols[j] = PyArray_SimpleNewFromData(1, &dim,
				smlp::with(sh->shai->column(j), overloaded {
					[](int8_t  *){ return NPY_INT8; },
					[](int16_t *){ return NPY_INT16; },
					[](int32_t *){ return NPY_INT32; },
					[](int64_t *){ return NPY_INT64; },
					[](float   *){ return NPY_FLOAT32; },
					[](double  *){ return NPY_FLOAT64; },
				}), sh->shai->column(j).v);
		dim = sh->shai->nondup.size();
		static_assert(std::is_same_v<size_t,uint64_t>);
		sh->np_idcs = PyArray_SimpleNewFromData(1, &dim, NPY_UINT64,
		                                        sh->shai->nondup.data());
#endif
	}

	return 0;
}

void smlp_mrc_shai_prep_fini(struct smlp_mrc_shai *sh)
{
	delete sh->shai;

	free(sh->np_cols);
}

void smlp_mrc_shai_prep_rad(struct smlp_mrc_shai *sh, int trad)
{
	sh->shai->prepare(trad);
}

}

int main(int argc, char **argv)
{
	smlp_mrc_shai_params par = {
		.csv_in_path = NULL,
		.spec_in_path = NULL,
		.spec_out_path = NULL,
		.timing_lbl = DEFAULT_TIMING,
		.delta_lbl = DEFAULT_DELTA,
		.v1 = NULL,
	};
	const char *csv_out_path = NULL;
	bool use_jobserver = false;
	bool quiet = false;
	int jobs = 0;

	progname = basename(argv[0]);

	for (int opt; (opt = getopt(argc, argv, ":1:D:hi:j:mo:qs:t:T:v")) != -1;)
		switch (opt) {
		case '1': par.v1 = optarg; break;
		case 'D': par.delta_lbl = optarg; break;
		case 'h': usage();
		case 'i': par.csv_in_path = optarg; break;
		case 'j': jobs = atoi(optarg); break;
		case 'm': use_jobserver = true; break;
		case 'o': csv_out_path = optarg; break;
		case 'q': quiet = true; break;
		case 's': par.spec_in_path = optarg; break;
		case 't': par.spec_out_path = optarg; break;
		case 'T': par.timing_lbl = optarg; break;
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

	smlp_mrc_shai sh;
	int r = smlp_mrc_shai_prep_init(&sh, &par, 0*SMLP_MRC_SHAI_PREP_NO_PY);
	if (r < 0)
		DIE(-r,"error: %s\n", strerror(-r));
	if (r > 0)
		DIE(r,"error: %s\n", sh.error);

	FILE *out = csv_out_path ? fopen(csv_out_path, "w") : stdout;
	if (!out)
		DIE(1,"%s: %s\n",csv_out_path,strerror(errno));

	sh.shai->write_csv_header(out);
	for (auto diam : { 70, 80, 90, 100, 110, 120, 130 }) {
		smlp_mrc_shai_prep_rad(&sh, diam / 2);
		for (size_t i : sh.shai->nondup)
			sh.shai->write_csv_row(out, i);
	}
	fclose(out);

	smlp_mrc_shai_prep_fini(&sh);

	return 0;
}
