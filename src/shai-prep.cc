
#include <cstring>
#include <cstdlib>
#include <cinttypes>	/* strtoimax */
#include <cstdarg>
#include <cmath>	/* exp */

#include <sys/time.h>

#include <smlp/spec.hh>

extern "C" int verbosity = 1;

using std::vector, std::move;

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

	size_t            timing_idx, delta_idx;
	vector<size_t>    index, by_cols;
	vector<size_t *>  groups;

	virtual ~shai() = default;

	auto timing(size_t i) const { return get(i, timing_idx).i; }
	auto delta (size_t i) const { return get(i,  delta_idx).d; }

	shai(smlp::speced_csv &&speced,
	     const char *timing_lbl,
	     const char *delta_lbl,
	     bool dedup = true)
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
		if (dedup) {
			TIMED(t, unique_rows(std::back_inserter(index)); );
			log(1, "de-dup: %g sec -> %zux%zu\n", t, width(), index.size());
		} else
			index = smlp::indices(height());

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
				          { return this->timing(a) < this->timing(b); });

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
			                  }, index);
		);
		groups.push_back(index.data() + index.size());
		log(1, "grouping into %zu groups: %g sec\n", groups.size()-1, t);
	}

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

static ::smlp_spec_entry trad_entry()
{
	return {
		.dtype   = SMLP_DTY_INT,
		.purpose = SMLP_PUR_CONFIG,
		.radius_type = SMLP_RAD_0,
		.label   = strdup("trad"),
	//	.safe    = { time_window_radii.data(), eye_n },
	};
}

static ::smlp_spec_entry area_entry()
{
	return {
		.dtype   = SMLP_DTY_DBL,
		.purpose = SMLP_PUR_RESPONSE,
		.label   = strdup("area"),
	};
}

static ::smlp_spec_entry eye_w_entry()
{
	return {
		.dtype   = SMLP_DTY_INT,
		.purpose = SMLP_PUR_RESPONSE,
		.label   = strdup("eye_w"),
	};
}

static ::smlp_spec_entry eye_h_entry()
{
	return {
		.dtype   = SMLP_DTY_DBL,
		.purpose = SMLP_PUR_RESPONSE,
		.label   = strdup("eye_h"),
	};
}

struct shai_v1 : shai {

	static constexpr const size_t ADDED_COLS = 2;

	const size_t trad_idx, area_idx;
	const bool is_rx;

	shai_v1(speced_csv &&speced,
	        const char *timing_lbl,
	        const char *delta_lbl,
	        bool is_rx)
	: shai(move(speced), timing_lbl, delta_lbl)
	, trad_idx(add_column(trad_entry()))
	, area_idx(add_column(area_entry()))
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

	static constexpr const size_t ADDED_COLS = 3;

	const size_t trad_idx, eye_w_idx, eye_h_idx;

	shai_v2(speced_csv &&speced,
	        const char *timing_lbl,
	        const char *delta_lbl)
	: shai(move(speced), timing_lbl, delta_lbl)
	, trad_idx(add_column(trad_entry()))
	, eye_w_idx(add_column(eye_w_entry()))
	, eye_h_idx(add_column(eye_h_entry()))
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
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# define PY_ARRAY_UNIQUE_SYMBOL smlp_ARRAY_API
# include <numpy/arrayobject.h>
static struct init_numpy {
	init_numpy() {
#if 0
		/* we don't want Python to steal our signal handlers... */
		Py_InitializeEx(0);
#endif

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
	size_t width;
	size_t const_width;
	const ::smlp_spec *spec;
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

#define FAIL(code,...) do {                                                    \
		return asprintf(&sh->error, __VA_ARGS__) == -1                 \
		       ? errno ? -errno : -ENOMEM : code;                      \
	} while (0)

int smlp_mrc_shai_prep_init(struct smlp_mrc_shai *sh,
                            const struct smlp_mrc_shai_params *par,
                            unsigned flags)
{
	if (!par->spec_in_path)
		FAIL(1,"IN.spec not set, use option '-s'");

	if (par->v1 && strcmp(par->v1, "rx") && strcmp(par->v1, "tx"))
		FAIL(1,"parameter '%s' to '-1' is neither 'rx' nor 'tx'",par->v1);

	FILE *csv_in = par->csv_in_path ? fopen(par->csv_in_path, "r") : stdin;
	if (!csv_in)
		FAIL(1,"%s: %s",par->csv_in_path,strerror(errno));

	try {
		smlp::speced_csv sp(csv_in, smlp::specification(par->spec_in_path));

		if (log(1,nullptr))
			for (size_t i=0; i<sp.width(); i++) {
				const auto &[a,e] = sp.col(i);
				log(1, "  %16s: %u byte(s) %s ~ %.1f MiB\n",
				    e.label, 1 << a.log_bytes, smlp::to_str(a.dty),
				    (sp.height() << a.log_bytes) / 1024.0 / 1024.0);
			}

		if (par->v1) {
			sh->shai = new shai_v1(move(sp), par->timing_lbl,
			                       par->delta_lbl,
			                       !strcmp(par->v1, "rx"));
			sh->const_width = sh->shai->width() - shai_v1::ADDED_COLS;
		} else {
			sh->shai = new shai_v2(move(sp), par->timing_lbl,
			                       par->delta_lbl);
			sh->const_width = sh->shai->width() - shai_v2::ADDED_COLS;
		}

	} catch (const std::runtime_error &ex) {
		FAIL(1,"%s", ex.what());
	}

	if (par->spec_out_path) {
		FILE *f = fopen(par->spec_out_path, "w");
		if (!f)
			FAIL(1,"%s: %s",par->spec_out_path,strerror(errno));
		smlp_spec_write(&sh->shai->spec(), f);
		fclose(f);
	}

	sh->np_cols = NULL;
	sh->np_idcs = NULL;
	sh->width = sh->shai->width();
	sh->error = NULL;
	sh->spec = &sh->shai->spec();

	if (~flags & SMLP_MRC_SHAI_PREP_NO_PY) {
#ifdef SMLP_PY
		// Py_BEGIN_ALLOW_THREADS
		PyGILState_STATE gstate = PyGILState_Ensure();

		sh->np_cols = (PyObject **)malloc(sizeof(*sh->np_cols) * sh->shai->width());
		npy_intp dim = sh->shai->height();
		for (size_t j=0; j<sh->shai->width(); j++) {
			int ty = smlp::with(sh->shai->column(j), overloaded {
				[](int8_t  *){ return NPY_INT8; },
				[](int16_t *){ return NPY_INT16; },
				[](int32_t *){ return NPY_INT32; },
				[](int64_t *){ return NPY_INT64; },
				[](float   *){ return NPY_FLOAT32; },
				[](double  *){ return NPY_FLOAT64; },
			});
			sh->np_cols[j] = PyArray_SimpleNewFromData(1, &dim, ty,
				sh->shai->column(j).v);
		}
		dim = sh->shai->index.size();
		static_assert(std::is_same_v<size_t,uint64_t>);
		sh->np_idcs = PyArray_SimpleNewFromData(1, &dim, NPY_UINT64,
		                                        sh->shai->index.data());
		fprintf(stderr, "init idcs refcnt: %zd\n", Py_REFCNT(sh->np_idcs));

		PyGILState_Release(gstate);
		// Py_END_ALLOW_THREADS
#else
		FAIL(1,"smlp was compiled without python support");
#endif
	}

	return 0;
}

void smlp_mrc_shai_prep_fini(struct smlp_mrc_shai *sh)
{
	free(sh->error);

#ifdef SMLP_PY
	// Py_BEGIN_ALLOW_THREADS
	PyGILState_STATE gstate = PyGILState_Ensure();

	if (sh->np_cols)
		for (size_t i=0; i<sh->shai->width(); i++) {
			// PyObject_Del(sh->np_cols[i]);
			Py_DECREF(sh->np_cols[i]);
		}
	if (sh->np_idcs) {
		fprintf(stderr, "fini idcs refcnt: %zd\n", Py_REFCNT(sh->np_idcs));
		//PyObject_Del(sh->np_idcs);
		Py_DECREF(sh->np_idcs);
	}

	PyGILState_Release(gstate);
	// Py_END_ALLOW_THREADS
	free(sh->np_cols);
#endif

	delete sh->shai;
}

void smlp_mrc_shai_prep_rad(struct smlp_mrc_shai *sh, int trad)
{
	sh->shai->prepare(trad);
}

}
