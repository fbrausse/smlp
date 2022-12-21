/*
 * safe.cc
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 *
 * See the LICENSE file for terms of distribution.
 */

#define IV_DEBUG(x)

#include <cassert>
#include <unistd.h>	/* getopt() */
#include <charconv>
#include <csignal>	/* sig_atomic_t */
#include <iostream>	/* std::cout, std::cerr */

#include <iv/ival.hh>
#include <iv/safe.hh>

#define DIE(code,...) do { fprintf(stderr, __VA_ARGS__); exit(code); } while (0)

using namespace iv;

using std::vector;
using std::min;
using std::max;
using std::cerr;
using std::cout;

static const rounding_mode rnd(FE_DOWNWARD);
static volatile std::sig_atomic_t signalled = 0;

#ifdef _WIN32
# define install_signal_handlers(...)
#else
static void signal_handler(int sig, siginfo_t *info, void *uctxt)
{
	signalled = sig;
}

static void install_signal_handlers(std::initializer_list<int> ss)
{
	struct sigaction act;
	memset(&act, 0, sizeof(act));
	act.sa_sigaction = signal_handler;
	act.sa_flags = SA_RESETHAND | SA_SIGINFO;
	for (int s : ss)
		if (sigaction(s, &act, NULL) < 0)
			std::cerr << "ignoring error registering handler for "
			             "signal " << s << ": " << strerror(errno)
			          << "\n";
}
#endif

namespace {

class model_fun {
	iv_model_fun *p;
public:
	explicit model_fun(const char *gen_path, const char *hdf5_path,
	                   const char *spec_path, const char *io_bounds,
	                   char *err = NULL)
	: p(iv_model_fun_create(gen_path, hdf5_path, spec_path, io_bounds, &err))
	{
		if (!p) {
			std::string msg(err);
			free(err);
			throw std::runtime_error(msg);
		}
	}

	~model_fun() { if (p) iv_model_fun_destroy(p); }

	model_fun(model_fun &&o) : p(o.p) { o.p = nullptr; }
	model_fun & operator=(model_fun o) { using std::swap; swap(p, o.p); return *this; }

	operator iv_model_fun *() const noexcept { return p; }
};

struct PR {

	size_t n;
	ival y;

	PR(size_t n, ival y)
	: n(std::move(n))
	, y(std::move(y))
	{}

	friend bool operator>(const PR &a, const PR &b)
	{
		using std::pair;
		return pair { lo(a.y), hi(a.y) } > pair { lo(b.y), hi(b.y) };
	}
};

struct collect_upto_capacity {

	vector<PR> &res;
	mutable size_t n = 0;
	mutable size_t below = 0;
	double cutoff;

	collect_upto_capacity(vector<PR> &res, double cutoff = -INFINITY)
	: res(res)
	, cutoff(cutoff)
	{}

	~collect_upto_capacity()
	{
		std::cerr << "evaluated NN on " << n
		          << " grid points, collected "
		          << size(res) << " points, < " << cutoff
		          << ": " << below << "\n";
	}

	void operator()(size_t n, ival y) const
	{
		this->n = n;

		if (hi(y) < cutoff) {
			below++;
			return;
		}

		PR v { n, y };
		auto lt = [](const PR &a, const PR &b) { return a > b; };
		auto it = std::lower_bound(res.begin(), res.end(), v, lt);
		if ((size_t)(it - res.begin()) >= res.capacity())
			return;
		if (res.size() == res.capacity())
			res.pop_back();
		res.insert(it, std::move(v));
	}
};

}

static void eval_dataset(const iv_target_function &tf, const char *dataset_path)
{
#if 0
	auto output_ival = [](ival y){ std::cout << y << "\n"; };
	auto eval = [&tf](auto x){ return tf_eval_ival(tf, x); };
	forall_in_dataset(*tf.mf, File(dataset_path, "r"), output_ival)(eval);
#else
	iv_tf_eval_dataset(&tf, File(dataset_path, "r"),
	                   [](double lo, double hi, void *){
		std::cout << ival { endpts { lo, hi } } << "\n";
	}, NULL);
#endif
}

static void eval_grid(const iv_target_function &tf, size_t grid,
                      const char *th_str, double grid_cutoff)
{
	const iv_model_fun *mf = tf.mf;

	if (!iv_model_fun_has_grid(mf))
		DIE(1,"finite grid not deducible from given SPEC and IO_BNDS\n");
	// std::cerr << "grid: " << *m.grid << "\n";

	vector<PR> res;
	if (th_str) {
		struct cb_data {
			vector<PR> &res;
			double th;
			size_t n = 0;
		} data { res, atof(th_str) };
		iv_tf_eval_grid(&tf, [](double lo, double hi, void *udata){
			cb_data &d = *static_cast<cb_data *>(udata);
			if (hi >= d.th)
				d.res.emplace_back(d.n, endpts { lo, hi });
			d.n++;
		}, &data);
	} else {
		res.reserve(grid);
		struct cb_data : collect_upto_capacity {
			using collect_upto_capacity::collect_upto_capacity;
			using collect_upto_capacity::operator();
			size_t n = 0;
		} data(res, grid_cutoff);
		iv_tf_eval_grid(&tf, [](double lo, double hi, void *udata){
			cb_data &d = *static_cast<cb_data *>(udata);
			d(d.n++, endpts { lo, hi });
		}, &data);
	}

	size_t N = iv_mf_input_dim(mf);
	for (size_t i=0; i<N; i++)
		std::cout << (i ? "," : "") << iv_mf_dom_label(mf, i, NULL);
	std::cout << ",delta-lo,delta-hi\n";
	vector<iv_ival> x(N);
	for (const auto &[n,y] : res) {
		iv_model_fun_grid_sget_ival(mf, n, x.data());
		for (size_t i=0; i<N; i++) {
			const iv_ival &v = x[i];
			// assert(ispoint(ival { endpts { v.lo, v.hi } }));
			assert(v.lo == v.hi);
			std::cout << (i ? "," : "") << v.lo;
		}
		std::cout << "," << lo(y) << "," << hi(y) << "\n";
	}
}

static void search_threshold(const iv_target_function &tf, double prec,
                             double th_width, const char *th_str,
                             const char *regions_path)
{
	install_signal_handlers({ SIGINT, SIGALRM });

	/* Computes an interval for threshold T on region
	 * centered at argument */

	auto g = [&,th=th_str ? atof(th_str) : INFINITY](auto x) -> ival {
		iv_ival sup, sub;
		vector<iv_ival> x_;
		x_.reserve(x.size());
		for (const ival &c : x)
			x_.push_back(iv_ival { lo(c), hi(c) });
		int r = iv_tf_search_supsub_min(&tf, x_.data(), prec, th_width,
		                                th, &signalled, &sup, &sub);
		if (r <= 0)
			std::cerr << "signalled, search not complete: " << r << "\n";
		if (r < 0)
			raise(signalled);
		return endpts { sup.lo, sub.lo };
	};

	auto output_ival = [](ival y){ std::cout << y << "\n"; };

	forall_in_dataset(*tf.mf, regions_path, output_ival)(g);
}

static void usage(int code, const char *progname)
{
	FILE *f = code ? stderr : stdout;
	fprintf(f, "usage: %s [-OPTS] MODEL.h5 DATA.spec MODEL.gen\n",
	        progname);
	if (!code)
		fprintf(f, "%s", "\n\
Options affecting f(x) [default]:\n\
  MODEL.h5     sequential keras model in hdf5 format containing only dense layers\n\
               with activations either 'linear' or 'relu'\n\
  DATA.spec    specification of the domain and the grid and regions in JSON\n\
  MODEL.gen    additional definition for MODEL: normalizations, output labels\n\
  -c           clamp inputs to MODEL to [0,1] [don't clamp]\n\
  -i IN_BNDS   scale input according to min-max input bounds [none]\n\
  -o OUT_BNDS  scale output according to min-max output bounds [none]\n\
\n\
Threshold search:\n\
  -p PREC      starting value for the search for a limit on the width of\n\
               f(R)=[lo,hi], until which splitting of R is performed [1]\n\
  -r REGIONS   perform 'threshold search' for all regions R around the centers\n\
               in REGIONS CSV [read centers from stdin (w/o CSV header)]\n\
  -t THRESH    stop 'threshold search' for I as soon as THRESH is not contained\n\
               in I anymore.\n\
  -w TH_WIDTH  stop 'threshold search' when width of I is <= TH_WIDTH [∞]\n\
\n\
Exhaustive grid search:\n\
  -C CUTOFF    consider only grid points evaluating [lo,hi] with hi > CUTOFF\n\
               (<= 5% speed-up for grid search)\n\
  -g GRID_N    evaluate all grid points and output the highest N ones including\n\
               the endpoints of [lo,hi] the objective function evaluated to\n\
  -t THRESH    collect only those (x,y) with hi(y) >= THRESH, ignore GRID_N.\n\
\n\
Data search:\n\
  -d DATA      evaluate all points from the DATA CSV file\n\
\n\
Other options:\n\
  -h           display this help message\n\
  -T TIMEOUT   send SIGALRM to itself after TIMEOUT seconds to stop the\n\
               threshold search; it's result will still be printed, but may not\n\
               satisfy the additional conditions '-t THRESH' or '-w TH_WIDTH'.\n\
\n\
The default action is the 'threshold search', which searches for an interval I\n\
that contains T such that for all x in region R, f(x) >= T.\n\
\n\
One of the 2 other actions is selected by '-d DATA' and '-g GRID_N',\n\
respectively.\n\
");
	exit(code);
}

int main(int argc, char **argv)
{
	const char *in_bounds = NULL, *out_bounds = NULL;
	bool clamp_inputs = false;

	const char *dataset_path = NULL;
	const char *regions_path = NULL;
	char *th_str = NULL;
	size_t grid = 0;
	double grid_cutoff = -INFINITY;
	double prec = 1;
	double th_width = INFINITY;
	for (int opt; (opt = getopt(argc, argv, ":cC:d:g:hi:o:p:r:t:T:w:")) != -1;)
		switch (opt) {
		case 'c': clamp_inputs = true; break;
		case 'C': grid_cutoff = atof(optarg); break;
		case 'd': dataset_path = optarg; break;
		case 'g': grid = (size_t)atoll(optarg); break;
		case 'h': usage(0, argv[0]);
		case 'i': in_bounds = optarg; break;
		case 'o': out_bounds = optarg; break;
		case 'p': prec = atof(optarg); break;
		case 'r': regions_path = optarg; break;
		case 't': th_str = optarg; break;
#ifndef _WIN32
		case 'T': alarm(atoi(optarg)); break;
#endif
		case 'w': th_width = atof(optarg); break;
		case ':': DIE(1,"error: option '-%c' expects an argument\n", optopt);
		case '?': DIE(1,"error: unknown option '-%c'\n", optopt);
		}
	if (argc - optind != 3)
		usage(1, argv[0]);

	try {
		const char *hdf5_path = argv[optind+0];
		const char *spec_path = argv[optind+1];
		const char *gen_path = argv[optind+2];
		model_fun mf(gen_path, hdf5_path, spec_path, in_bounds);
		iv_target_function tf = { mf, clamp_inputs, out_bounds };

		if (dataset_path) {
			eval_dataset(tf, dataset_path);
		} else if (grid) {
			eval_grid(tf, grid, th_str, grid_cutoff);
		} else {
			search_threshold(tf, prec, th_width, th_str,
			                 regions_path);
		}
	} catch (const std::exception &s) {
		DIE(1, "error: %s\n", s.what());
	}
}
