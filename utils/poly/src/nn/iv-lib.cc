/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2020 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2020 The University of Manchester
 */

#if 0 /* taken care of # define IV_ANON_NN_COMMON 1 below */
#ifndef IV_ANON_FUNCTIONS
# define IV_ANON_FUNCTIONS 1 /* speed-up of > 20% */
#endif
#ifndef IV_ANON_NN_MODEL
# define IV_ANON_NN_MODEL 1 /* additional speed-up of 10%, separate: > 20% */
#endif
#endif

#ifndef IV_ANON_NN_COMMON
# define IV_ANON_NN_COMMON 1
#endif

#include <iv/safe.h>
#include <iv/safe.hh>
#include "nn-common.hh"
#include <iv/tm.hh>

#include <deque>
#include <stack>
#include <functional>

#ifndef SAFE_R_TYPE
# define SAFE_R_TYPE		ival
#endif

#ifndef SAFE_UNROLL_MAX
# define SAFE_UNROLL_MAX	3
#endif

#if !(IV_ANON_FUNCTIONS-0)
using namespace IV_NS_FUNCTIONS;
#endif
#if !(IV_ANON_NN_MODEL-0)
using namespace IV_NS_NN_MODEL;
#endif
#if !(IV_ANON_NN_COMMON-0)
using namespace IV_NS_NN_COMMON;
#endif

using namespace iv;

extern "C"
struct iv_model_fun : IV_NS_NN_COMMON::model_fun2 {
	using model_fun2::model_fun2;
};

extern "C"
iv_model_fun * iv_model_fun_create(const char *gen_path,
                                   const char *keras_hdf5_path,
                                   const char *spec_path,
                                   const char *io_bounds, char **err)
{
	try {
		return new iv_model_fun(gen_path, keras_hdf5_path, spec_path,
		                        io_bounds);
	} catch (const std::exception &ex) {
		if (err)
			*err = strdup(ex.what());
		return NULL;
	}
}

extern "C" void iv_model_fun_destroy(iv_model_fun *f) { delete f; }


extern "C" size_t iv_mf_input_dim(const iv_model_fun *mf)
{
	return input_dim(mf->spec);
}

extern "C"
const char * iv_mf_dom_label(const iv_model_fun *mf, size_t i, size_t *len)
{
	return len ? dom_label_cstr(mf->spec, i, *len)
	           : dom_label_cstr(mf->spec, i);
}

extern "C"
int     iv_model_fun_has_grid(const iv_model_fun *f)
{
	return f->grid ? true : false;
}

extern "C"
ssize_t iv_model_fun_grid_get_rank(const iv_model_fun *f)
{
	return f->grid ? f->grid->dim() : -EINVAL;
}

extern "C"
int     iv_model_fun_grid_get_dims(const iv_model_fun *f, size_t *dims)
{
	if (!f->grid)
		return -EINVAL;
	for (const auto &u : f->grid->g)
		*dims++ = u.size();
	return 0;
}

extern "C"
ssize_t iv_model_fun_grid_size(const iv_model_fun *f)
{
	if (!f->grid)
		return -EINVAL;
	Z sz = size(*f->grid);
	if (sz > std::min<size_t>(SSIZE_MAX, ULONG_MAX))
		return -ERANGE;
	return sz % (ulong)std::min<size_t>(SSIZE_MAX, ULONG_MAX);
}

extern "C"
int iv_model_fun_grid_sget_ival(const iv_model_fun *f, size_t grid_idx,
                                iv_ival v[STATIC(iv_model_fun_grid_get_rank(f))])
{
	if (!f->grid)
		return -EINVAL;
	for (const ival &w : (*f->grid)[grid_idx])
		*v++ = { lo(w), hi(w) };
	return 0;
}

namespace iv {
template <> inline double unbox0(const ival &v, lambda) { return mid(v); }
}

namespace {

[[maybe_unused]] ival box(double a) { return ival(a); }

template <typename T>
vector<ival> box(const vector<T> &w)
{
	vector<ival> r;
	for (const T &v : w)
		r.push_back(box(v));
	return r;
}

template <typename F>
auto with1(const iv_target_function &tf, F &&f)
{
	return with<SAFE_UNROLL_MAX>(*tf.mf, std::forward<F>(f),
	                             tf.clamp_inputs, tf.out_bounds);
}

[[maybe_unused]]
auto with_pt(const iv_target_function &tf)
{
	using R = std::function<ival(finite_product<ival>)>;
	return with1(tf, [](auto g) -> R { return g; });
}

template <typename X>
auto with_pt(const iv_target_function &tf, X &&x)
{
	return with1(tf, [&](auto g){ return g(std::forward<X>(x)); });
}

static finite_product<ival> mk_product(const iv_ival *x, size_t n)
{
	finite_product<ival> x_;
	x_.reserve(n);
	for (size_t i=0; i<n; i++)
		x_.emplace_back(endpts { x[i].lo, x[i].hi });
	return x_;
}

static finite_product<ival> mk_product(const iv_ival *x,
                                       const specification &spec)
{
	dim d = input_dim(spec);
	assert(d >= 0);
	return mk_product(x, (size_t)d);
}

template <typename HRes>
struct forall_in_grid {

	const grid_t<ival> &grid;
	HRes handle_result;
	Z m;

	forall_in_grid(const grid_t<ival> &grid, HRes handle_result)
	: grid(grid)
	, handle_result(handle_result)
	, m(size(this->grid))
	{}

	template <typename G>
	void operator()(G &&g) const
	{
		size_t n = 0;
		forall_in(grid, [&](const finite_product<ival> &r) {
			handle_result(n, g(std::move(r)));
			if (!(++n % 100000))
				std::cerr << n << " / " << m << "\r";
		});
	}
};

} /* end anon ns */

ival iv::tf_eval_ival(const iv_target_function &tf, const vector<ival> &x)
{
	return with_pt(tf, x);
}

ival iv::tf_eval_ival(const iv_target_function &tf,       vector<ival> &&x)
{
	return with_pt(tf, std::move(x));
}

extern "C"
void iv_tf_eval_ival(const iv_target_function *f, const iv_ival *x, iv_ival *y)
{
	ival y_ = tf_eval_ival(*f, mk_product(x, f->mf->spec));
	y->lo = lo(y_);
	y->hi = hi(y_);
}

extern "C"
int iv_tf_eval_grid(const iv_target_function *f,
                    void (*handle_next)(double lo, double hi, void *udata),
                    void *udata)
{
	if (!f->mf->grid)
		return -EINVAL;
	with1(*f, forall_in_grid(*f->mf->grid,
	                         [handle_next,udata](size_t n, ival y)
	                         { handle_next(lo(y), hi(y), udata); }));
	return 0;
}

extern "C"
int iv_tf_eval_dataset(const iv_target_function *f, FILE *dataset,
                       void (*handle_res)(double lo, double hi, void *udata),
                       void *udata)
{
	try {
		with1(*f, forall_in_dataset(*f->mf, dataset,
	                                    [handle_res,udata](ival y)
	                                    { handle_res(lo(y), hi(y), udata); }));
		return 0;
	} catch (const iv::table_exception &ex) {
		return ex.table_ret;
	}
}

namespace {

template <typename R>
struct eval_types {

	static R unbox(const ival &a, lambda l) { return unbox0<R>(a, l); }

	static vector<R> unbox(const finite_product<ival> &a)
	{
		vector<R> x;
		for (size_t i=0; i<size(a); i++)
			x.push_back(unbox(a[i], lambda{i}));
		return x;
	}

	template <typename F>
	struct unboxed {

		F f;

		unboxed(F &&f)
		: f(move(f))
		{}

		friend dim input_dim(const unboxed &ub) { return input_dim(ub.f); }
		friend dim output_dim(const unboxed &ub) { return output_dim(ub.f); }

		template <typename T>
		friend auto evaluate(const unboxed &ub, T &&x)
		{
			return box(evaluate(ub.f, unbox(x)));
		}

		friend ostream & operator<<(ostream &os, const unboxed &ub)
		{
			return os << "(box ∘ " << ub.f << " ∘ unbox)";
		}

		auto operator()(const finite_product<ival> &x) const
		{
			return box(f(unbox(x)));
		}

		ostream & print_fun(ostream &os) const
		{
			return os << *this;
		}
	};

//	template <typename F> unboxed(F) -> unboxed<std::remove_cv_t<F>>;

	/* workaround the inability to specify the above template deduction
	 * guide due to GCC bug #86403
	 * <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=86403>. */
	template <typename F>
	static auto unboxed2(F &&f) {
		return unboxed<std::remove_cv_t<F>>(std::forward<F>(f));
	}
};

/* when R == iv::ival, box() and unbox() and therefore unboxed as well are the
 * identity; omit these calls. */
template <>
template <typename F>
struct eval_types<ival>::unboxed : F {
	unboxed(F &&f) : F(move(f)) {}
};

template <typename F, typename Split1,
          typename T = ival, typename X = finite_product<T>,
          typename Y = std::invoke_result_t<F,X>,
          typename List = std::deque<X>
         >
class region_eval {

	void split(std::stack<X,List> &todo, X x, size_t idx=0,
	           bool ispoint_sofar=true) const
	{
		if (idx == size(x)) {
			assert(!ispoint_sofar);
			todo.push(move(x));
			return;
		}
		if (ispoint(x[idx]))
			return split(todo, move(x), idx+1, ispoint_sofar);
		for (auto w : split1(x[idx])) {
			x[idx] = w;
			split(todo, x, idx+1, false);
		}
	}

public:
	F fun;
	Split1 split1;

	region_eval(F fun = {}, Split1 split1 = {})
	: fun(move(fun))
	, split1(move(split1))
	{
		std::cerr << "fun   : " << this->fun << "\n";
	}

	template <bool verbose, typename Pred>
	void run(finite_union<X> r, Pred pred = {}) const
	{
		std::stack<X,List> todo;
		for (auto &v : r)
			todo.push(move(v));
		size_t itr;
		for (itr=0; !empty(todo); ++itr) {
			if constexpr (verbose)
				if (!(itr & 0xffff))
					std::cerr << itr << ", " << size(todo) << "   \r";
			X x = todo.top();
			todo.pop();
			Y y = fun(x);
			if (pred(move(y)))
				continue;
			split(todo, move(x));
		}
		if constexpr (verbose)
			std::cerr << itr << ", " << size(todo) << "   \n";
	}

	template <bool verbose, typename Pred>
	void run(X r, Pred pred = {}) const
	{
		run<verbose>(finite_union<X> { std::move(r) }, pred);
	}
};

/*
 * Using interval evaluations with subdivisions based on a predicate P,
 * we compute
 *
 * a) an outer approximation 'res' of the function's range, and
 * b) an inner approximation [lub,glb] of the function's range, where
 *    'lub' is the minimum of all upper bounds hi(y) of the interval y and
 *    'glb' is the maximum of all lower bounds lo(y) of the interval y.
 *
 * Invariants: [lub,glb] \subseteq range(f) \subseteq res
 *
 * Once [lub,glb] is not empty, we can start finding out about the threshold T
 * which is the maximum of all T' for which range(f) >= T' - or, more simply -
 * T = min range(f).
 *
 * We then have T \in [lo(res),lub].
 *
 * This explains P = Q \/ R where
 *
 * Q is the conjunuction of
 * - lub <= glb <-> [lub,glb] not empty
 * - lub - lo(res) <= th_width <-> wid([lo(res),lub]) <= th_width
 * - (lub < th || lo(res) >= th) <-> TH \notin [lo(res),lub] \ni T
 *                                -> TH is a lower or an upper bound on T
 *
 * and R is the disjunction of
 * - wid(y) < p: stop splitting when wid(y) is too small; the outer loop
 *   decreasing p together with continuity of f ensures that starting at some i
 *   p_i is small enough such that Q will become true. Once Q is true, not just
 *   splitting of intervals will stop, but also the outer loop searching for i
 *   terminates.
 * - lo(y) >= lub: y is large enough have no effect on (the search for) T.
 *
 * The argument about continuity of f only holds when input intervals (and y)
 * are allowed to get arbitrary small.  This is not the case.  Here, we have
 * 'double's as endpoints and p is a 'double' as well.  In order to accomodate
 * for that case, we note that splitting a point interval results in exactly the
 * same interval leading to an infinite loop.
 */

struct supsubset {
	ival sup;
	ival sub;
	bool correct = false;
	bool valid = false;
};

#define PREC_DIV        2 /* must be > 1 s.t. (p_i)_i -> 0
                             * where p_{i+1} = p_i/PREC_DIV */

template <typename F>
static supsubset range_supsubset(F &&f, double start_prec, double th,
                                 double th_width,
                                 const volatile sig_atomic_t &signalled)
{
	supsubset old;
	old.sup = endpts { -INFINITY, INFINITY };
	uintmax_t was_signalled;
	for (double p = start_prec;; p /= PREC_DIV) { /* slow exponential decrease */
		bool is_empty = true;
		ival res;
		double glb = -INFINITY;
		double lub = INFINITY;
		f([&](ival &&y){
			double glb_ = fmax(glb, lo(y));
			double lub_ = fmin(lub, hi(y));
			if (!is_empty && issubset(y, res)) {
				glb = glb_;
				lub = lub_;
				return true;
			}
			// lo(y) >= th || wid(y) < p */
			// lo(y) >= lub_  -> kein split
			if ((lub_ <= glb_
			     && lub_ <= th_width + fmin(lo(res), lo(y))
			     && (lub_ < th || fmin(lo(res), lo(y)) >= th))
			    || lo(y) >= lub_ || wid(y) < p) {
				res = is_empty ? y : convex_hull(res, y);
				glb = glb_;
				lub = lub_;
				is_empty = false;
				return true;
			}
			if (signalled)
				return true;
			return false;
		});
		assert(!is_empty);
		/* range is subset of res */
		// double lb = -(-lo(res) - p); // lo(res) + p
		// double ub = hi(res) - p;
		double lb = lub;
		double ub = glb;
		was_signalled = signalled;
		if (!was_signalled && !is_empty)
			old.sup = intersect(old.sup, res);
		if (lb <= ub) {
			if (!was_signalled) {
				old.sub = { endpts { lb, ub } };
			} else if (old.valid) {
				old.sub = { endpts {
					fmin(lb, lo(old.sub)),
					fmax(ub, hi(old.sub)),
				} };
			} else {
				old.sub = { endpts { lb, ub } };
			}
			old.valid = true;
		}
		if (was_signalled)
			break;
		if (old.valid && (lub <= th_width+lo(res)
		                  && (lb < th || lo(res) >= th))) {
			/* [lb,ub] is subset of range */
			std::cerr << "success with p: " << p << ": res: " << res
			          << ", lub: " << lub << ", glb: " << glb << "\n";
			old.correct = true;
			return old;
		} else
			std::cerr << "fail with p: " << p << ": res: " << res
			          << ", lub: " << lub << ", glb: " << glb << "\n";
	}
/*
	if (!old_valid)
		raise(was_signalled);
*/
	return old;
}

template <size_t n>
static vector<ival> split_ival(const ival &v);

template <>
vector<ival> split_ival<2>(const ival &v)
{
	double m = mid(v);
	ival a = endpts { lo(v), m };
	ival b = endpts { nextafter(m, INFINITY), hi(v) };
	vector<ival> r = { a };
	if (!isempty(b))
		r.push_back(b);
	return r;
}

} // end anon ns

extern "C"
int iv_tf_region_eval(const iv_target_function *f,
                      const iv_ival c[STATIC(iv_model_fun_grid_get_rank(f))],
                      bool (*split_dom)(double lo, double hi, void *udata),
                      void *udata)
{
	with1(*f,
	      [&,x=f->mf->spec.region_from_center(mk_product(c, f->mf->spec))]
	      (auto g){
		using eval = eval_types<SAFE_R_TYPE>;
		region_eval(eval::unboxed2(move(g)), split_ival<2>)
		.template run<true>(x, [&](ival &&v){
			return split_dom(lo(v), hi(v), udata);
		});
	});
	return 0;
}

extern "C"
int iv_tf_search_supsub_min(const iv_target_function *f, const iv_ival *x,
                            double prec, double tgt_width, double threshold,
                            const volatile sig_atomic_t *signalled,
                            iv_ival *sup, iv_ival *sub)
{
	const model_fun2 &m = *f->mf;
	static const volatile sig_atomic_t sig_def = 0;
	const volatile sig_atomic_t *sig_ref = signalled ? signalled : &sig_def;
	using ret_t = std::function<supsubset(finite_product<ival>)>;
	auto g = with1(*f, [&](auto g) -> ret_t {
		using eval = eval_types<SAFE_R_TYPE>;
		return [threshold,tgt_width,prec,&m,sig_ref
		       ,re = region_eval(eval::unboxed2(move(g)), split_ival<2>)
		       ](finite_product<ival> x){
			return range_supsubset(
					[&,r=m.spec.region_from_center(x)]
					(auto pred)
					{ re.template run<true>(r, pred); },
					prec, threshold, tgt_width, *sig_ref);
			/*
			if (!correct)
				std::cerr << "signalled, search not complete\n";
			return endpts { lo(sup), lo(sub) };
			*/
		};
	});
#ifndef NDEBUG
	try
#endif
	{
		auto [sup_,sub_,correct,valid] = g(mk_product(x, m.spec));
		sup->lo = lo(sup_);
		sup->hi = hi(sup_);
		sub->lo = lo(sub_);
		sub->hi = hi(sub_);
		return valid ? correct : -1;
	}
#ifndef NDEBUG
	catch (const std::exception &ex) {
		std::cerr << "libiv error: " << ex.what() << "\n";
		assert(false);
	}
#endif
}

iv::detail::forall_in_dataset_base::forall_in_dataset_base(const iv_model_fun &mf,
                                                           FILE *dataset)
: t(dataset, true)
{
	size_t n = input_dim(mf.spec);
	dom_idcs.reserve(n);
	for (size_t i=0; i<n; i++) {
		ssize_t j = smlp_table_col_idx(&t, dom_label_cstr(mf.spec, i));
		assert(j >= 0);
		dom_idcs.push_back(j);
	}
	assert(compatible(dim(size(dom_idcs)), input_dim(mf.spec)));
}
