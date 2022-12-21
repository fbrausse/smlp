/*
 * nn-common.hh
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 *
 * See the LICENSE file for terms of distribution.
 */

#ifndef IV_NN_COMMON_HH
#define IV_NN_COMMON_HH

#if (IV_ANON_NN_COMMON-0)
# define IV_NS_NN_COMMON
#else
# define IV_NS_NN_COMMON iv::nn::common
#endif

/* Prevent sub-objects '{in,out}_scaler' and 'model' of model_fun2 to be
 * without linkage when model_fun2 itself has linkage. */
#ifndef IV_ANON_FUNCTIONS
# define IV_ANON_FUNCTIONS	IV_ANON_NN_COMMON
#endif
#ifndef IV_ANON_NN_MODEL
# define IV_ANON_NN_MODEL	IV_ANON_NN_COMMON
#endif

#include "nn-model.hh"
#include "functions.hh"

#include <iv/ival.hh>
#include <iv/safe.hh>

#include <smlp/table.h>
#include <smlp/response.h>

#include <iostream>
#include <fstream>

namespace IV_NS_NN_COMMON {

#if !(IV_ANON_FUNCTIONS-0)
using namespace IV_NS_FUNCTIONS;
#endif
#if !(IV_ANON_NN_MODEL-0)
using namespace IV_NS_NN_MODEL;
#endif

using iv::Z;
using iv::Q;
using iv::ival;
using iv::cnt_rad;
using iv::endpts;
using iv::Table;
using iv::File;

using kjson::json;

using std::move;
using std::vector;
using std::ostream;

/* workaround missing from_chars(..., double &) library support */
struct Double {

	double d;

	friend auto from_chars(const char *a, const char *end, Double &r)
	{
		char *ep;
		std::string s(a, end - a);
		errno = 0;
		r.d = strtod(s.c_str(), &ep);
		return std::pair { a + (ep - s.c_str()), std::errc(errno) };
	}
};

} // end ns IV_NS_NN_COMMON

template <> struct kjson::requests_number<IV_NS_NN_COMMON::Double> : std::true_type {};
template <> struct kjson::requests_number<IV_NS_NN_COMMON::Q> : std::true_type {};

namespace IV_NS_NN_COMMON {
inline auto from_chars(const char *a, const char *end, Q &r)
{
	r = kay::Q_from_str(std::string(a, end - a).data());
	return std::pair { end, std::errc() };
}
}
using IV_NS_NN_COMMON::from_chars;
namespace kay::flintxx {
	using IV_NS_NN_COMMON::from_chars;
}

namespace IV_NS_NN_COMMON {

static json json_parse(const char *path)
{
	if (std::ifstream in(path); in) try {
		return json::parse(in);
	} catch (const std::exception &ex) {
		std::stringstream ss;
		ss << path << ": " << ex.what();
		throw std::runtime_error(ss.str());
	} else
		throw std::system_error(errno, std::system_category(), path);
}

template <typename T>
using affine_scaler = pointwise<affine1<T>>;


template <typename V>
struct finite_product : vector<V> {

	using vector<V>::vector;
	using vector<V>::operator[];

	friend ostream & operator<<(ostream &os, const finite_product &u)
	{
		if (u.size() > 1)
			os << "(";
		bool first = true;
		for (const auto &s : u) {
			if (!first)
				os << " × ";
			os << s;
			first = false;
		}
		if (u.size() > 1)
			os << ")";
		return os;
	}

	friend finite_product product(finite_product p, const V &q)
	{
		p.push_back(q);
		return p;
	}
};

template <typename S>
struct finite_union : protected vector<S> {

	using vector<S>::vector;
	using vector<S>::push_back;
	using vector<S>::emplace_back;
	using vector<S>::begin;
	using vector<S>::end;
	using vector<S>::cbegin;
	using vector<S>::cend;
	using vector<S>::size;
	using vector<S>::empty;
	using vector<S>::front;
	using vector<S>::back;
	using vector<S>::operator[];

	friend ostream & operator<<(ostream &os, const finite_union &u)
	{
		if (u.empty())
			return os << "∅";
		if (u.size() > 1)
			os << "(";
		bool first = true;
		for (const auto &s : u) {
			if (!first)
				os << " ∪ ";
			os << s;
			first = false;
		}
		if (u.size() > 1)
			os << ")";
		return os;
	}
};

template <typename V>
finite_union<finite_product<V>> product(const finite_union<finite_product<V>> &a,
                                        const finite_union<V> &b)
{
	finite_union<finite_product<V>> r;
	for (const V &vb : b)
		for (finite_product<V> va : a) {
			va.push_back(vb);
			r.push_back(move(va));
		}
	return r;
}

typedef finite_union<finite_product<ival>> region;

template <typename V, typename F>
static void forall_products(const finite_product<finite_union<V>> &p, F f,
                            finite_product<V> q = {})
{
	if (p.size() == q.size())
		f(std::move(q));
	else
		for (const V &r : p[q.size()])
			forall_products(p, f, product(q, r));
}

template <typename T>
struct grid_t {

	finite_product<finite_union<T>> g;

	friend Z size(const grid_t &g)
	{
		Z r = 1;
		for (const auto &u : g.g)
			r *= u.size();
		return r;
	}

	size_t dim() const
	{
		return g.size();
	}

	template <typename Eq>
	Z idx_of(const finite_product<T> &v, Eq &&eq) const
	{
		Z r;
		for (size_t i=0; i<this->dim();) {
			auto b = g[i].begin();
			auto e = g[i].end();
			auto k = find_if(b, e, [&](const T &w){ return eq(v[i], w); });
			if (k == e)
				return -1;
			r += k - b;
			if (++i < this->dim())
				r *= g[i].size();
		}
		return r;
	}

	Z idx_of(const finite_product<T> &v) const
	{
		return idx_of(v, [](const T &a, const T &b){ return a == b; });
	}

	finite_product<T> operator[](Z k) const
	{
		finite_product<T> r(this->dim());
		for (size_t i=this->dim(); i; i--) {
			r[i-1] = g[i-1][(size_t)(k % g[i-1].size())];
			k /= g[i-1].size();
		}
		return r;
	}

	template <typename F>
	friend void forall_in(const grid_t &g, F &&f)
	{
		return forall_products(g.g, std::forward<F>(f));
	}

	friend ostream & operator<<(ostream &os, const grid_t &g)
	{
		return os << g.g;
	}
};

template <bool r>
struct constant_predicate {
	template <typename... Args>
	bool operator()(Args &&... args) const { return r; }
};

struct spec_exception : std::runtime_error {

	using std::runtime_error::runtime_error;
};

class specification {

	finite_union<ival> region1(size_t i, const Q &center) const
	{
		using kay::floor;
		using kay::ceil;

		auto jsonnum2str = [](auto x){ return std::string(x.get_number_rep()); };
		auto json2Q = [&](auto x){ return kay::Q_from_str(jsonnum2str(x).data()); };

		auto sp = spec[dom2spec[i]];
		auto type = sp["range"];
		if (type == "int") {
			if (center.get_den() != 1) {
				std::stringstream ss;
				ss << "center '" << center << "' for region of "
				      "'range' == 'int' is not an integer\n";
				throw std::invalid_argument(ss.str());
			}
			Z c = center.get_num();
			Z lo, hi;
			if (sp.contains("rad-abs")) {
				Z rz = Z(jsonnum2str(sp["rad-abs"]).c_str());
				lo = c - rz;
				hi = c + rz - 1;
			} else {
				Q rq = json2Q(sp["rad-rel"]);
				if (c)
					rq *= c;
				lo = ceil(c - rq);
				hi = floor(c + rq);
			}
			finite_union<ival> r;
			for (; lo <= hi; ++lo)
				r.push_back(ival(lo));
			return r;
		} else if (type == "float") {
			Q v = center;
			double r;
			if (sp.contains("rad-abs")) {
				r = hi(ival(json2Q(sp["rad-abs"])));
			} else {
				Q rq = json2Q(sp["rad-rel"]);
				if (v)
					rq *= abs(v);
				r = hi(ival(rq));
			}
			return { ival(v) + cnt_rad { 0.0, r } };
		} else
			throw spec_exception("invalid type '" +
			                     type.get<std::string>() + "'");
	}

	template <typename B>
	ival bounds(const B &bounds, size_t i) const
	{
		const auto &b = bounds[spec[i]["label"].get<std::string>()];
		return endpts {
			b["min"].template get<Double>().d,
			b["max"].template get<Double>().d,
		};
	}

	template <bool norm, typename B>
	affine_scaler<double> scaler(const B &bnds,
	                             const vector<size_t> &spec_idcs) const
	{
		vector<affine1<double>> sc;
		for (size_t i : spec_idcs) {
			ival b = bounds(bnds, i);
			sc.push_back(minmax_scaler<norm>(lo(b), hi(b)));
		}
		return { move(sc) };
	}

	template <typename ResponseFilter>
	static std::tuple<json,vector<size_t>,vector<size_t>>
	make(const char *spec_path, ResponseFilter resp_filter)
	{
		json spec = json_parse(spec_path);
		vector<size_t> dom2spec;
		vector<size_t> resp2spec;
		size_t i = 0;
		for (auto it = begin(spec); it != end(spec); i++, ++it) {
			if ((*it)["type"] == "response") {
				if (resp_filter((*it)["label"].get<std::string>()))
					resp2spec.push_back(i);
			} else {
				auto rng = (*it)["range"];
				assert(rng == "int" || rng == "float");
				assert((*it)["type"] == "input" ||
				       it->contains("rad-abs") ||
				       it->contains("rad-rel"));
				dom2spec.push_back(i);
			}
		}
		return { move(spec), move(dom2spec), move(resp2spec) };
	}

	specification(std::tuple<json,vector<size_t>,vector<size_t>> items)
	: spec(move(std::get<0>(items)))
	, dom2spec(move(std::get<1>(items)))
	, resp2spec(move(std::get<2>(items)))
	{}

public:
	const json spec;
	const vector<size_t> dom2spec;
	const vector<size_t> resp2spec;

	template <typename ResponseFilter = constant_predicate<true>>
	specification(const char *spec_path,
	              ResponseFilter resp_filter = ResponseFilter {})
	: specification(make(spec_path, resp_filter))
	{}

	friend auto input(const specification &)
	{
	}

	friend dim input_dim(const specification &s) { return (dim)size(s.dom2spec); }
	friend dim output_dim(const specification &s) { return (dim)size(s.resp2spec); }

	template <bool norm, typename T>
	affine_scaler<double> in_scaler(const T &bounds) const
	{
		return scaler<norm>(bounds, dom2spec);
	}

	template <bool norm, typename T>
	affine_scaler<double> out_scaler(const T &bounds) const
	{
		return scaler<norm>(bounds, resp2spec);
	}

	friend const char * dom_label_cstr(const specification &s, size_t i)
	{
		/* guaranteed by kjson */
		return dom_label_sv(s, i).data();
	}

	friend const char * dom_label_cstr(const specification &s, size_t i,
	                                   size_t &len)
	{
		/* guaranteed by kjson */
		std::string_view l = dom_label_sv(s, i);
		len = l.length();
		return l.data();
	}

	friend std::string_view dom_label_sv(const specification &s, size_t i)
	{
		return s.spec[s.dom2spec[i]]["label"].get<std::string_view>();
	}

	friend vector<std::string> dom_labels(const specification &s)
	{
		vector<std::string> r;
		for (size_t i : s.dom2spec)
			r.emplace_back(s.spec[i]["label"].get<std::string_view>());
		return r;
	}

	template <typename Bounds>
	grid_t<ival> grid(const Bounds &in_bounds) const
	{
		auto Q2ival = [](const Q &v){
			return v.get_den() == 1 ? ival(v.get_num())
			                        : ival(v);
		};
		grid_t<ival> n;
		for (size_t i : dom2spec) {
			finite_union<ival> u;
			json si = spec[i];
			if (si.contains("safe")) {
				for (const json &k : si["safe"])
					u.push_back(Q2ival(k.template get<Q>()));
			} else if (si["range"] == "int") {
				auto b = in_bounds[si["label"].get<std::string>()];
				Q qlo = b["min"].template get<Q>();
				Q qhi = b["max"].template get<Q>();
				if (qlo.get_den() != 1 || qhi.get_den() != 1) {
					std::stringstream ss;
					ss << "bounds on '"
					   << si["label"].get<std::string_view>()
					   << "' for 'range' == 'int' are not "
					      "integers: "
					   << qlo << ", " << qhi << "\n";
					throw std::invalid_argument(ss.str());
				}
				Z lo = qlo.get_num();
				Z hi = qhi.get_num();
				std::cerr << "'" << si["label"].get<std::string_view>()
				          << "': min: " << lo << ", max: " << hi << "\n";
				for (; lo <= hi; ++lo)
					u.emplace_back(lo);
			} else {
				std::stringstream ss;
				ss << "dimension " << i << " labelled '"
				   << si["label"].get<std::string_view>()
				   << "' neither has 'safe' grid nor has "
				      "'range' == 'int'";
				throw std::invalid_argument(ss.str());
			}
			n.g.push_back(move(u));
		}
		return n;
	}

	region region_from_center(const finite_product<ival> &row) const
	{
		region r { {} };
		for (size_t i=0; i<dom2spec.size(); i++) {
			std::cerr << "region1 around '" << row[i] << "' is ";
			assert(ispoint(row[i]));
			Q c(lo(row[i]));
			finite_union<ival> reg = region1(i, c);
			std::cerr << reg << "\n";
			r = product(r, reg);
		}
		return r;
	}

	region region_from_center_csv(char *s) const
	{
		char *save;
		size_t i = 0;
		region r;
		r.push_back({});
		for (char *tok = strtok_r(s, ",", &save); tok;
		     tok = strtok_r(NULL, ",", &save), i++) {
			Q c = kay::Q_from_str(tok);
			finite_union<ival> reg = region1(i, c);
			std::cerr << "region1 around '" << tok << "' is " << reg << "\n";
			r = product(r, reg);
		}
		assert((dim)i == input_dim(*this));
		return r;
	}
};

static ssize_t resp_idx(const json &gen, const std::string_view &label)
{
	json resp = gen["response"];
	auto it = find(begin(resp), end(resp), label);
	if (it == end(resp))
		return -1;
	return it - begin(resp);
}

struct Response : ::smlp_response {

	Response(json &gen)
	: ::smlp_response SMLP_RESPONSE_INIT
	{
		std::string o = gen["objective"].get<std::string>();
		int r = smlp_response_parse(this, o.c_str(),
		                            [](const char *label, void *udata)
		                            -> ssize_t {
			return resp_idx(*static_cast<json *>(udata), label);
		                       }, &gen);
		if (r) {
			std::stringstream ss;
			ss << "error " << r << " parsing objective function '"
			   << o << "'";
			throw std::invalid_argument(ss.str());
		}
	}

	friend dim input_dim(const Response &r)
	{
		switch (r.type) {
		case SMLP_RESPONSE_ID: return (dim)1;
		case SMLP_RESPONSE_SUB: return (dim)2;
		default: unreachable();
		}
	}

	friend dim output_dim(const Response &)
	{
		return SCALAR;
	}

	ostream & print_fun(ostream &os) const
	{
		switch (type) {
		case SMLP_RESPONSE_ID:
			return os << "proj_" << idx[0];
		case SMLP_RESPONSE_SUB:
			return os << "(proj_" << idx[0]
			          << "-proj_" << idx[1] << ")";
		default: unreachable();
		}
	}

	friend ostream & operator<<(ostream &os, const Response &r)
	{
		return r.print_fun(os);
	}

	template <typename X>
	X operator()(const vector<X> &v) const
	{
		switch (type) {
		case SMLP_RESPONSE_ID: return v[idx[0]];
		case SMLP_RESPONSE_SUB: return v[idx[0]]-v[idx[1]];
		default: unreachable();
		}
	}
};

struct model_fun2 {

	typedef opt_fun_t<affine_scaler<double>> opt_scaler;

	specification spec;
	sequential_dense model;
	Response objective;
	opt_scaler in_scaler, out_scaler;
	std::optional<grid_t<ival>> grid;

	model_fun2(const char *gen_path, const char *keras_hdf5_path,
	           const char *spec_path, const char *io_bounds)
	: model_fun2(json_parse(gen_path), keras_hdf5_path, spec_path,
	             io_bounds)
	{}

	model_fun2(json gen, const char *keras_hdf5_path, const char *spec_path,
	           const char *io_bounds)
	: spec(spec_path, [&gen](const auto &s){ return resp_idx(gen, s) >= 0; })
	, model(model_from_keras_hdf5(keras_hdf5_path))
	, objective { gen }
	{
		std::cerr << "in: " << input_dim(model) << ", " << input_dim(spec) << "\n";
		std::cerr << "out: " << output_dim(model) << ", " << output_dim(spec) << "\n";
		assert(input_dim(model) == input_dim(spec));
		assert(output_dim(model) == output_dim(spec));

		json pp = gen["pp"];
		int io_scale = (int)(pp["features"] == "min-max") << 0 |
		               (int)(pp["response"] == "min-max") << 1;
		if (io_scale) {
			if (!io_bounds) {
				std::stringstream ss;
				ss << "model.gen defined io-scaler 'min-max', "
				      "need io-bounds";
				throw std::invalid_argument(ss.str());
			}
			json b = json_parse(io_bounds);
			if (io_scale & (1 << 0))
				in_scaler = spec.in_scaler<true>(b);
			if (io_scale & (1 << 1))
				out_scaler = spec.out_scaler<false>(b);
			grid = spec.grid(b);
		} else try {
			grid = spec.grid(Table_proxy {});
		} catch (const std::invalid_argument &) {
			/* ignore; no grid */
		}
	}

	affine1<double> objective_scaler(const char *out_bounds) const
	{
		if (out_bounds) {
			/* assumes response output has the same label in spec as
			 * in out_bounds (e.g., "delta") */
			assert(output_dim(spec) == 1);
			try {
				return spec.out_scaler<true>(
					Table_proxy(File(out_bounds, "r"), 1)
				).f[0];
			} catch (const std::invalid_argument &c) {
				std::stringstream ss;
				ss << out_bounds << ": " << c.what()
				   << " while scaling objective values";
				throw std::invalid_argument(ss.str());
			}
		} else
			return affine1<double> { .a = 1.0, .b = 0.0 };
	}

	class Table_proxy : public Table {

		struct const_val_proxy {

			const float &v;

			template <typename T>
			std::enable_if_t<std::is_same_v<T,float>,float>
			get() const { return v; }

			template <typename T>
			std::enable_if_t<std::is_same_v<T,double>,double>
			get() const { return v; }

			template <typename T>
			std::enable_if_t<std::is_same_v<T,Double>,Double>
			get() const { return { v }; }

			template <typename T>
			std::enable_if_t<std::is_same_v<T,Q>,Q>
			get() const { return v; }
		};

		struct const_col_proxy {
			const Table *t;
			size_t col;

			const_val_proxy operator[](const char *s) const
			{
				size_t row = t->n_rows;;
				if (!strcmp(s, "min"))
					row = 0;
				else if (!strcmp(s, "max"))
					row = 1;
				if (row >= t->n_rows)
					throw std::invalid_argument(s);
				return { smlp_table_data_row(t, row)[col] };
			}
		};

	public:
		using Table::Table;

		auto operator[](const std::string &s) const
		{
			return (*this)[s.c_str()];
		}

		const_col_proxy operator[](const char *label) const
		{
			ssize_t r = smlp_table_col_idx(this, label);
			if (r < 0)
				throw std::invalid_argument(label);
			return { this, (size_t)r };
		}
	};
};

struct clamp {
	template <typename T> using domain_type = T;
	template <typename T> T operator()(const T &x) const
	{
		return min(max(x, 0), 1);
	}
	friend ostream & operator<<(ostream &os, const clamp &)
	{ return os << "clamp"; }
};

/* CPS is a reasonably concise way of static dispatch letting the compiler infer
 * the appropriate types. We use it here to create optimized code-paths for the
 * different versions of the complete function (e.g., with or without clamping,
 * etc.) and also by unrolling NNs up to unroll_max layers.
 * Deeper NNs will take the slower evaluation path via the finite_composition
 * class. We don't dispatch based on existance of the i/o-scalers as they are
 * point-wise affine functions which can represent the identity keeping fast
 * evaluation without changing the type. Otherwise the number of combinations
 * would increase by a factor of at least 3.
 * Code growth is limited by the compiler's inlining thresholds.
 */
template <size_t unroll_max = 0, typename Run>
static auto with(const model_fun2 &m, Run &&run0,
                 bool clamp_inputs = false, const char *out_bounds = nullptr)
{
	auto run1 = [&](auto f){
		return std::forward<Run>(run0)(composition(m.in_scaler, f));
	};
	auto run2 = [&](auto f){
		if (clamp_inputs)
			return run1(composition(pointwise_single<clamp>(), f));
		else
			return run1(f);
	};
	auto run3 = [&](auto f){
		if (out_bounds)
			return run2(composition(f, m.objective_scaler(out_bounds)));
		else
			return run2(f);
	};
	auto run4 = [&](auto model){
		return run3(composition(model, m.out_scaler, m.objective));
	};
	return m.model.try_decompose_upto<unroll_max>(run4);
}

}

#endif
