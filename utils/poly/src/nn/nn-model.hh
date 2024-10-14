/*
 * nn-model.hh
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 *
 * See the LICENSE file for terms of distribution.
 */

#ifndef IV_NN_MODEL_HH
#define IV_NN_MODEL_HH

#ifndef IV_DEBUG
# define IV_DEBUG(x)
#endif

#if (IV_ANON_NN_MODEL-0)
# define IV_NS_NN_MODEL
#else
# define IV_NS_NN_MODEL iv::nn::model
#endif

#include "functions.hh"

#include <kay/bits.hh>

#include <string_view>
#include <sstream>
#include <iostream>

#include <H5Cpp.h>

#include <kjson.hh>

namespace IV_NS_NN_MODEL {

#if !(IV_ANON_FUNCTIONS-0)
using namespace IV_NS_FUNCTIONS;
#endif

using namespace std::literals::string_view_literals;

template <typename T, size_t n, typename Store = std::vector<T>> struct contiguous;

template <typename T, typename Store>
struct contiguous<T,2,Store> : protected Store {

	contiguous(size_t n0, size_t n1)
	: Store(n0 * n1)
	, n0(n0)
	{}

	template <size_t d>
	std::enable_if_t<(d < 2),size_t> dim() const
	{
		if constexpr (d == 0)
			return n0;
		else
			return size(static_cast<const Store &>(*this)) / dim<0>();
	}

	const T & operator()(size_t i0, size_t i1) const
	{
		return (*this)[i0+i1*n0];
	}

	      T & operator()(size_t i0, size_t i1)
	{
		return (*this)[i0+i1*n0];
	}

private:
	size_t n0;
};

template <typename T, bool col_first, typename Store = std::vector<T>>
struct matrix : protected contiguous<T,2,Store> {

	using contiguous<T,2,Store>::data;

protected:
	struct steal {};

	explicit matrix(steal, contiguous<T,2,Store> &&o)
	: contiguous<T,2,Store>(move(o))
	{}

public:
	matrix(size_t width, size_t height)
	: contiguous<T,2,Store>(col_first ? height : width,
	                        col_first ? width : height)
	{}

	/* explicit since it is slow */
	template <typename S, typename St,
	          typename = std::enable_if_t<std::is_assignable_v<T &,const S &>>
	         >
	explicit matrix(const matrix<S,!col_first,St> &o)
	: matrix(width(o), height(o))
	{
		for (size_t y=0; y<height(o); y++)
			for (size_t x=0; x<width(o); x++)
				(*this)(x,y) = o(x,y);
	}

	friend size_t width(const matrix &v) noexcept
	{
		if constexpr (col_first)
			return v.template dim<1>();
		else
			return v.template dim<0>();
	}

	friend size_t height(const matrix &v) noexcept
	{
		if constexpr (col_first)
			return v.template dim<0>();
		else
			return v.template dim<1>();
	}

	const T & operator()(size_t x, size_t y) const
	{
		if constexpr (col_first)
			return contiguous<T,2,Store>::operator()(y, x);
		else
			return contiguous<T,2,Store>::operator()(x, y);
	}

	      T & operator()(size_t x, size_t y)
	{
		if constexpr (col_first)
			return contiguous<T,2,Store>::operator()(y, x);
		else
			return contiguous<T,2,Store>::operator()(x, y);
	}

	friend matrix<T,!col_first,Store> transpose(matrix &&c)
	{
		return {
			steal {},
			move(static_cast<contiguous<T,2,Store> &&>(c)),
		};
	}

	friend std::ostream & operator<<(std::ostream &os, const matrix &c)
	{
		os << "matrix " << height(c) << " rows " << width(c) << " cols:\n";
		for (size_t i=0; i<height(c); i++) {
			for (size_t j=0; j<width(c); j++)
				os << (j ? ", " : "") << c(j,i);
			os << "\n";
		}
		return os;
	}
};

template <typename S>
struct affine_matrix {

	template <typename T> using domain_type = std::vector<T>;

	template <bool cfk>
	affine_matrix(matrix<S,cfk> kernel, std::vector<S> bias)
	: a(move(kernel))
	, b(move(bias))
	{
		assert(height(a) == size(b));
	}

	friend dim input_dim(const affine_matrix &m)
	{
		return (dim)width(m.a);
	}

	friend dim output_dim(const affine_matrix &m)
	{
		return (dim)height(m.a);
	}

	template <typename T>
	std::vector<T> operator()(const std::vector<T> &x) const
	{
		assert(width(a) == size(x));
		std::vector<T> r(begin(b), end(b));
		for (size_t y=0; y<height(a); y++)
			for (size_t i=0; i<size(x); i++)
				r[y] += a(i,y) * x[i];
		return r;
	}

	std::ostream & print_fun(std::ostream &os) const
	{
		return os << "affine_matrix["
		          << height(a) << "×" << width(a) << "]";
	}

	friend std::ostream & operator<<(std::ostream &os, const affine_matrix &m)
	{
		return m.print_fun(os);
	}

	/* col_first=true is just an optimization to avoid expensive copy in
	 * context of model_from_keras_hdf5() */
	matrix<S,true> a;
	std::vector<S> b;
};

template <bool norm, typename S, typename T = S>
affine1<T> minmax_scaler(S min, S max)
{
	assert(min < max);
	if constexpr (norm)
		return inverse(minmax_scaler<false,S,T>(min, max));
	else
		return { T(max - min), T(min) };
}

template <typename Act, typename... LayerTypes>
using layer = composition<var_fun_t<LayerTypes...>,pointwise_single<Act>>;

template <typename Act>
using dense_layer = layer<Act,affine_matrix<float>>; // composition<affine_matrix<float>,pointwise_single<Act>>;


namespace {
[[maybe_unused]] double max(double a, int b) { using std::max; return max(a, (double)b); /*return fmax(a, b);*/ }
[[maybe_unused]] double min(double a, int b) { using std::min; return min(a, (double)b); /*return fmin(a ,b);*/ }
}


namespace keras {

struct relu {
	static constexpr char keras_label[] = "relu";
	template <typename S> using domain_type = S;
	template <typename S> auto operator()(S &&x) const
	{ return max(std::forward<S>(x), 0); }
	friend std::ostream & operator<<(std::ostream &os, const relu &)
	{ return os << "relu"; }
};

struct linear : identity {
	static constexpr char keras_label[] = "linear";
	friend std::ostream & operator<<(std::ostream &os, const linear &)
	{ return os << "linear"; }
};

struct tanh_impl {
	static constexpr char keras_label[] = "tanh";
	template <typename S> using domain_type = S;
	template <typename S> auto operator()(S &&x) const
	{ return tanh(std::forward<S>(x)); }
	friend std::ostream & operator<<(std::ostream &os, const tanh_impl &)
	{ return os << "tanh"; }
};

using tanh = tanh_impl;

}
} // end ns IV_NS_NN_MODEL

namespace IV_NS_FUNCTIONS {
template <> struct is_identity<IV_NS_NN_MODEL::keras::linear> : std::true_type {};
}

namespace IV_NS_NN_MODEL {
namespace keras {

template <size_t i, typename... Ts>
static var_fun<Ts...> make_activation_(const std::string_view &label)
{
	if constexpr (i >= sizeof...(Ts)) {
		std::stringstream ss;
		ss << "keras_activation '" << label << "' unknown";
		throw std::invalid_argument(ss.str());
	} else {
		using T = std::tuple_element_t<i,std::tuple<Ts...>>;
		if (T::keras_label == label)
			return T {};
		return make_activation_<i+1,Ts...>(label);
	}
}

template <typename... Ts>
static auto make_activation0(const std::string_view &label)
{
	return make_activation_<0,Ts...>(label);
}

static auto make_activation(const std::string_view &label)
{
	return make_activation0<linear,relu>(label);
}

typedef std::invoke_result_t<decltype(keras::make_activation)
                            ,const std::string_view &
                            > activation;

} // end ns keras

typedef finite_composition<dense_layer<keras::activation>> sequential_dense;

sequential_dense model_from_keras_hdf5(const char *path)
{
	using H5::H5File;
	using H5::Attribute;
	using H5::Group;
	using H5::DataSet;
	using H5::IdComponent;

	using kjson::json;

//	try {
		H5File f(path, H5F_ACC_RDONLY);
		Group root = f.openGroup("/");
		Attribute keras_version = root.openAttribute("keras_version");
		H5std_string str;
		keras_version.read(keras_version.getDataType(), str);
		std::cerr << "Keras v" << str << "\n";
		Group weights = root.openGroup("model_weights");
		Attribute layer_names = weights.openAttribute("layer_names");
		layer_names.getSpace().selectAll();
		auto read_str_data_space = [str](Attribute &a) {
			std::vector<std::string> r;
			if (str == "2.3.0-tf") {
				std::cerr << "atype dims: " << a.getStorageSize() << "\n";
				size_t layer_names_sz1 = a.getStrType().getSize();
				std::cerr << "xx: " << layer_names_sz1 << "\n";
				size_t layer_names_sz = a.getInMemDataSize();
				std::vector<char> arr(layer_names_sz);
				a.read(a.getStrType(), arr.data());
				std::cerr << "layer names arr size: " << arr.size() << "\n";
				for (size_t i=0; i<layer_names_sz; i+=layer_names_sz1)
					r.emplace_back(arr.data() + i, layer_names_sz1);
			} else if (str == "2.9.0") {
				size_t n_layers = a.getSpace().getSimpleExtentNpoints();
				std::vector<char *> test(n_layers);
				a.read(a.getStrType(), test.data());
				for (size_t i=0; i<n_layers; i++)
					r.emplace_back(test[i]);
			} else
				assert(0); /* unknown keras version */
			return r;
		};
		std::vector<std::string> layers = read_str_data_space(layer_names);
		std::cerr << "layer names";
		for (size_t i=0; i<layers.size(); i++)
			std::cerr << (i ? ',' : ':') << ' ' << layers[i];
		std::cerr << "\n";
		// sequential_model model;
		sequential_dense model;
		Attribute model_config = root.openAttribute("model_config");
		H5std_string model_config_s;
		model_config.read(model_config.getDataType(), model_config_s);
		std::cerr << "model_config: " << model_config_s << "\n";
		json config = json::parse(std::move(model_config_s));
		assert(config["class_name"] == "Sequential");
		// std::cout << "model_config: " << config << "\n";
		json config_layers = config["config"]["layers"];
		// std::cout << "layers config: " << config_layers << "\n";
		for (size_t i=0; i<size(layers); i++) {
			std::string &name = layers[i];
			Group layer = weights.openGroup(name);
			Attribute names = layer.openAttribute("weight_names");
			DataSet a, b;
			if (str == "2.3.0-tf") {
				std::vector<char> names_s(names.getInMemDataSize());
				names.read(names.getStrType(), names_s.data());
				std::string sa(names_s.data(), names.getStrType().getSize());
				a = layer.openDataSet(sa.c_str());
				b = layer.openDataSet(names_s.data() + names.getStrType().getSize());
			} else if (str == "2.9.0") {
				std::vector<std::string> wnames = read_str_data_space(names);
				assert(size(wnames) == 2);
				a = layer.openDataSet(wnames[0]);
				b = layer.openDataSet(wnames[1]);
			}
			assert(a.getSpace().getSimpleExtentNdims() == 2);
			assert(b.getSpace().getSimpleExtentNdims() == 1);
			hsize_t x[2], y[1];
			a.getSpace().getSimpleExtentDims(x);
			b.getSpace().getSimpleExtentDims(y);
			std::cerr << "kernel: " << x[0] << ", " << x[1] << "\n";
			std::cerr << "bias  : " << y[0] << "\n";
			auto lc = config_layers[str == "2.3.0-tf" ? i : i+1];
			assert(lc["class_name"] == "Dense");
			assert(lc["config"]["name"] == name.c_str());
			auto act = lc["config"]["activation"];
			std::cerr << "act   : " << act << "\n";
			assert(y[0] == x[1]);
			matrix<float,true> kernel(x[0], x[1]);
			std::vector<float> bias(y[0]);
			a.read(kernel.data(), a.getFloatType());
			b.read(bias.data(), b.getFloatType());
			IV_DEBUG(std::cerr << "kernel: " << kernel << "\n");
			IV_DEBUG(std::cerr << "bias  : " << bias << "\n");
			model.append({
				affine_matrix<float>(move(kernel), move(bias)),
				keras::make_activation(act.get<std::string_view>())
			});
		}
		return model;/*
	} catch (const H5::Exception &f) {
		std::stringstream ss;
		ss << path << ": " << f.getDetailMsg();
		throw std::runtime_error(ss.str());
	}*/
}

}

#endif
