
#ifndef IV_ANON_NN_COMMON
# define IV_ANON_NN_COMMON 1
#endif

#include <iv/nn-common.hh>

#if !(IV_ANON_FUNCTIONS-0)
using namespace IV_NS_FUNCTIONS;
#endif
#if !(IV_ANON_NN_MODEL-0)
using namespace IV_NS_NN_MODEL;
#endif
#if !(IV_ANON_NN_COMMON-0)
using namespace IV_NS_NN_COMMON;
#endif

namespace IV_NS_NN_MODEL {
sequential_dense model_from_keras_hdf5(const char *path)
{
	using H5::H5File;
	using H5::Attribute;
	using H5::Group;
	using H5::DataSet;
	using H5::IdComponent;

	using kjson::json;

	try {
		H5File f(path, H5F_ACC_RDONLY);
		Group root = f.openGroup("/");
		Attribute keras_version = root.openAttribute("keras_version");
		H5std_string str;
		keras_version.read(keras_version.getDataType(), str);
		std::cerr << "Keras v" << str << "\n";
		Group weights = root.openGroup("model_weights");
		Attribute layer_names = weights.openAttribute("layer_names");
		layer_names.getSpace().selectAll();
		std::cerr << "atype dims: " << layer_names.getStorageSize() << "\n";
		size_t layer_names_sz1 = layer_names.getStrType().getSize();
		std::cerr << "xx: " << layer_names_sz1 << "\n";
		size_t layer_names_sz = layer_names.getInMemDataSize();
		std::vector<char> arr(layer_names_sz);
		layer_names.read(layer_names.getStrType(), arr.data());
		std::cerr << "layer names: " << arr << "\n" << "size: " << arr.size() << "\n";
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
		for (size_t i=0; i<layer_names_sz; i+=layer_names_sz1) {
			std::string name(arr.data() + i, layer_names_sz1);
			Group layer = weights.openGroup(name);
			Attribute names = layer.openAttribute("weight_names");
			std::vector<char> names_s(names.getInMemDataSize());
			names.read(names.getStrType(), names_s.data());
			std::string sa(names_s.data(), names.getStrType().getSize());
			DataSet a = layer.openDataSet(sa.c_str());
			DataSet b = layer.openDataSet(names_s.data() + names.getStrType().getSize());
			assert(a.getSpace().getSimpleExtentNdims() == 2);
			assert(b.getSpace().getSimpleExtentNdims() == 1);
			hsize_t x[2], y[1];
			a.getSpace().getSimpleExtentDims(x);
			b.getSpace().getSimpleExtentDims(y);
			std::cerr << "kernel: " << x[0] << ", " << x[1] << "\n";
			std::cerr << "bias  : " << y[0] << "\n";
			auto lc = config_layers[i/layer_names_sz1];
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
		return model;
	} catch (const H5::Exception &f) {
		std::stringstream ss;
		ss << path << ": " << f.getDetailMsg();
		throw std::runtime_error(ss.str());
	}
}
}


#if 0
finite_union<ival> specification::region1(size_t i, const Q &center) const
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
			      "'range' == 'int' is not an interger\n";
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

region specification::region_from_center(const finite_product<ival> &row) const
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

region specification::region_from_center_csv(char *s) const
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


static ssize_t resp_idx(const json &gen, const std::string_view &label)
{
	json resp = gen["response"];
	auto it = find(begin(resp), end(resp), label);
	if (it == end(resp))
		return -1;
	return it - begin(resp);
}


Response::Response(json &gen)
: response RESPONSE_INIT
{
	std::string o = gen["objective"].get<std::string>();
	int r = response_parse(this, o.c_str(),
	                       [](const char *label, void *udata) -> ssize_t {
		return resp_idx(*static_cast<json *>(udata), label);
	                       }, &gen);
	if (r) {
		std::stringstream ss;
		ss << "error " << r << " parsing objective function '"
		   << o << "'";
		throw std::invalid_argument(ss.str());
	}
}


namespace {

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
			return { table_data_row(t, row)[col] };
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
		ssize_t r = table_col_idx(this, label);
		if (r < 0)
			throw std::invalid_argument(label);
		return { this, (size_t)r };
	}
};
}


model_fun2::model_fun2(const char *gen_path, const char *keras_hdf5_path,
                       const char *spec_path, const char *io_bounds)
: model_fun2(json_parse(gen_path), keras_hdf5_path, spec_path,
             io_bounds)
{}

model_fun2::model_fun2(json gen, const char *keras_hdf5_path,
                       const char *spec_path, const char *io_bounds)
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

affine1<double> model_fun2::objective_scaler(const char *out_bounds) const
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
#endif
