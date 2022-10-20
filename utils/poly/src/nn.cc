
#include <nn.hh>

#define SAFE_UNROLL_MAX 0
#include <nn-common.hh>

using namespace smlp;
using namespace iv::functions;

using scaler = affine1<double,double>;
using pt_scaler = pointwise<scaler>;

static sptr<term2>
apply_scaler(const scaler &sc, const sptr<term2> &in, bool clamp_outputs)
{
	sptr<term2> c = make2t(bop2 { bop::ADD,
		make2t(bop2 { bop::MUL, make2t(cnst2 { sc.a }), in }),
		make2t(cnst2 { sc.b })
	});
	if (clamp_outputs) {
		sptr<term2> zero = make2t(cnst2 { kay::Z(0) });
		sptr<term2> one  = make2t(cnst2 { kay::Z(1) });
		c = make2t(ite2 { make2f(prop2 { LT, c, zero }), zero, c });
		c = make2t(ite2 { make2f(prop2 { GT, c, one }), one, c });
	}
	return c;
}

static vec<sptr<term2>>
apply_scaler(const pt_scaler &sc,
             const vec<sptr<term2>> &in, bool clamp_outputs)
{
	size_t n = size(in);
	assert(n == size(sc.f));
	vec<sptr<term2>> in_scaled;
	in_scaled.reserve(n);
	for (size_t i=0; i<n; i++)
		in_scaled.emplace_back(apply_scaler(sc.f[i], in[i], clamp_outputs));
	return in_scaled;
}

static sptr<term2> abs(const sptr<term2> &e)
{
	return make2t(ite2 {
		make2f(prop2 { LT, e, make2t(cnst2 { kay::Q(0) }) }),
		make2t(uop2 { uop::USUB, e }),
		e
	});
}

pre_problem smlp::parse_nn(const char *gen_path, const char *hdf5_path,
                           const char *spec_path, const char *io_bounds,
                           const char *out_bounds, bool clamp_inputs,
                           bool single_obj)
{
	iv::nn::common::model_fun2 mf2(gen_path, hdf5_path, spec_path, io_bounds);

	kjson::json io_bnds = iv::nn::common::json_parse(io_bounds);
	vec<sptr<term2>> in_vars;
	hmap<str,ival> in_bnds;
	vec<sptr<form2>> eta; /* conjunction, for candidates */

	domain dom;
	hmap<str,size_t> name2spec;
	for (size_t i=0; i<input_dim(mf2.spec); i++) {
		kjson::json s = mf2.spec.spec[mf2.spec.dom2spec[i]];
		str id = s["label"].template get<str>();
		name2spec[id] = mf2.spec.dom2spec[i];

		kjson::json bnds = io_bnds[id];
		kay::Q lo = bnds["min"].template get<kay::Q>();
		kay::Q hi = bnds["max"].template get<kay::Q>();
		component c;
		if (s["range"] == "int")
			c.type = component::INT;
		else {
			assert(s["range"] == "float");
			c.type = component::REAL;
		}
		dom.emplace_back(id, move(c));
		in_vars.emplace_back(make2t(name { id }));
		in_bnds.emplace(move(id), ival { lo, hi });
		if (s.contains("safe")) {
			vec<sptr<form2>> safe;
			for (const kjson::json &v : s["safe"])
				safe.emplace_back(make2f(prop2 {
					EQ,
					in_vars.back(),
					make2t(cnst2 { v.template get<kay::Q>() }),
				}));
			eta.emplace_back(make2f(lbop2 { lbop2::OR, move(safe) }));
		}
	}
	// dump_smt2(stdout, dom);
	// dump_smt2(stdout, lbop2 { lbop2::AND, move(in_bnds) });

	const opt_fun<pt_scaler> &in_scaler_opt = mf2.in_scaler;
	assert(in_scaler_opt);
	vec<sptr<term2>> in_scaled = apply_scaler(*in_scaler_opt, in_vars, clamp_inputs);

	sptr<term2> zero = make2t(cnst2 { kay::Q(0) });
	sptr<term2> one = make2t(cnst2 { kay::Q(1) });

	/* sequential_dense is
	 *   finite_composition<dense_layer<keras::activation>>
	 *
	 * dense_layer<..> is
	 *   composition<affine_matrix<float>,keras::activation>
	 *
	 * keras::activation is
	 *   var_fun<keras::linear,keras::relu>
	 */
	iv::nn::common::sequential_dense &m = mf2.model;
	size_t n_layers = size(m.funs);
	size_t layer = 0;
	vec<sptr<term2>> out = in_scaled;
	for (const auto &f : m.funs) {
		const iv::nn::common::affine_matrix<float> &am = std::get<0>(f.t);
		const auto &kernel = am.a; /* matrix<float> */
		const vec<float> &bias = am.b;
		fprintf(stderr, "layer %zu: w: %zu, h: %zu, bias: %zu\n",
		        layer, width(kernel), height(kernel), size(bias));
		/* matrix-vector product */
		assert(width(kernel) == size(out));
		assert(height(kernel) == size(bias));
		std::vector<sptr<term2>> next;
		for (float b : bias)
			next.push_back(make2t(cnst2 { kay::Q(b) }));
		for (size_t y=0; y<height(kernel); y++)
			for (size_t x=0; x<width(kernel); x++)
				next[y] = make2t(bop2 { bop::ADD,
					next[y],
					make2t(bop2 { bop::MUL,
						make2t(cnst2 { kay::Q(kernel(x,y)) }),
						out[x],
					})
				});
		match(std::get<1>(f.t).f.f,
		[&](const iv::nn::common::keras::relu &) {
			for (sptr<term2> &e : next)
				e = make2t(ite2 { make2f(prop2 { LE, e, zero }), zero, e });
		},
		[](const iv::nn::common::keras::linear &) {});
		layer++;
		out = move(next);
	}

	assert(size(out) == output_dim(mf2.spec));
	if (mf2.out_scaler)
		out = apply_scaler(*mf2.out_scaler, out, false);

	if (mf2.objective.type != smlp_response::SMLP_RESPONSE_ID) {
		std::stringstream ss;
		ss << mf2.objective;
		DIE(1,"unsupported objective function type, only identity is: "
		      "%s\n", ss.str().c_str());
	}

	vec<str> out_names;
	for (kjson::json comp : mf2.spec.spec)
		if (comp["type"] == "response")
			out_names.emplace_back(comp["label"].get<std::string_view>());

	assert(size(out_names) == size(out));

	hmap<str,sptr<term2>> outs;
	for (size_t i=0; i<size(out); i++)
		outs[out_names[i]] = out[i];

	sptr<term2> obj;
	if (single_obj) {
		/* apply mf2.objective: select right output(s) */
		fprintf(stderr, "obj response idx: %zd\n", mf2.objective.idx[0]);
		ssize_t idx = mf2.objective.idx[0];
		assert(idx >= 0);
		assert(idx < size(out));
		obj = out[idx];

		if (out_bounds)
			obj = apply_scaler(mf2.objective_scaler(out_bounds),
			                   obj, false);
	} else {
		/* Pareto */
		assert(size(out) == 1); /* not implemented, yet */
		obj = out[0];
		assert(!out_bounds); /* not implemented, yet */
	}

	/* construct theta */
	auto theta = [spec=move(mf2.spec.spec), name2spec=move(name2spec)]
	             (bool left, const hmap<str,sptr<term2>> &v) {
		vec<sptr<form2>> conj;
		for (const auto &[n,e] : v) {
			auto it = name2spec.find(n);
			assert(it != name2spec.end());
			kjson::json sp = spec[it->second];
			sptr<term2> nm = make2t(name { n });
			sptr<term2> r;
			if (sp.contains("rad-abs")) {
				r = make2t(cnst2 { sp["rad-abs"].template get<kay::Q>() });
			} else if (sp.contains("rad-rel")) {
				r = make2t(cnst2 { sp["rad-rel"].template get<kay::Q>() });
				r = make2t(bop2 { bop::MUL, move(r), abs(left ? e : nm) });
			} else if (sp["type"] == "input")
				continue;
			else
				DIE(1,"error: .spec contains neither 'rad-abs' "
				      "nor 'rad-rel' for '%s'\n", n.c_str());
			conj.emplace_back(make2f(prop2 { LE,
				abs(make2t(bop2 { bop::SUB, nm, e })),
				move(r)
			}));
		}
		return make2f(lbop2 { lbop2::AND, move(conj) });
	};

	return pre_problem {
		move(dom),
		move(obj),
		move(outs),
		move(in_bnds),
		make2f(lbop2 { lbop2::AND, move(eta) }),
		true2,
		move(theta),
	};
}
