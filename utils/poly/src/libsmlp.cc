
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "expr2.hh"
#include "infix.hh"
#include "nn.hh"
#include "poly.hh"
#include "solver.hh"

using namespace smlp;
using namespace reals::eager;

template <decltype(bop2::op) op>
static sptr<term2> mk_bop2(sptr<term2> a, sptr<term2> b)
{
	return make2t(bop2 { op, move(a), move(b) });
}

template <decltype(uop2::op) op>
static sptr<term2> mk_uop2(sptr<term2> a)
{
	return make2t(uop2 { op, move(a) });
}

static sptr<term2> mk_ite(sptr<form2> c, sptr<term2> y, sptr<term2> n)
{
	return make2t(ite2 { move(c), move(y), move(n) });
}

template <decltype(lbop2::op) op>
static sptr<form2> mk_lbop2_bin(sptr<form2> a, sptr<form2> b)
{
	return make2f(lbop2 { op, { move(a), move(b) } });
}

template <cmp_t cmp>
static sptr<form2> mk_prop(sptr<term2> a, sptr<term2> b)
{
	return make2f(prop2 { cmp, move(a), move(b) });
}

template <decltype(lbop2::op) op>
static sptr<form2> mk_lbop2(boost::python::object o)
{
	using namespace boost::python;
	stl_input_iterator<sptr<form2>> begin(o), end;
	return make2f(lbop2 { op, vec<sptr<form2>>(begin, end) });
}

static sptr<term2> mk_name(str s)
{
	return make2t(name { move(s) });
}

static bool parse_Z(const str &o, kay::Z &z)
{
	const char *end = o.data() + o.length();
	auto r = kay::from_chars(o.data(), end, z);
	return r.ec == std::errc {} && r.ptr == end;
}

static sptr<term2> mk_cnst(kay::Q v)
{
	return make2t(cnst2 { move(v) });
}

static boost::python::object dt_cnst_term(sptr<term2> t)
{
	using boost::python::object;
	if (const cnst2 *c = t->get<cnst2>())
		return c->value.match(
		[](const kay::Z &z) { return object(kay::Q(z)); },
		[](const kay::Q &q) { return object(q); },
		[](const A &a) { return object(a); }
		);
	return {};
}

static boost::python::object dt_cnst_form(sptr<form2> t)
{
	using boost::python::object;
	if (const lbop2 *c = t->get<lbop2>(); c && c->args.empty())
		return object(c->op == lbop2::AND);
	return {};
}

static kay::Q _mk_Q_F(double d) { return kay::Q(d); }
static boost::python::object _mk_Q_Z(str s)
{
	kay::Z z;
	if (parse_Z(s, z))
		return boost::python::object(kay::Q(move(z)));
	return {};
}

static boost::python::object _mk_Q_Q(str n, str d)
{
	kay::Z u, v;
	if (parse_Z(n, u) && parse_Z(d, v) && v)
		return boost::python::object(kay::Q(move(u), move(v)));
	return {};
}

static str term2_str(const sptr<term2> &t) { return to_string(t, true); }
static str form2_str(const sptr<form2> &f) { return to_string(f, true); }

static hmap<str,sptr<term2>> convert_term_dict(boost::python::dict d)
{
	using boost::python::extract;
	boost::python::list l = d.items();
	hmap<str,sptr<term2>> r;
	for (ssize_t i=0; i<len(l); i++) {
		boost::python::object o = l[i];
		r[extract<str>(o[0])] = extract<sptr<term2>>(o[1]);
	}
	return r;
}

static domain convert_domain_dict(boost::python::dict d)
{
	domain r;
	using boost::python::extract;
	using boost::python::object;
	boost::python::list l = d.items();
	for (ssize_t i=0; i<len(l); i++) {
		object o = l[i];
		r.emplace_back(extract<str>(o[0]), extract<component>(o[1]));
	}
	return r;
}

namespace {
struct domain_to_dict {
	static PyObject * convert(const domain &d)
	{
		boost::python::dict r;
		for (const auto &[n,c] : d)
			r[n] = c;
		PyObject *p = r.ptr();
		Py_INCREF(p);
		return p;
	}
};
struct extract_domain {
	static domain execute(boost::python::dict d)
	{
		return convert_domain_dict(d);
	}
};
}

template <typename T>
static sptr<T> _cnst_fold(const sptr<T> &t, boost::python::dict d)
{
	return cnst_fold(t, convert_term_dict(d));
}

template <typename T>
static sptr<T> _subst(const sptr<T> &t, boost::python::dict d)
{
	return subst(t, convert_term_dict(d));
}

static boost::python::list set_to_list(const hset<str> &s)
{
	boost::python::list r;
	for (const str &v : s)
		r.append(boost::python::str(v));
	return r;
}

template <typename T>
static boost::python::list _free_vars(const sptr<T> &p)
{
	return set_to_list(free_vars(p));
}

static boost::python::dict options(boost::python::object o)
{
	if (!o.is_none()) {
		using boost::python::extract;
		boost::python::dict p = extract<boost::python::dict>(o);
		boost::python::list l = p.items();
		for (ssize_t i=0; i<len(l); i++) {
			boost::python::object o = l[i];
			str k = extract<str>(o[0]);
			boost::python::object v = o[1];
			if (k == "ext_solver_cmd") {
				if (v.is_none())
					ext_solver_cmd.reset();
				else
					ext_solver_cmd = extract<str>(v);
			} else if (k == "inc_solver_cmd") {
				if (v.is_none())
					inc_solver_cmd.reset();
				else
					inc_solver_cmd = extract<str>(v);
			} else if (k == "intervals")
				intervals = extract<decltype(intervals)>(v);
			else if (k == "log_color")
				switch (extract<int>(v)) {
				case 0: Module::log_color = false; break;
				case 1: Module::log_color = true; break;
				default:
					throw std::invalid_argument("log_color setting to options()");
				}
			else if (k == "alg_dec_prec_approx")
				solver::alg_dec_prec_approx = extract<decltype(solver::alg_dec_prec_approx)>(v);
			else
				throw std::invalid_argument("key '" + k + "' not known for options()");
		}
	}
	boost::python::dict r;
	if (ext_solver_cmd)
		r[boost::python::str("ext_solver_cmd")] = boost::python::str(*ext_solver_cmd);
	if (inc_solver_cmd)
		r[boost::python::str("inc_solver_cmd")] = boost::python::str(*inc_solver_cmd);
	r[boost::python::str("intervals")] = boost::python::long_(intervals);
	r[boost::python::str("log_color")] = boost::python::long_(Module::log_color ? 1 : 0);
	r[boost::python::str("alg_dec_prec_approx")] = boost::python::long_(solver::alg_dec_prec_approx);
	return r;
}

static void solver_declare(const sptr<solver> &s, const domain &dom)
{
	return s->declare(dom);
}

static void solver_declare_dict(const sptr<solver> &s, const boost::python::dict &dom)
{
	return s->declare(convert_domain_dict(dom));
}

static void solver_add(const sptr<solver> &s, const sptr<form2> &f)
{
	return s->add(f);
}

static auto solver_check(const sptr<solver> &s)
{
	using boost::python::object;
	return s->check().match([](const auto &s) { return object(s); });
}

static sptr<solver> _mk_solver(bool incremental, const char *logic)
{
	return sptr<solver>(mk_solver(incremental, logic).release());
}

static str Z_str(const kay::Z &z)
{
	return kay::to_string(z);
}

static str Q_str(const kay::Q &q)
{
	return kay::to_string(q);
}

static component mk_component_entire(type ty)
{
	return component { entire {}, ty };
}

static component mk_component_ival(type ty, kay::Q lo, kay::Q hi)
{
	if (lo > hi)
		throw std::invalid_argument("constructing interval: lower endpoint must be >= upper endpoint");
	return component { ival { move(lo), move(hi) }, ty };
}

static component mk_component_list(type ty, boost::python::object l)
{
	boost::python::stl_input_iterator<kay::Q> begin(l), end;
	return component { smlp::list { vec<kay::Q>(begin, end) }, ty };
}

static boost::python::dict sat_get_model(const sat &s)
{
	boost::python::dict r;
	for (const auto &[n,t] : s.model)
		r[n] = t;
	return r;
}

static boost::python::long_ _Z_to_long(const kay::Z &v)
{
	return boost::python::long_(kay::to_string(v));
}

static boost::python::long_ _Q_num(const kay::Q &q) { return _Z_to_long(q.get_num()); }
static boost::python::long_ _Q_den(const kay::Q &q) { return _Z_to_long(q.get_den()); }

template <cmp_t c>
static bool _Q_cmp(const kay::Q &a, const kay::Q &b)
{
	return do_cmp(a, c, b);
}

template <decltype(bop2::op) op>
static kay::Q _Q_bin(const kay::Q &a, const kay::Q &b)
{
	switch (op) {
	case bop2::ADD: return a + b;
	case bop2::SUB: return a - b;
	case bop2::MUL: return a * b;
	}
}

static kay::Q _Q_div(const kay::Q &a, const kay::Q &b)
{
	if (!b)
		throw std::runtime_error("division by zero on type libsmlp.Q");
	return a / b;
}

template <decltype(uop2::op) op>
static kay::Q _Q_un(const kay::Q &a)
{
	switch (op) {
	case uop2::UADD: return +a;
	case uop2::USUB: return -a;
	}
}

static kay::Q _Q_abs(const kay::Q &a)
{
	using namespace kay;
	return abs(a);
}

static kay::Z _lbound_log2(const R &r)
{
	return lbound_log2(r);
}

static const char * smlp_version() { return SMLP_VERSION; }

template <cmp_t c, typename T>
static bool do_cmp(const T &l, const T &r)
{
	static_assert(c <= NE, "template parameter c is not a valid cmp_t");
	if constexpr (c == EQ)
		return l == r;
	if constexpr (c == NE)
		return l != r;
	if constexpr (c == LT)
		return l < r;
	if constexpr (c == LE)
		return l <= r;
	if constexpr (c == GT)
		return l > r;
	if constexpr (c == GE)
		return l >= r;
	unreachable();
}

static boost::python::object R_approx(const R &r, const boost::python::long_ &p)
{
	str s = boost::python::extract<str>(boost::python::str(p));
	kay::Z z;
	if (parse_Z(s, z))
		return boost::python::object(approx(r, z));
	return {};
}

BOOST_PYTHON_MODULE(libsmlp)
{
	using namespace boost::python;

	def("_version", smlp_version);

	/* exported expr2.hh API */
	class_<sptr<term2>>("term2", no_init)
		.def("__add__", mk_bop2<bop2::ADD>, args("a","b"), "\
Returns a new term2 instance corresponding to the sum of a and b.\n\
Both parameters must be term2 instances.\n\
This function handles the binary Python 'a+b' expression."
		)
		.def("__sub__", mk_bop2<bop2::SUB>, args("a","b"), "\
Returns a new term2 instance corresponding to the difference of a and b.\n\
Both parameters must be term2 instances.\n\
This function handles the binary Python 'a-b' expression."
		)
		.def("__mul__", mk_bop2<bop2::MUL>, args("a","b"), "\
Returns a new term2 instance corresponding to the product of a and b.\n\
Both parameters must be term2 instances.\n\
This function handles the binary Python 'a*b' expression."
		)
		.def("__pos__", mk_uop2<uop2::UADD>, args("a"), "\
Returns a new term2 instance corresponding to the value of a.\n\
It is thus a no-op even though a new term2 is actually constructed.\n\
The parameter must be a term2 instance.\n\
This function handles the unary Python '+a' expression."
		)
		.def("__neg__", mk_uop2<uop2::USUB>, args("a"), "\
Returns a new term2 instance corresponding to the negative value of a.\n\
The parameter must be a term2 instance.\n\
This function handles the unary Python '-a' expression."
		)
		.def("__abs__", smlp::abs, args("a"), "\
Returns a new term2 instance corresponding to the absolute value of a.\n\
This is represented as (ite (< a 0) (- a) a) in SMT format.\n\
The parameter must be a term2 instance.\n\
This function handles the Python 'abs(a)' expression."
		)
		.def("__lt__", mk_prop<LT>, args("a","b"), "\
Returns a new form2 instance that evaluates to true iff the value of a is\n\
less than that of b.\n\
Both parameters must be term2 instances.\n\
This function handles the binary Python 'a<b' expression."
		)
		.def("__le__", mk_prop<LE>, args("a","b"), "\
Returns a new form2 instance that evaluates to true iff the value of a is\n\
less than or equal to that of b.\n\
Both parameters must be term2 instances.\n\
This function handles the binary Python 'a<=b' expression."
		)
		.def("__gt__", mk_prop<GT>, args("a","b"), "\
Returns a new form2 instance that evaluates to true iff the value of a is\n\
greater than that of b.\n\
Both parameters must be term2 instances.\n\
This function handles the binary Python 'a>b' expression."
		)
		.def("__ge__", mk_prop<GE>, args("a","b"), "\
Returns a new form2 instance that evaluates to true iff the value of a is\n\
greater than or equal to that of b.\n\
Both parameters must be term2 instances.\n\
This function handles the binary Python 'a>=b' expression."
		)
		.def("__eq__", mk_prop<EQ>, args("a","b"), "\
Returns a new form2 instance that evaluates to true iff the value of a is\n\
equal to that of b.\n\
Both parameters must be term2 instances.\n\
This function handles the binary Python 'a==b' expression."
		)
		.def("__ne__", mk_prop<NE>, args("a","b"), "\
Returns a new form2 instance that evaluates to true iff the value of a is\n\
not equal to that of b.\n\
Both parameters must be term2 instances.\n\
This function handles the binary Python 'a!=b' expression."
		)
		.def("__str__", term2_str)
		;
	class_<sptr<form2>>("form2", no_init)
		.def("__and__", mk_lbop2_bin<lbop2::AND>, args("a","b"), "\
Returns a new form2 instance that corresponds to the conjunction of a and b.\n\
Both parameters must be form2 instances.\n\
This function handles the binary Python 'a&b' expression.\n\
Note: The Python keyword 'and' is not defined for form2 expressions, use '&'."
		)
		.def("__or__", mk_lbop2_bin<lbop2::OR>, args("a","b"), "\
Returns a new form2 instance that corresponds to the disjunction of a and b.\n\
Both parameters must be form2 instances.\n\
This function handles the binary Python 'a|b' expression.\n\
Note: The Python keyword 'or' is not defined for form2 expressions, use '|'."
		)
		.def("__invert__", smlp::neg, args("a"), "\
Returns a new form2 instance that corresponds to the negation of a.\n\
The parameter must be a form2 instance.\n\
This function handles the unary Python '~a' expression.\n\
Note: The Python keyword 'not' is not defined for form2 expressions, use '~'."
		)
		.def("__str__", form2_str)
		;

	def("_mk_and", mk_lbop2<lbop2::AND>, args("as : list[form2]"));
	def("_mk_or", mk_lbop2<lbop2::OR>, args("as : list[form2]"));
	def("_mk_cnst", mk_cnst);
	def("_dt_cnst", dt_cnst_term);
	def("_dt_cnst", dt_cnst_form);

	def("Ite", mk_ite);
	def("Var", mk_name);

	def("true" , (sptr<form2>(*)())[]() -> sptr<form2> { return true2; });
	def("false", (sptr<form2>(*)())[]() -> sptr<form2> { return false2; });
	def("zero" , (sptr<term2>(*)())[]() -> sptr<term2> { return zero; });
	def("one"  , (sptr<term2>(*)())[]() -> sptr<term2> { return one; });

	def("_cnst_fold", _cnst_fold<term2>);
	def("_cnst_fold", _cnst_fold<form2>);

	def("simplify", (sptr<term2>(*)(const sptr<term2> &))simplify);
	def("simplify", (sptr<form2>(*)(const sptr<form2> &))simplify);

	def("subst", _subst<term2>);
	def("subst", _subst<form2>);

	def("is_ground", (bool(*)(const sptr<term2> &))is_ground);
	def("is_ground", (bool(*)(const sptr<form2> &))is_ground);

	def("is_nonlinear", (bool(*)(const sptr<term2> &))is_nonlinear);
	def("is_nonlinear", (bool(*)(const sptr<form2> &))is_nonlinear);

	def("_free_vars", _free_vars<term2>);
	def("_free_vars", _free_vars<form2>);

	def("derivative", derivative);

	def("to_nnf", to_nnf);

	/* missing: all_eq, is_linear */

	/* exported nn.hh API */
	class_<pre_problem>("pre_problem")
		.def_readwrite("dom", &pre_problem::dom)
		.def_readwrite("obj", &pre_problem::obj)
		.def_readwrite("funcs", &pre_problem::funcs)
		.def_readwrite("func_bounds", &pre_problem::func_bounds)
		.def_readwrite("input_bounds", &pre_problem::input_bounds)
		.def_readwrite("eta", &pre_problem::eta)
		.def_readwrite("partial_domain", &pre_problem::partial_domain)
		.def_readwrite("theta", &pre_problem::theta)
		.def("interpret_input_bounds", &pre_problem::interpret_input_bounds)
		;
#ifdef SMLP_ENABLE_KERAS_NN
	def("_parse_nn", parse_nn,
	    args("gen_path", "hdf5_path", "spec_path", "io_bounds",
	         "obj_bounds", "clamp_inputs", "single_obj"));
#endif

	/* exported poly.hh API */
	def("_parse_poly", parse_poly_problem,
	    args("simple_domain_path", "poly_expression_path", "python_compat",
	         "dump_pe", "infix"));

	/* exported infix.hh API */
	def("_parse_infix", (expr(*)(std::string, bool))parse_infix,
	    args("str", "python_compat"));

	def("_options", options);
	def("set_loglvl", set_loglvl);

	/* exported domain.hh API */
	class_<component>("component", no_init)
		.def_readonly("range", &component::range)
		.def_readwrite("type", &component::type)
		.def("__eq__", do_cmp<EQ, component>)
		.def("__ne__", do_cmp<NE, component>)
		;

	enum_<smlp::type>("type")
		.value("Real", type::REAL)
		.value("Integer", type::INT)
		.export_values()
		;

	def("_mk_component_entire", mk_component_entire);
	def("_mk_component_ival", mk_component_ival);
	def("_mk_component_list", mk_component_list);

/*
	class_<kay::Z>("Z", init<signed long>())
		.def(init<const char *>())
		.def("__str__", Z_str);
*/
	class_<kay::Q>("Q", init<signed long>())
		.def(init<const char *>())
		.def(init<double>())
		.def("__str__", Q_str)
		.add_property("numerator", _Q_num)
		.add_property("denominator", _Q_den)
		.def("__eq__", _Q_cmp<EQ>)
		.def("__ne__", _Q_cmp<NE>)
		.def("__lt__", _Q_cmp<LT>)
		.def("__le__", _Q_cmp<LE>)
		.def("__gt__", _Q_cmp<GT>)
		.def("__ge__", _Q_cmp<GE>)
		.def("__add__", _Q_bin<bop2::ADD>)
		.def("__sub__", _Q_bin<bop2::SUB>)
		.def("__mul__", _Q_bin<bop2::MUL>)
		.def("__truediv__", _Q_div)
		.def("__pos__", _Q_un<uop2::UADD>)
		.def("__neg__", _Q_un<uop2::USUB>)
		.def("__abs__", _Q_abs)
		;
	def("_mk_Q_Z", _mk_Q_Z);
	def("_mk_Q_F", _mk_Q_F);
	def("_mk_Q_Q", _mk_Q_Q);

	class_<A>("A", no_init)
		.def("__eq__", do_cmp<EQ, A>)
		.def("__ne__", do_cmp<NE, A>)
		.def("__lt__", do_cmp<LT, A>)
		.def("__le__", do_cmp<LE, A>)
		.def("__gt__", do_cmp<GT, A>)
		.def("__ge__", do_cmp<GE, A>)
		.def("__str__", &A::get_str)
		.def("to_R", &A::to_R)
		.def("to_Q", (kay::Q (*)(const A &))[](const A &a){ return to_Q(a); })
		.def("known_Q", (bool (*)(const A &))[](const A &a){ return known_Q(a); })
		;

	class_<R>("R", no_init)
		.def("_approx", R_approx);
		;
	def("_lbound_log2", _lbound_log2);

	class_<sptr<solver>>("solver", no_init)
		.def("declare", solver_declare)
		.def("declare", solver_declare_dict)
		.def("add", solver_add)
		.def("check", solver_check)
		;
	def("_mk_solver", _mk_solver);

	class_<domain>("domain", no_init)
		.def(vector_indexing_suite<domain>())
		;
	class_<typename domain::value_type>("_domain_entry", no_init)
		.def_readwrite("name", &domain::value_type::first)
		.def_readwrite("comp", &domain::value_type::second)
		;
	def("_mk_domain", convert_domain_dict);
	// to_python_converter<domain,domain_to_dict,false>();

	class_<sat>("sat", no_init)
		.add_property("model", sat_get_model)
		;
	class_<unsat>("unsat", no_init);
	class_<unknown>("unknown", no_init)
		.def_readonly("reason", &unknown::reason)
		;
}
