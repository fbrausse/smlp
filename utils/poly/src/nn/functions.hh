/*
 * functions.hh
 *
 * Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
 *
 * See the LICENSE file for terms of distribution.
 */

#ifndef IV_FUNCTIONS_HH
#define IV_FUNCTIONS_HH

#include <iostream>
#include <vector>
#include <memory>	/* std::unique_ptr */
#include <cassert>
#include <optional>
#include <variant>

#ifndef IV_DEBUG
# define IV_DEBUG(x)
#endif

#if (IV_ANON_FUNCTIONS-0)
# define IV_NS_FUNCTIONS
#else
# define IV_NS_FUNCTIONS iv::functions
#endif

#include <iv/common.hh>

namespace IV_NS_FUNCTIONS {

enum dim : ssize_t { ANY = -1, SCALAR = -2, };

[[maybe_unused]]
static inline bool compatible(const dim &a, const dim &b)
{
	return a == ANY || b == ANY || a == b;
}

template <typename T, typename S> using codomain_type =
	std::invoke_result_t<T,const typename T::template domain_type<S> &>;

/* ax+b */
template <typename A, typename B = A>
struct affine1 {

	template <typename S> using domain_type = S;

	A a;
	B b;

	friend affine1 inverse(const affine1 &f)
	{
		return { A(1)/f.a, -f.b/f.a };
	}

	template <typename S>
	auto operator()(const S &x) const //noexcept(noexcept(a*x+b))
	{
		return a*x + b;
	}

	friend dim input_dim(const affine1 &f) { return SCALAR; }
	friend dim output_dim(const affine1 &f) { return SCALAR; }

	std::ostream & print_fun(std::ostream &os) const
	{
		return os << "affine[" << a << "," << b << "]";
	}

	friend std::ostream & operator<<(std::ostream &os, const affine1 &a)
	{
		return a.print_fun(os);
	}
};

template <typename F>
struct pointwise_single {

	template <typename S> using domain_type   = std::vector<typename F::template domain_type  <S>>;

	F f;

	template <typename... Args>
	pointwise_single(Args &&... args)
	: f(std::forward<Args>(args)...)
	{}

	friend dim input_dim(const pointwise_single &) { return ANY; }
	friend dim output_dim(const pointwise_single &) { return ANY; }

	template <typename S>
	std::enable_if_t<std::is_same_v<domain_type<S>,std::vector<codomain_type<F,S>>>
	                ,std::vector<codomain_type<F,S>>>
	operator()(domain_type<S> x) const noexcept(noexcept(f(x.front())))
	{
		for (auto &v : x)
			v = f(v);
		return x;
	}

	template <typename S>
	std::enable_if_t<!std::is_same_v<domain_type<S>,std::vector<codomain_type<F,S>>>
	                ,std::vector<codomain_type<F,S>>>
	operator()(const domain_type<S> &x) const noexcept(noexcept(f(x.front())))
	{
		std::vector<codomain_type<F,S>> y;
		for (auto &v : x)
			y.push_back(f(v));
		return y;
	}

	friend std::ostream & operator<<(std::ostream &os, const pointwise_single &p)
	{
		return os << "pointwise_single<" << p.f << ">";
	}
};

template <typename F>
struct pointwise {

	template <typename S> using domain_type   = std::vector<typename F::template domain_type  <S>>;

	std::vector<F> f;

	pointwise(std::vector<F> f)
	: f(move(f))
	{}

	friend dim input_dim(const pointwise &p) { return (dim)size(p.f); }
	friend dim output_dim(const pointwise &p) { return (dim)size(p.f); }

	template <typename S>
	std::enable_if_t<std::is_same_v<domain_type<S>,std::vector<codomain_type<F,S>>>
	                ,std::vector<codomain_type<F,S>>>
	operator()(domain_type<S> x) const //noexcept(noexcept(f.front()(x.front())))
	{
		assert((dim)size(x) == input_dim(*this));
		for (size_t i=0; i<size(f); i++)
			x[i] = f[i](x[i]);
		return x;
	}

	template <typename S>
	std::enable_if_t<!std::is_same_v<domain_type<S>,std::vector<codomain_type<F,S>>>
	                ,std::vector<codomain_type<F,S>>>
	operator()(const domain_type<S> &x) const //noexcept(noexcept(f.front()(x.front())))
	{
		assert((dim)size(x) == input_dim(*this));
		std::vector<codomain_type<F,S>> y;
		for (size_t i=0; i<size(f); i++)
			y.push_back(f[i](x[i]));
		return y;
	}

	friend std::ostream & operator<<(std::ostream &os, const pointwise &p)
	{
		os << "pointwise<";
		bool first = true;
		for (const F &g : p.f) {
			if (!first)
				os << ",";
			os << g;
			first = false;
		}
		return os << ">";
	}
};

template <typename... As> struct composition;

namespace {
template <size_t n, typename... As>
void composition_assert_compat(const composition<As...> &t)
{
	if constexpr (n > 0) {
		assert(compatible(output_dim(std::get<n-1>(t.t)),
		                  input_dim(std::get<n>(t.t))));
		composition_assert_compat<n-1>(t);
	}
}
}

template <typename... As>
struct composition {

private:
	template <size_t n>
	dim _input_dim(dim d) const
	{
		if constexpr (n < sizeof...(As)) {
			if (d == ANY)
				d = _input_dim<n+1>(input_dim(std::get<n>(t)));
		}
		return d;
	}

	template <size_t n>
	dim _output_dim(dim d) const
	{
		if constexpr (n > 0) {
			if (d == ANY)
				d = _output_dim<n-1>(output_dim(std::get<n-1>(t)));
		}
		return d;
	}

	template <size_t n, typename T>
	std::enable_if_t<n == sizeof...(As),T> eval(T &&x) const
	{
		return std::forward<T>(x);
	}

	template <size_t n, typename T, typename = std::enable_if_t<(n < sizeof...(As))>>
	auto eval(T &&x) const
	{
		return eval<n+1>(std::get<n>(t)(std::forward<T>(x)));
	}

	template <size_t n>
	std::ostream & _print_comp(std::ostream &os) const
	{
		os << std::get<n>(t);
		if constexpr (n > 0) {
			os << " ∘ ";
			_print_comp<n-1>(os);
		}
		return os;
	}

public:
	std::tuple<As...> t;

	composition(As... as)
	: t(std::move(as)...)
	{
		if constexpr (sizeof...(As) > 0)
			composition_assert_compat<sizeof...(As)-1>(*this);
	}

	friend dim input_dim(const composition &c)
	{
		return c._input_dim<0>(ANY);
	}

	friend dim output_dim(const composition &c)
	{
		return c._output_dim<sizeof...(As)>(ANY);
	}

	template <typename S>
	auto operator()(S &&x) const
	{
		return eval<0>(std::forward<S>(x));
	}

	friend std::ostream & operator<<(std::ostream &os, const composition &c)
	{
		if constexpr (sizeof...(As) == 0)
			return os << "id";
		else if constexpr (sizeof...(As) == 1)
			return os << std::get<0>(c.t);
		else {
			os << "(";
			c._print_comp<sizeof...(As)-1>(os);
			return os << ")";
		}
	}
};

struct identity {
	template <typename S> using domain_type = S;
	template <typename S> auto operator()(S &&x) const { return std::forward<S>(x); }
	friend dim input_dim(const identity &) { return ANY; }
	friend dim output_dim(const identity &) { return ANY; }
	friend std::ostream & operator<<(std::ostream &os, const identity &)
	{ return os << "id"; }
};

template <typename F>
struct finite_composition {

	using func_type = F;

	void prepend(F &&f)
	{
		// assert(empty(funs) || f->get_output_dim() == input_dim(*this));
		assert(compatible(output_dim(f), input_dim(*this)));
		funs.insert(begin(funs), move(f));
	}

	void append(F &&f)
	{
		// assert(empty(funs) || f->get_input_dim() == output_dim(*this));
		// assert(empty(funs) || compatible_with_input_dim(*f, output_dim(*this)));
		assert(compatible(input_dim(f), output_dim(*this)));
		funs.push_back(std::move(f));
	}

	friend dim input_dim(const finite_composition &c) noexcept
	{
		dim r = ANY;
		for (auto it = c.funs.begin(); r == ANY && it != c.funs.end(); ++it)
			r = input_dim(*it);
		return r;
		// return empty(c.funs) ? ANY : c.funs.front()->get_input_dim();
		// assert(!empty(c.funs));
		// return c.funs.front()->get_input_dim();
	}

	friend dim output_dim(const finite_composition &c) noexcept
	{
		dim r = ANY;
		for (auto it = c.funs.rbegin(); r == ANY && it != c.funs.rend(); ++it)
			r = output_dim(*it);
		return r;
		// return empty(c.funs) ? ANY : c.funs.back()->get_output_dim();
		// assert(!empty(c.funs));
		// return c.funs.back()->get_output_dim();
	}

	template <typename T>
	auto operator()(T &&x) const
	{
		return eval<T>(std::forward<T>(x));
	}

	friend std::ostream & operator<<(std::ostream &os,
	                                 const finite_composition &c)
	{
		bool first = true;
		os << "(";
		for (auto it = rbegin(c.funs); it != rend(c.funs); ++it) {
			if (!first)
				os << " ∘ ";
			os << *it;
			first = false;
		}
		return os << ")";
	}

	template <size_t max, typename G, typename H>
	auto try_decompose_upto(G &&g, H &&dyn_fallback) const
	{
		if (size(funs) <= max)
			return _with_decompose_upto<max>(std::forward<G>(g));
		else
			return dyn_fallback(*this);
	}

	template <size_t max, typename G>
	auto try_decompose_upto(G &&g) const
	{
		return try_decompose_upto<max>(g, g);
	}

	std::vector<F> funs;

protected:
	template <size_t... ns>
	auto get_elements(std::index_sequence<ns...>) const
	{
		return composition(funs[ns]...);
	}

	template <size_t max, size_t i=0, typename G>
	auto _with_decompose_upto(G &&g) const
	{
		if constexpr (max == i) {
			return g(get_elements(std::make_index_sequence<i>{}));
		} else if (i == size(funs)) {
			return g(get_elements(std::make_index_sequence<i>{}));
		} else {
			return _with_decompose_upto<max,i+1>(std::forward<G>(g));
		}
	}

	template <typename T>
	auto eval(std::invoke_result_t<F,T> x) const
	{
		[[maybe_unused]] size_t i = 0;
		for (const F &f : funs) {
			IV_DEBUG(std::cerr << "composition: f_" << i++ << "(" << x << ")");
			x = f(x);
			IV_DEBUG(std::cerr << " = " << x << "\n");
		}
		return x;
	}
};

template <typename T> struct fun_ptr : std::unique_ptr<T> {

	using std::unique_ptr<T>::unique_ptr;

	fun_ptr(std::unique_ptr<T> &&v)
	: std::unique_ptr<T>(move(v))
	{}

	template <typename S>
	auto operator()(S &&x) const { return (*this->get())(x); }

	friend dim input_dim(const fun_ptr &f) { return input_dim(*f); }
	friend dim output_dim(const fun_ptr &f) { return output_dim(*f); }

	friend std::ostream & operator<<(std::ostream &os, const fun_ptr &p)
	{ return os << *p; }
};

template <typename T> struct opt_fun : protected std::optional<T> {

	template <typename S> using domain_type = S;

	using std::optional<T>::optional;

	using std::optional<T>::operator bool;
	using std::optional<T>::operator*;

	template <typename S>
	std::invoke_result_t<T,S> operator()(S &&x) const
	{
		if (*this)
			return (**this)(std::forward<S>(x));
		return std::forward<S>(x);
	}

	friend dim output_dim(const opt_fun &o)
	{
		return o ? output_dim(*o) : ANY;
	}

	friend dim input_dim(const opt_fun &o)
	{
		return o ? input_dim(*o) : ANY;
	}

	friend std::ostream & operator<<(std::ostream &os, const opt_fun &o)
	{
		os << "opt<";
		if (o)
			os << *o;
		return os << ">";
	}
};

template <typename T>
struct opt_fun<affine1<T>> : affine1<T> {

	opt_fun()
	: affine1<T> { .a = T(1), .b = T(0) }
	{}

	opt_fun(affine1<T> o)
	: affine1<T>(std::move(o))
	{}
};

template <typename F>
struct opt_fun<finite_composition<F>> : finite_composition<F> {

	opt_fun() = default;

	opt_fun(finite_composition<F> c)
	: finite_composition<F>(std::move(c))
	{}
};

template <typename F>
struct pointwise_single<opt_fun<F>> : opt_fun<pointwise_single<F>> {

	pointwise_single(opt_fun<F> &&f)
	: opt_fun<pointwise_single<F>> {
		f ? opt_fun<pointwise_single<F>>(std::move(*f))
		  : opt_fun<pointwise_single<F>>{}
	}
	{}
};

template <typename F> struct is_identity : std::false_type {};
template <typename F> constexpr bool is_identity_v = is_identity<F>::value;
template <> struct is_identity<identity> : std::true_type {};
template <typename F> struct is_identity<pointwise_single<F>> : is_identity<F> {};
template <typename F> struct is_identity<opt_fun<F>> : is_identity<F> {};


template <typename F> struct opt_fun_t_dispatch { using type = opt_fun<F>; };
template <typename F> using opt_fun_t = typename opt_fun_t_dispatch<F>::type;
template <typename F> struct opt_fun_t_dispatch<opt_fun<F>> { using type = opt_fun_t<F>; };


namespace detail {
	/* <https://en.cppreference.com/w/cpp/utility/variant/visit> */
	// helper type for the std::variant visitor #4
	template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
	template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
}

template <typename... Fs> struct var_fun {

	/* make std::variant a little bit easier to use as a sum type */
	struct sum_t : std::variant<Fs...> {

		using std::variant<Fs...>::variant;

		template <typename... Cs>
		friend auto match(const sum_t &s, Cs &&... cs)
		{
			using std::visit;
			return visit(detail::overloaded { std::forward<Cs>(cs)... },
			             static_cast<const std::variant<Fs...> &>(s));
		}
	};

	sum_t f;

	template <typename F
	         ,typename = std::enable_if_t<std::is_constructible_v<std::variant<Fs...>,F>>
	         >
	var_fun(F &&f) : f(std::forward<F>(f)) {}

	template <typename S> using domain_type = S;

	template <typename S>
	auto operator()(S &&x) const
	{
		return match(f, [&](const auto &f){ return f(std::forward<S>(x)); });
	}

	friend dim input_dim(const var_fun &v)
	{
		return match(v.f, [](const auto &f){ return input_dim(f); });
	}

	friend dim output_dim(const var_fun &v)
	{
		return match(v.f, [](const auto &f){ return output_dim(f); });
	}

	friend std::ostream & operator<<(std::ostream &os, const var_fun &v)
	{
		os << "var<";
		match(v.f, [&os](const auto &f){ os << f; });
		return os << ">";
	}
};

template <typename... F> struct var_fun_t_dispatch { using type = var_fun<F...>; };
template <typename... F> using var_fun_t = typename var_fun_t_dispatch<F...>::type;
template <typename F> struct var_fun_t_dispatch<F> { using type = F; };

#if 0
template <typename... Fs>
struct pointwise_single<var_fun<Fs...>> : var_fun_t<pointwise_single<Fs>...> {

private:
	using vctor_f = var_fun_t<pointwise_single<Fs>...> (*)(const var_fun<Fs...> &);

	template <size_t I>
	static constexpr vctor_f vctor()
	{
		using T = std::tuple_element_t<I,std::tuple<Fs...>>;
		return [](const var_fun<Fs...> &f) {
			return var_fun_t<pointwise_single<Fs>...>(
				pointwise_single<T>(std::get<I>(f.f))
			);
		};
	}

	template <size_t... Is>
	static constexpr auto vctors(std::index_sequence<Is...>)
	{
		return std::array<vctor_f,sizeof...(Fs)> { vctor<Is>()... };
	}

	static constexpr auto VCTORS = vctors(std::make_index_sequence<sizeof...(Fs)> {});

public:
	pointwise_single(const var_fun<Fs...> &f)
	: var_fun_t<pointwise_single<Fs>...>(VCTORS[f.f.index()](f))
	{}
};
#endif


#if 1
template <typename Id, typename F> /* TODO: F should be Fs... */
struct var_id_opt_proxy : opt_fun_t<F> {

	template <typename S> using domain_type = S;

	var_id_opt_proxy(const Id &) : opt_fun_t<F> {} {}
	var_id_opt_proxy(const F &f) : opt_fun_t<F> { f } {}
};

template <typename Id, typename F>
struct is_identity<var_id_opt_proxy<Id,F>> : is_identity<opt_fun<F>> {};

/* TODO: this should partition F,Gs... by is_identity_v<.> */
template <typename F, typename G>
struct var_fun_t_dispatch<F,G>
: std::conditional<is_identity_v<F>
                  ,var_id_opt_proxy<F,var_fun_t<G>>
                  ,var_fun<F,G>
                  > {};

/*
template <typename... Fs>
struct opt_var_id_proxy : var_fun_t<identity,Fs...> {

	opt_var_id_proxy() : var_fun_t<identity,Fs...> { identity {} } {}

	template <typename F,
	          typename = std::enable_if_t<std::is_constructible_v<var_fun_t<Fs...>,F>>>
	opt_var_id_proxy(F &&f) : var_fun_t<identity,Fs...>(std::forward<F>(f)) {}
};

template <typename... Fs>
struct opt_fun_t_dispatch<var_fun<Fs...>> {
	using type = opt_var_id_proxy<Fs...>;
};
*/
#endif

// template <typename F> struct var_fun_t_dispatch<opt_fun<F>> { using type = var_fun_t<identity,F>; }

} // end namespace functions

#endif
