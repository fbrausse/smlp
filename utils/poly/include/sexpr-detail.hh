
#ifndef ES_SEXPR_DETAIL_HH
#define ES_SEXPR_DETAIL_HH

namespace es::_detail {

struct ign {};

template <typename T> struct rm_fst_s;
template <typename T> using rm_fst = typename rm_fst_s<T>::type;
template <size_t I, size_t... Is> struct rm_fst_s<std::index_sequence<I,Is...>> {
	using type = std::index_sequence<Is...>;
};

template <size_t I, typename T> struct cons_s;
template <size_t I, typename T> using cons = typename cons_s<I,T>::type;
template <size_t I, size_t... Is> struct cons_s<I,std::index_sequence<Is...>> {
	using type = std::index_sequence<I,Is...>;
};

template <typename T> struct inc_s { using type = T; };
template <typename T> using inc = typename inc_s<T>::type;
template <size_t I, size_t... Is> struct inc_s<std::index_sequence<I,Is...>> {
	using type = cons<I+1,inc<std::index_sequence<Is...>>>;
};

template <typename... Ts> struct ign_index_sequence_s;
template <typename... Ts> using ign_index_sequence = typename ign_index_sequence_s<Ts...>::type;

template <> struct ign_index_sequence_s<> {
	using type = std::index_sequence<>;
};

template <typename T, typename... Ts>
struct ign_index_sequence_s<T,Ts...> {
	using type = cons<0,inc<ign_index_sequence<Ts...>>>;
};

template <typename... Ts>
struct ign_index_sequence_s<ign,Ts...> {
	using type = inc<ign_index_sequence<Ts...>>;
};

static_assert(std::is_same_v<ign_index_sequence<int>,std::index_sequence<0>>);
static_assert(std::is_same_v<ign_index_sequence<ign>,std::index_sequence<>>);
static_assert(std::is_same_v<ign_index_sequence<ign,int>,std::index_sequence<1>>);
static_assert(std::is_same_v<ign_index_sequence<int,ign>,std::index_sequence<0>>);
static_assert(std::is_same_v<ign_index_sequence<int,ign,int>,std::index_sequence<0,2>>);
static_assert(std::is_same_v<ign_index_sequence<int,ign,int,ign>,std::index_sequence<0,2>>);
static_assert(std::is_same_v<ign_index_sequence<ign,int,ign,int,ign>,std::index_sequence<1,3>>);

template <size_t I, typename Arg, typename... Ts>
static inline auto extract(const std::vector<Arg> &e)
{
	using T = std::tuple_element_t<I,std::tuple<Ts...>>;
	if constexpr (std::is_same_v<std::remove_cv_t<T>,ign>)
		return nullptr;
	else if constexpr (std::is_same_v<std::remove_cv_t<T>,Arg>)
		return &e[I];
	else
		return std::get_if<T>(&e[I]);
}

template <typename Arg, typename... Ts, size_t... Js, size_t... Is>
static inline
std::optional<std::tuple<const std::tuple_element_t<Is,std::tuple<Ts...>> &...>>
_as_tuple(const std::vector<Arg> &e, std::index_sequence<Js...>,
                                     std::index_sequence<Is...>)
{
	if (e.size() != sizeof...(Ts))
		return {};
	std::tuple t = { extract<Js,Arg,Ts...>(e)... };
	if ((std::get<Is>(t) && ...))
		return { std::forward_as_tuple(*std::get<Is>(t)...) };
	return {};
}

}

#endif
