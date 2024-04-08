/*
 * sexpr.hh
 * 
 * Copyright 2022-2024 Franz Brauße <franz.brausse@manchester.ac.uk>
 *
 * See the LICENSE file for terms of distribution.
 */

#ifndef ES_SEXPR_HH
#define ES_SEXPR_HH

#include <cassert>
#include <cstring>

#include <string>
#include <variant>
#include <vector>
#include <optional>
#include <tuple>
#include <stdexcept>

#include "sexpr-detail.hh"

#if defined(__GNUC__)
# define ES_UNREACHABLE()	__builtin_unreachable()
#else
[[noreturn]] static inline void ES_UNREACHABLE() { /* intentional */ }
#endif

#ifdef NDEBUG
# define es_unreachable()	ES_UNREACHABLE()
#else
# define es_unreachable()	do { assert(0 && "unreachable"); ES_UNREACHABLE(); } while (0)
#endif

namespace es {

using _detail::ign;

struct sexpr;
struct slit : std::string {
	slit(std::string s) : std::string(std::move(s)) {}
};
using arg = std::variant<slit,sexpr>;
struct sexpr : std::vector<arg> {
	sexpr() = default;
	sexpr(std::initializer_list<arg> l) : std::vector<arg>(l) {}
};

static inline std::string to_string(const arg &a)
{
	if (auto *l = std::get_if<slit>(&a))
		return *l;
	std::string r = "(";
	const sexpr &s = std::get<sexpr>(a);
	for (size_t i=0; i<s.size(); i++) {
		if (i)
			r += " ";
		r += to_string(s[i]);
	}
	return r += ")";
}

template <typename... Ts>
static inline auto as_tuple(const std::vector<arg> &e)
{
	using namespace _detail;
	return _as_tuple<arg,Ts...>(e, std::index_sequence_for<Ts...> {},
	                            ign_index_sequence<Ts...> {});
}

struct as_tuple_error : std::runtime_error {
	as_tuple_error()
	: std::runtime_error::runtime_error("tuple unpack error")
	{}
};

template <typename... Ts>
static inline auto as_tuple_ex(const sexpr &e)
{
	if (auto o = as_tuple<Ts...>(e))
		return *o;
	throw as_tuple_error {};
}

template <typename... Ts>
static inline auto as_tuple_ex(const arg &a)
{
	if (auto *s = std::get_if<sexpr>(&a))
	if (auto o = as_tuple<Ts...>(*s))
		return *o;
	throw as_tuple_error {};
}

struct sexpr_parser {

	FILE *fin;
	struct pos {
		size_t line_no, col;
	} p;
	int c;
	const char *error = NULL;

	sexpr_parser(FILE *fin = stdin, size_t line_no = 1, size_t col = 0);

	virtual ~sexpr_parser() = default;

	explicit operator bool() const { return !error && c != EOF && !ferror(fin); }
	void fail(const char *msg) { error = msg; }

	virtual void get();
	void skip_space();

	virtual std::optional<slit> atom();
	std::optional<sexpr> compound();
	std::optional<sexpr> next();
};

struct formatter {

	size_t width = 80;
	size_t tab_w = 8;

	size_t indent = 0, pos = 0;
	FILE *f = stdout;

	void putc(char c);
	bool fits(std::optional<size_t> w, ssize_t add = 0) const;
	void emit(const arg &a, std::vector<size_t> &align);
	void spaces(size_t n);
	void emit(const sexpr &e, std::vector<size_t> &align);
	// void emit(const sexpr &e);
	void emit(const arg &e, bool newline = true);
};

}

#endif
