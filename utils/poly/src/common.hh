/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "config.h"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdarg>

#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <functional>
#include <unordered_set>
#include <unordered_map>
#include <optional>
#include <system_error>

#include <time.h>

#ifdef NDEBUG
# define unreachable() __builtin_unreachable()
#else
# define unreachable() abort()
#endif

#define ARRAY_SIZE(...)	(sizeof(__VA_ARGS__)/sizeof(*(__VA_ARGS__)))
#define DIE(code,...) do { fprintf(stderr, __VA_ARGS__); exit(code); } while (0)
#define MDIE(mod,code,...) do { err(mod, __VA_ARGS__); exit(code); } while (0)

namespace smlp {

/* Common definitions to allow for a more concise language than that of the C++
 * std library:
 *
 * - hmap<K,V>   : hash-map (std::unordered_map)
 * - str         : std::string
 * - vec         : std::vector
 * - uptr<T>     : std::unique_ptr, non-copyable heap-allocated T
 * - sptr<T>     : std::shared_ptr, reference-counted heap-allocated T
 * - fun         : std::function
 * - opt         : std::optional
 * - sumtype<...>: std::variant with a more intuitive name and '.match()' member
 *                 function to access its contents
 *
 * Imports with the same name as in std:
 * - pair
 * - move
 * - to_string
 * - min, max
 */

using namespace std::literals::string_view_literals;

template <typename K, typename V,
          typename Hash = std::hash<K>,
          typename Eq = std::equal_to<K>>
using hmap = std::unordered_map<K,V,Hash,Eq>;

template <typename K,
          typename Hash = std::hash<K>,
          typename Eq = std::equal_to<K>>
using hset = std::unordered_set<K,Hash,Eq>;

using str = std::string;

template <typename T>
using vec = std::vector<T>;

template <typename T, typename D = std::default_delete<T>>
using uptr = std::unique_ptr<T,D>;

template <typename T>
using sptr = std::shared_ptr<T>;

template <typename T>
using fun = std::function<T>;

using std::move;

template <typename T>
using opt = std::optional<T>;

using std::pair;

using std::to_string;

using std::max;
using std::min;

using std::swap;

using strview = std::string_view;

// helper type for the visitor #4
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename... Ts>
struct sumtype : std::variant<Ts...> {

	using std::variant<Ts...>::variant;

	template <typename... As>
	auto match(As &&... o) const
	{
		return std::visit(overloaded { std::forward<As>(o)... },
		                  static_cast<const std::variant<Ts...> &>(*this));
	}

	template <typename R, typename... As>
	R match(As &&... o) const
	{
		return std::visit<R>(overloaded { std::forward<As>(o)... },
		                     static_cast<const std::variant<Ts...> &>(*this));
	}

	template <typename... As>
	auto match(As &&... o)
	{
		return std::visit(overloaded { std::forward<As>(o)... },
		                  static_cast<std::variant<Ts...> &>(*this));
	}

	template <typename R, typename... As>
	R match(As &&... o)
	{
		return std::visit<R>(overloaded { std::forward<As>(o)... },
		                     static_cast<std::variant<Ts...> &>(*this));
	}

	template <typename T>
	const T * get() const
	{
		return std::get_if<T>(this);
	}

	template <typename T>
	T * get()
	{
		return std::get_if<T>(this);
	}
};

struct file {

	FILE *f;

	file() : f(nullptr) {}

	file(const char *path, const char *mode)
	: f(fopen(path, mode))
	{}

	file(int fd, const char *mode)
	: f(fdopen(fd, mode))
	{}

	~file() { if (f) ::fclose(f); }

	file(const file &) = delete;
	file(file &&o) : f(o.f) { o.f = nullptr; }

	friend void swap(file &a, file &b)
	{
		swap(a.f, b.f);
	}

	file & operator=(file o) { swap(*this, o); return *this; }

	operator FILE *() const { return f; }

	friend int fclose(file &f)
	{
		if (f.f) {
			f.close();
			return 0;
		}
		errno = EINVAL;
		return EOF;
	}

	void close() { assert(f); *this = file(); }
};

struct timing : timespec {

	timing()
	{
		if (clock_gettime(CLOCK_MONOTONIC, this) == -1)
			throw std::error_code(errno, std::system_category());
	}

	friend timing & operator-=(timing &a, const timing &b)
	{
		a.tv_sec -= b.tv_sec;
		if ((a.tv_nsec -= b.tv_nsec) < 0) {
			a.tv_sec--;
			a.tv_nsec += 1e9;
		}
		return a;
	}

	friend timing operator-(timing a, const timing &b)
	{
		return a -= b;
	}

	operator double() const { return tv_sec + tv_nsec / 1e9; }
};

enum loglvl : int {
	QUIET,
	ERROR,
	WARN,
	INFO,
	NOTE,
	DEBUG,
};

struct Module {

	static hmap<strview,Module *> modules;

	const char *name;
	const char *color;
	loglvl lvl;

	explicit Module(const char *name, const char *color = "", loglvl lvl = NOTE);

	bool logs(loglvl l) const { return l <= lvl; }
	friend bool logs(const Module &m, loglvl l) { return m.logs(l); }

	bool vlog(loglvl, const char *fmt, va_list) const;

#define BODY(level) { \
		va_list ap;                      \
		va_start(ap,fmt);                \
		bool r = m.vlog(level, fmt, ap); \
		va_end(ap);                      \
		return r;                        \
	}
	[[gnu::format(printf,3,4)]] friend bool log(const Module &m, loglvl level, const char *fmt, ...) BODY(level)
	[[gnu::format(printf,2,3)]] friend bool err(const Module &m, const char *fmt, ...) BODY(ERROR)
	[[gnu::format(printf,2,3)]] friend bool warn(const Module &m, const char *fmt, ...) BODY(WARN)
	[[gnu::format(printf,2,3)]] friend bool info(const Module &m, const char *fmt, ...) BODY(INFO)
	[[gnu::format(printf,2,3)]] friend bool note(const Module &m, const char *fmt, ...) BODY(NOTE)
	[[gnu::format(printf,2,3)]] friend bool dbg(const Module &m, const char *fmt, ...) BODY(DEBUG)
#undef BODY

	static bool log_color;
};

extern Module mod_ext, mod_z3, mod_ival, mod_crit, mod_nn, mod_poly, mod_prob;
extern Module mod_smlp, mod_cand, mod_coex, mod_par;

extern opt<str> ext_solver_cmd;
extern opt<str> inc_solver_cmd;
extern long intervals;

void set_loglvl(char *arg);

void signal_backtrace_handler(int);

} // end namespace smlp

extern "C" const char SMLP_VERSION[];

namespace std {

template <typename... Ts>
struct variant_size<smlp::sumtype<Ts...>>
: variant_size<variant<Ts...>> {};

template <size_t n, typename... Ts>
struct variant_alternative<n,smlp::sumtype<Ts...>>
: variant_alternative<n,variant<Ts...>> {};

}
