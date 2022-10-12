/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <functional>
#include <unordered_map>
#include <optional>

#ifdef NDEBUG
# define unreachable() __builtin_unreachable()
#else
# define unreachable() abort()
#endif

#define ARRAY_SIZE(...)	(sizeof(__VA_ARGS__)/sizeof(*(__VA_ARGS__)))
#define DIE(code,...) do { fprintf(stderr, __VA_ARGS__); exit(code); } while (0)

namespace smlp {

/* Common definitions to allow for a more concise language than that of the C++
 * std library:
 *
 * - hmap<K,V>   : hash-map (std::unordered_map)
 * - str         : std::string
 * - vec         : std::vector
 * - uptr<T>     : std::unique_ptr
 * - sptr<T>     : std::shared_ptr
 * - fun         : std::function
 * - opt         : std::optional
 * - pair        : std::pair
 * - move        : std::move
 * - sumtype<...>: std::variant with a more intuitive name and '.match()' member
 *                 function to access its contents
 */

using namespace std::literals::string_view_literals;

template <typename K, typename V,
          typename Hash = std::hash<K>,
          typename Eq = std::equal_to<K>>
using hmap = std::unordered_map<K,V,Hash,Eq>;

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

// helper type for the visitor #4
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
// template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename... Ts>
struct sumtype : std::variant<Ts...> {

	using std::variant<Ts...>::variant;

	template <typename... As>
	auto match(As &&... o) const
	{
		return std::visit(overloaded { std::forward<As>(o)... }, *this);
	}

	template <typename R, typename... As>
	R match(As &&... o) const
	{
		return std::visit<R>(overloaded { std::forward<As>(o)... }, *this);
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

} // end namespace smlp
