/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "solver.hh"

#include <es/smtlib2-parser.hh>

namespace smlp {

struct process {

	file in, out, err;
	pid_t pid = -1;

	explicit process(const char *cmd);
	~process();
};

struct split_version : protected vec<int> {

	split_version() = default;
	split_version(std::initializer_list<int> l) : vec<int>(move(l)) {}

	using vec<int>::size;
	using vec<int>::empty;
	using vec<int>::begin;
	using vec<int>::end;
	using vec<int>::operator[];

	friend std::from_chars_result from_chars(const char *first,
	                                         const char *last,
	                                         split_version &v)
	{
		using std::from_chars;
		v.clear();
		int n;
		std::from_chars_result r = from_chars(first, last, n);
		if (r.ec != std::errc {})
			return r;
		do {
			v.push_back(n);
			first = r.ptr;
			if (*first != '.')
				break;
			r = from_chars(first + 1, last, n);
		} while (r.ec == std::errc {});
		return { first, std::errc {} };
	}

	friend str to_string(const split_version &a)
	{
		using std::to_string;
		str s;
		for (size_t i=0; i<a.size(); i++) {
			if (i)
				s += '.';
			s += to_string(a[i]);
		}
		return s;
	}

	friend auto operator==(const split_version &a, const split_version &b)
	{
		return static_cast<const vec<int> &>(a) ==
		       static_cast<const vec<int> &>(b);
	}

	friend auto operator<=>(const split_version &a, const split_version &b)
	{
		return static_cast<const vec<int> &>(a) <=>
		       static_cast<const vec<int> &>(b);
	}
};

struct ext_solver : process, solver {

	explicit ext_solver(const char *cmd, const char *logic = nullptr);
	void declare(const domain &d) override;
	void add(const sptr<form2> &f) override;
	result check() override;

private:
	es::smtlib2::parser out_s;
	str name, version;
	split_version parsed_version;
	hset<str> symbols;

	str get_info(const char *what);

	pair<hmap<size_t,kay::Q>,ival>
	parse_algebraic_z3(const str &var, const es::arg &p, const es::slit &n);
	cnst2 cnst2_from_smt2(const str &var, const es::arg &s);
	pair<str,sptr<term2>> parse_smt2_asgn(const es::sexpr &a);
};

}
