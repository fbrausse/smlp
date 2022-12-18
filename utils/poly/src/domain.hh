/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#pragma once

#include "expr2.hh"

namespace smlp {

/* Explicit list of rational values */
struct list {
	vec<kay::Q> values;
	bool operator==(const list &) const = default;
};

struct entire { bool operator==(const entire &) const = default; };

/* A component (of the domain) is either an interval or a list of rational
 * values */
struct component {

	sumtype<entire,ival,list> range;
	enum type type;

	bool contains(const kay::Q &v) const
	{
		return range.match(
		[](const entire &) { return true; },
		[&](const ival &i) { return i.contains(v); },
		[&](const list &l) {
			for (const kay::Q &q : l.values)
				if (q == v)
					return true;
			return false;
		}
		);
	}

	bool operator==(const component &) const = default;
};

/* Translates a component 'rng' and the appropriate variable name 'var' into a
 * constraint in form of a 'form2' formula. */
sptr<form2> domain_constraint(const str &var, const component &c);

/* The domain is an (ordered) list of pairs (name, component) */
struct domain : vec<pair<str,component>> {

	friend str fresh(const domain &dom, str base)
	{
		if (!dom[base])
			return base;
		base += "_";
		if (!dom[base])
			return base;
		for (kay::Z i=0;; i++) {
			str n = base + i.get_str();
			if (!dom[n])
				return n;
		}
	}

	const component * operator[](const std::string_view &s) const
	{
		for (const auto &[n,c] : *this)
			if (n == s)
				return &c;
		return nullptr;
	}

	component * operator[](const std::string_view &s)
	{
		for (auto &[n,c] : *this)
			if (n == s)
				return &c;
		return nullptr;
	}

	using vec<pair<str,component>>::vec;
	using vec<pair<str,component>>::operator[];
};

sptr<form2> domain_constraints(const domain &d);

/* Parses the DOMAIN-FILE, see poly.cc for details. */
domain parse_simple_domain(FILE *f);

}
