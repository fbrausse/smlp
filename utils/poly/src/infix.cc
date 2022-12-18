/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "infix.hh"

using namespace smlp;

namespace {

struct infix_parser {

	enum type {
		NONE, INT, FLOAT, SPECIAL, IDENT,
	};

	str the_str;
	bool python_compat;
	char *s;
	type t;
	str tok;

	infix_parser(str s, bool python_compat)
	: the_str(move(s))
	, python_compat(python_compat)
	, s(the_str.data())
	{ next(); }

	void set(type t, str tok)
	{
		assert(t == NONE || !tok.empty());
		// fprintf(stderr, "set: tok: '%s'\n", tok.c_str());
		this->t = t;
		this->tok = move(tok);
	}

	void next()
	{
		while (isspace(*s))
			s++;
		if (!*s)
			return set(NONE, "");
		char *begin = s++;
		if (strchr("()[]{},", *begin))
			return set(SPECIAL, str(begin, s));
		if (strchr("<>!=", *begin) && *s == '=') {
			s++;
			return set(SPECIAL, str(begin, s));
		}
		if (strchr("<>", *begin))
			return set(SPECIAL, str(begin, s));
		if (isdigit(*begin) /*|| (strchr("+-", *begin) && (isdigit(*s) || *s == '.'))*/) {
			bool dot = false;
			while (isdigit(*s) || (!dot && *s == '.')) {
				dot |= *s == '.';
				s++;
			}
			type t = dot ? FLOAT : INT;
			if (strchr("eE", *s)) {
				t = FLOAT;
				s++;
				if (strchr("+-", *s))
					s++;
				while (isdigit(*s))
					s++;
			}
			return set(t, str(begin, s));
		}
		if (strchr("+-*^", *begin))
			return set(SPECIAL, str(begin, s));
		if (*begin == ':')
			*begin = '_';
		assert(isalpha(*begin) || *begin == '_' || *begin == '.');
		while (isalnum(*s) || *s == '_')
			s++;
		return set(IDENT, str(begin, s));
	}

	explicit operator bool() const { return t != NONE; }

	vec<expr> tuple()
	{
		vec<expr> v;
		if (t == SPECIAL && tok == ")")
			next();
		else
			while (1) {
				v.push_back(get());
				if (t == SPECIAL && tok == ",") {
					next();
					continue;
				}
				if (t == SPECIAL && tok == ")") {
					next();
					break;
				}
				abort();
			}
		return v;
	}

	expr low()
	{
		// fprintf(stderr, "low: tok: '%s', s: '%.*s'...\n", tok.c_str(), 20, s);
		assert(*this);
		switch (t) {
		case NONE: unreachable();
		case SPECIAL:
			if (tok == "+" || tok == "-") {
				str op = tok;
				next();
				return call { make1e(name { move(op) }), { low() } };
			}
			if (tok == "(") {
				next();
				expr e = get();
				assert(t == SPECIAL && tok == ")");
				next();
				return e;
			}
			abort(); /* not supported */
		case IDENT: {
			str id = move(tok);
			next();
			if (t == SPECIAL && tok == "(") {
				next();
				return call {
					make1e(name { move(id) }),
					tuple()
				};
			}
			if (id == ".")
				return cnst { "None" };
			return name { move(id) };
		}
		case FLOAT: {
			if (python_compat) {
				double d = atof(tok.c_str());
				next();
				char buf[64];
				sprintf(buf, "%.15g", d);
				return cnst { buf };
			} else {
				str s = move(tok);
				next();
				return cnst { move(s) };
			}
		}
		case INT: {
			str s = move(tok);
			next();
			return cnst { move(s) };
		}
		}
		unreachable();
	}

	expr exp()
	{
		expr r = low();
		if (t == SPECIAL && (tok == "^")) {
			next();
			r = call {
				make1e(name { "^" }),
				{ move(r), exp() },
			};
		}
		return r;
	}

	expr mul()
	{
		// fprintf(stderr, "mul: tok: '%s', s: '%.*s'...\n", tok.c_str(), 20, s);
		expr r = exp();
		while (t == SPECIAL && (tok == "*")) {
			next();
			r = call {
				make1e(name { "*" }),
				{ move(r), exp(), },
			};
		}
		return r;
	}

	expr add()
	{
		// fprintf(stderr, "add: tok: '%s', s: '%.*s'...\n", tok.c_str(), 20, s);
		expr r = mul();
		// fprintf(stderr, "add2: tok: '%s', s: '%.*s'...\n", tok.c_str(), 20, s);
		while (t == SPECIAL && (tok == "+" || tok == "-")) {
			str op = tok;
			next();
			r = call {
				make1e(name { move(op) }),
				{ move(r), mul(), },
			};
		}
		return r;
	}

	expr cmp()
	{
		expr r = add();
		if (t != SPECIAL)
			return r;
		cmp_t cmp;
		auto [rem,ec] = from_chars(tok.data(), tok.data() + tok.length(), cmp);
		if (rem != tok.data() + tok.length() || ec != std::errc {})
			return r;
		next();
		return cop { cmp, make1e(move(r)), make1e(add()) };
	}

	expr get()
	{
		assert(*this);
		return cmp();
	}
};

}

static str read_all(FILE *f)
{
	str r;
	static char BUF[4096];
	while (size_t rd = fread(BUF, 1, sizeof(BUF), f))
		r.append(BUF, rd);
	return r;
}

expr smlp::parse_infix(FILE *f, bool python_compat)
{
	return parse_infix(read_all(f), python_compat);
}

expr smlp::parse_infix(str s, bool python_compat)
{
	return infix_parser(move(s), python_compat).get();
}
