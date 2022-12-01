/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "prefix.hh"

using namespace smlp;

namespace {

struct pe_parser {

	FILE *f;
	str tok;

	explicit pe_parser(FILE *f) : f(f) {}

	pe_parser(const pe_parser &) = delete;
	pe_parser & operator=(const pe_parser &) = delete;

	str & next()
	{
		tok.clear();
		int c;
		while (isspace(c = getc(f)));
		do { tok.push_back(c); } while (!isspace(c = getc(f)));
		return tok;
	}

	uptr<expr> getu() { return std::make_unique<expr>(get()); }
	sptr<expr> gets() { return make1e(get()); }

	expr get()
	{
		next();
		// fprintf(stderr, "debug: next tok: '%s'\n", tok.c_str());
		assert(tok.size() == 1 || tok.size() == 2);
		if (tok.size() == 1) {
			switch (tok[0]) {
			case '+':
			case '-':
			case '*':
				return call { make1e(name { tok }), { get(), get() } };
			case 'x': return call { make1e(name { "+" }), { get() } };
			case 'y': return call { make1e(name { "-" }), { get() } };
			case 'C': {
				size_t n = strtol(next().c_str(), NULL, 0);
				vec<expr> args;
				args.reserve(n);
				sptr<expr> func = gets();
				assert(func->get<name>());
				while (n--)
					args.push_back(get());
				return call { move(func), move(args) };
			}
			case 'V': return cnst { next() };
			case 'N': return name { next() };
			}
		} else if (tok.size() == 2) {
			cmp_t cmp;
			auto [_,ec] = from_chars(tok.data(), tok.data() + tok.length(), cmp);
			if (ec == std::errc {})
				return cop { cmp, gets(), gets() };
		}
		MDIE(mod_poly,1,"unhandled operator '%s' in expression\n",
		     tok.c_str());
	}
};

}

void smlp::dump_pe(FILE *f, const expr &e)
{
	e.match(
	[&](const name &n) { fprintf(f, "N %s\n", n.id.c_str()); },
	[&](const cnst &c) { fprintf(f, "V %s\n", c.value.c_str()); },
	[&](const call &c) {
		const name *n = c.func->get<name>();
		bool have = true;
		if (n && n->id.length() == 1) {
			have = true;
			if (size(c.args) == 2 && strchr("+-*", n->id[0])) {
				fprintf(f, "%c\n", n->id[0]);
			} else if (size(c.args) == 1 && strchr("+-", n->id[0])) {
				fprintf(f, "%c\n", "xy"[n->id[0] == '+' ? 0 : 1]);
			} else
				have = false;
		}
		if (!have) {
			fprintf(f, "C %zu\n", c.args.size());
			dump_pe(f, *c.func);
		}
		for (const expr &a : c.args)
			dump_pe(f, a);
	},
	[&](const cop &c) {
		fprintf(f, "%s\n", cmp_s[c.cmp]);
		dump_pe(f, *c.left);
		dump_pe(f, *c.right);
	}
	);
}

expr smlp::parse_pe(FILE *f)
{
	return pe_parser(f).get();
}
