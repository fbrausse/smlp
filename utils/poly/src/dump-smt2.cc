/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "dump-smt2.hh"

using namespace smlp;

template <typename T>
static void dump_smt2_n(FILE *f, const char *op, const vec<sptr<T>> &args)
{
	fprintf(f, "(%s", op);
	for (const sptr<T> &t : args) {
		fprintf(f, " ");
		dump_smt2(f, *t);
	}
	fprintf(f, ")");
}

static void dump_smt2_bin(FILE *f, const char *op, const term2 &l, const term2 &r)
{
	fprintf(f, "(%s ", op);
	dump_smt2(f, l);
	fprintf(f, " ");
	dump_smt2(f, r);
	fprintf(f, ")");
}

template <typename T>
static void dump_smt2_un(FILE *f, const char *op, const T &o)
{
	fprintf(f, "(%s ", op);
	dump_smt2(f, o);
	fprintf(f, ")");
}

static void dump_smt2(FILE *f, const prop2 &p)
{
	dump_smt2_bin(f, cmp_smt2[p.cmp], *p.left, *p.right);
}

static void dump_smt2(FILE *f, const kay::Z &z)
{
	if (z < 0)
		fprintf(f, "(- %s)", kay::Z(-z).get_str().c_str());
	else
		fprintf(f, "%s", z.get_str().c_str());
}

static void dump_smt2(FILE *f, const kay::Q &q)
{
	if (q.get_den() == 1)
		dump_smt2(f, q.get_num());
	else {
		fprintf(f, "(/ ");
		dump_smt2(f, q.get_num());
		fprintf(f, " %s)", q.get_den().get_str().c_str());
	}
}

void smlp::dump_smt2(FILE *f, const form2 &e)
{
	e.match(
	[&](const prop2 &p) { dump_smt2(f, p); },
	[&](const lbop2 &b) {
		if (size(b.args) > 0)
			return dump_smt2_n(f, lbop_s[b.op], b.args);
		switch (b.op) {
		case lbop2::AND: fprintf(f, "true"); return;
		case lbop2::OR: fprintf(f, "false"); return;
		}
		unreachable();
	},
	[&](const lneg2 &n) { dump_smt2_un(f, "not", *n.arg); }
	);
}

void smlp::dump_smt2(FILE *f, const term2 &e)
{
	e.match(
	[&](const name &n){ fprintf(f, "%s", n.id.c_str()); },
	[&](const bop2 &b){ dump_smt2_bin(f, bop_s[b.op], *b.left, *b.right); },
	[&](const uop2 &u){ dump_smt2_un(f, uop_s[u.op], *u.operand); },
	[&](const cnst2 &c){
		c.value.match(
		[&](const str &) { abort(); },
		[&](const auto &v) { dump_smt2(f, v); }
		);
	},
	[&](const ite2 &i){
		fprintf(f, "(ite ");
		dump_smt2(f, *i.cond);
		fprintf(f, " ");
		dump_smt2(f, *i.yes);
		fprintf(f, " ");
		dump_smt2(f, *i.no);
		fprintf(f, ")");
	}
	);
}

void smlp::dump_smt2(FILE *f, const domain &d)
{
	for (const auto &[var,rng] : d) {
		const char *t = nullptr;
		switch (rng.type) {
		case component::INT: t = "Int"; break;
		case component::REAL: t = "Real"; break;
		}
		assert(t);
		fprintf(f, "(declare-const %s %s)\n", var.c_str(), t);
		dump_smt2_un(f, "assert", domain_constraint(var, rng));
		fprintf(f, "\n");
	}
}
