/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "dump-smt2.hh"

using namespace smlp;

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

static void dump_smt2(FILE *f, const term2 &e, const hmap<const void *,size_t> &m);
static void dump_smt2(FILE *f, const form2 &e, const hmap<const void *,size_t> &m);

template <typename T>
static void dump_smt2(FILE *f, const sptr<T> &p, const hmap<const void *,size_t> &m)
{
	if (auto it = m.find(p.get()); it != end(m))
		fprintf(f, "|:%zu|", it->second);
	else
		dump_smt2(f, *p, m);
}

template <typename T>
static void dump_smt2_n(FILE *f, const char *op, const vec<sptr<T>> &args,
                        const hmap<const void *,size_t> &m)
{
	fprintf(f, "(%s", op);
	for (const sptr<T> &t : args) {
		fprintf(f, " ");
		dump_smt2(f, t, m);
	}
	fprintf(f, ")");
}

static void dump_smt2_bin(FILE *f, const char *op, const sptr<term2> &l,
                          const sptr<term2> &r, const hmap<const void *,size_t> &m)
{
	fprintf(f, "(%s ", op);
	dump_smt2(f, l, m);
	fprintf(f, " ");
	dump_smt2(f, r, m);
	fprintf(f, ")");
}

template <typename T>
static void dump_smt2_un(FILE *f, const char *op, const sptr<T> &o,
                         const hmap<const void *,size_t> &m)
{
	fprintf(f, "(%s ", op);
	dump_smt2(f, o, m);
	fprintf(f, ")");
}

static void dump_smt2(FILE *f, const form2 &e, const hmap<const void *,size_t> &m)
{
	e.match(
	[&](const prop2 &p) {
		dump_smt2_bin(f, cmp_smt2[p.cmp], p.left, p.right, m);
	},
	[&](const lbop2 &b) {
		if (size(b.args) > 0)
			return dump_smt2_n(f, lbop_s[b.op], b.args, m);
		switch (b.op) {
		case lbop2::AND: fprintf(f, "true"); return;
		case lbop2::OR: fprintf(f, "false"); return;
		}
		unreachable();
	},
	[&](const lneg2 &n) { dump_smt2_un(f, "not", n.arg, m); }
	);
}

static void dump_smt2(FILE *f, const term2 &e, const hmap<const void *,size_t> &m)
{
	e.match(
	[&](const name &n){ fprintf(f, "%s", n.id.c_str()); },
	[&](const bop2 &b){ dump_smt2_bin(f, bop_s[b.op], b.left, b.right, m); },
	[&](const uop2 &u){ dump_smt2_un(f, uop_s[u.op], u.operand, m); },
	[&](const cnst2 &c){
		c.value.match([&](const auto &v) { dump_smt2(f, v); });
	},
	[&](const ite2 &i){
		fprintf(f, "(ite ");
		dump_smt2(f, i.cond, m);
		fprintf(f, " ");
		dump_smt2(f, i.yes, m);
		fprintf(f, " ");
		dump_smt2(f, i.no, m);
		fprintf(f, ")");
	}
	);
}

template <typename T>
static bool is_terminal(const T &g)
{
	return g.match(
	[](const name &) { return true; },
	[](const cnst2 &) { return true; },
	[](const auto &) { return false; }
	);
}

template <typename T>
static void refs(const T &g, hmap<const void *,size_t> &m,
                 vec<sumtype<const term2 *, const form2 *>> &v)
{
	auto ref = [&] <typename U> (const sptr<U> &p) {
		bool process = true;
		bool record = false;
		if (is_terminal(*p))
			return;
		if (p.use_count() > 1) {
			process = !m.contains(p.get());
			record = true;
		}
		if (process)
			refs(*p, m, v);
		if (record && !m[p.get()]++)
			v.emplace_back(p.get());
	};

	g.match(
	[&](const name &) {},
	[&](const cnst2 &) {},
	[&](const bop2 &b) { ref(b.left); ref(b.right); },
	[&](const uop2 &u) { ref(u.operand); },
	[&](const ite2 &i) { ref(i.cond); ref(i.yes); ref(i.no); },
	[&](const prop2 &p) { ref(p.left); ref(p.right); },
	[&](const lbop2 &b) { for (const sptr<form2> &a : b.args) ref(a); },
	[&](const lneg2 &n) { ref(n.arg); }
	);
}

template <typename T>
static void lets(FILE *f, const T &g)
{
	using E = sumtype<const term2 *, const form2 *>;
	hmap<const void *,size_t> m;
	vec<E> v;
	refs(g, m, v);
	size_t n = 0;
	for (const E &e : v)
		e.match([&](const auto *p) {
			auto it = m.find(p);
			assert(it != end(m));
			if (it->second > 1) {
				fprintf(f, "(let ((|:%zu| ", n);
				it->second = n++;
				dump_smt2(f, *p, m);
				fprintf(f, ")) ");
			} else
				m.erase(it);
		});
	assert(size(m) == n);
	dump_smt2(f, g, m);
	for (size_t i=0; i<n; i++)
		fprintf(f, ")");
}

void smlp::dump_smt2(FILE *f, const form2 &e, bool let)
{
	if (let)
		lets(f, e);
	else
		dump_smt2(f, e, hmap<const void *,size_t>{});
}

void smlp::dump_smt2(FILE *f, const term2 &e, bool let)
{
	if (let)
		lets(f, e);
	else
		dump_smt2(f, e, hmap<const void *,size_t>{});
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
		fprintf(f, "(assert ");
		dump_smt2(f, domain_constraint(var, rng), hmap<const void *,size_t>{});
		fprintf(f, ")\n");
	}
}
