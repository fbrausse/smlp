/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "dump-smt2.hh"

using namespace smlp;

template <typename T>
static bool is_terminal(const T &g)
{
	return g.match(
	[](const name &) { return true; },
	[](const cnst2 &) { return true; },
	[](const auto &) { return false; }
	);
}

namespace {
struct smt2_output {

	FILE *f;
	hmap<const void *,size_t> m;

	explicit smt2_output(FILE *f) : f(f) {}

	void dump_smt2(const kay::Z &z) const
	{
		if (z < 0)
			fprintf(f, "(- %s)", kay::Z(-z).get_str().c_str());
		else
			fprintf(f, "%s", z.get_str().c_str());
	}

	void dump_smt2(const kay::Q &q) const
	{
		if (q.get_den() == 1)
			dump_smt2(q.get_num());
		else {
			fprintf(f, "(/ ");
			dump_smt2(q.get_num());
			fprintf(f, " %s)", q.get_den().get_str().c_str());
		}
	}

	template <typename T>
	void dump_smt2(const sptr<T> &p) const
	{
		if (auto it = m.find(p.get()); it != end(m))
			fprintf(f, "|:%zu|", it->second);
		else
			dump_smt2(*p);
	}

	template <typename T>
	void dump_smt2_n(const char *op, const vec<sptr<T>> &args) const
	{
		fprintf(f, "(%s", op);
		for (const sptr<T> &t : args) {
			fprintf(f, " ");
			dump_smt2(t);
		}
		fprintf(f, ")");
	}

	void dump_smt2_bin(const char *op, const sptr<term2> &l,
	                   const sptr<term2> &r) const
	{
		fprintf(f, "(%s ", op);
		dump_smt2(l);
		fprintf(f, " ");
		dump_smt2(r);
		fprintf(f, ")");
	}

	template <typename T>
	void dump_smt2_un(const char *op, const sptr<T> &o) const
	{
		fprintf(f, "(%s ", op);
		dump_smt2(o);
		fprintf(f, ")");
	}

	void dump_smt2(const form2 &e) const
	{
		e.match(
		[&](const prop2 &p) {
			dump_smt2_bin(cmp_smt2[p.cmp], p.left, p.right);
		},
		[&](const lbop2 &b) {
			if (size(b.args) > 0)
				return dump_smt2_n(lbop_s[b.op], b.args);
			switch (b.op) {
			case lbop2::AND: fprintf(f, "true"); return;
			case lbop2::OR: fprintf(f, "false"); return;
			}
			unreachable();
		},
		[&](const lneg2 &n) { dump_smt2_un("not", n.arg); }
		);
	}

	void dump_smt2(const term2 &e) const
	{
		e.match(
		[&](const name &n){ fprintf(f, "%s", n.id.c_str()); },
		[&](const bop2 &b){ dump_smt2_bin(bop_s[b.op], b.left, b.right); },
		[&](const uop2 &u){ dump_smt2_un(uop_s[u.op], u.operand); },
		[&](const cnst2 &c){
			c.value.match([&](const auto &v) { dump_smt2(v); });
		},
		[&](const ite2 &i){
			fprintf(f, "(ite ");
			dump_smt2(i.cond);
			fprintf(f, " ");
			dump_smt2(i.yes);
			fprintf(f, " ");
			dump_smt2(i.no);
			fprintf(f, ")");
		}
		);
	}

	template <typename T>
	void refs(const T &g, vec<sumtype<const term2 *, const form2 *>> &v)
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
				refs(*p, v);
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
	void lets(const T &g)
	{
		using E = sumtype<const term2 *, const form2 *>;
		vec<E> v;
		m.clear();
		refs(g, v);
		size_t n = 0;
		for (const E &e : v)
			e.match([&](const auto *p) {
				auto it = m.find(p);
				assert(it != end(m));
				if (it->second > 1) {
					fprintf(f, "(let ((|:%zu| ", n);
					it->second = n++;
					dump_smt2(*p);
					fprintf(f, ")) ");
				} else
					m.erase(it);
			});
		assert(size(m) == n);
		dump_smt2(g);
		for (size_t i=0; i<n; i++)
			fprintf(f, ")");
	}
};
}

template <typename T>
static void dump_smt2(FILE *f, const T &e, bool let)
{
	smt2_output o(f);
	if (let)
		o.lets(e);
	else
		o.dump_smt2(e);
}

void smlp::dump_smt2(FILE *f, const form2 &e, bool let)
{
	::dump_smt2(f, e, let);
}

void smlp::dump_smt2(FILE *f, const term2 &e, bool let)
{
	::dump_smt2(f, e, let);
}

void smlp::dump_smt2(FILE *f, const domain &d)
{
	for (const auto &[var,rng] : d) {
		const char *t = nullptr;
		switch (rng.type) {
		case type::INT: t = "Int"; break;
		case type::REAL: t = "Real"; break;
		}
		assert(t);
		fprintf(f, "(declare-const %s %s)\n", var.c_str(), t);
		fprintf(f, "(assert ");
		dump_smt2(f, *domain_constraint(var, rng));
		fprintf(f, ")\n");
	}
}

#ifdef _GNU_SOURCE
static const cookie_io_functions_t str_cookie = {
	/* .read = */
	[](void *, char *, size_t) -> ssize_t { return -1; },
	/* .write = */
	[](void *c, const char *buf, size_t sz) -> ssize_t {
		str *s = static_cast<str *>(c);
		s->append(buf, sz);
		return sz;
	},
	/* .seek = */
	[](void *, off64_t *, int) { return -1; },
	/* .close = */
	[](void *) { return 0; },
};
template <typename T>
static str to_string_help(const sptr<T> &g, bool let)
{
	str s;
	FILE *f = fopencookie(&s, "w", str_cookie);
	::dump_smt2(f, *g, let);
	fclose(f);
	return s;
}
#elif _POSIX_C_SOURCE >= 200809L
template <typename T>
static str to_string_help(const sptr<T> &g, bool let)
{
	char *buf = NULL;
	size_t sz = 0;
	FILE *f = open_memstream(&buf, &sz);
	::dump_smt2(f, *g, let);
	fclose(f);
	str r(buf, sz);
	free(buf);
	return r;
}
#else
# error "no implementation for to_string(sptr<T>, bool) available"
#endif

str detail::to_string(const sptr<term2> &g, bool let)
{
	return to_string_help(g, let);
}

str detail::to_string(const sptr<form2> &g, bool let)
{
	return to_string_help(g, let);
}
