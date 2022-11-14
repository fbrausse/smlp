/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "domain.hh"

using namespace smlp;

namespace {

struct simple_domain_parser {

	FILE *f;
	str line;

	explicit simple_domain_parser(FILE *f) : f(f) {}

	simple_domain_parser(const simple_domain_parser &) = delete;
	simple_domain_parser & operator=(const simple_domain_parser &) = delete;

	str & next()
	{
		line.clear();
		int c;
		while ((c = getc(f)) >= 0 && c != '\n')
			line.push_back(c);
		return line;
	}

	domain get()
	{
		static const char WHITE[] = " \t\r\n\f\v";
		static const char WHITE_COMMA[] = " \t\r\n\f\v,";

		domain d;
		while (!feof(f)) {
			next();
			char *s = line.data();
			char *t = line.data();
			/* skip initial space, parse variable name */
			s = t + strspn(t, WHITE);
			t = s + strcspn(s, WHITE);
			if (t == s)
				continue;
			if (*s == ':')
				*s = '_';
			str name(s, t);
			// fprintf(stderr, "debug: name: '%s'\n", name.c_str());

			/* skip space after name, skip middle token */
			s = t + strspn(t, WHITE);
			t = s + strcspn(s, WHITE);
			// fprintf(stderr, "debug: ignore: '%.*s'\n", (int)(t - s), s);

			/* skip space after middle token, parse range */
			s = t + strspn(t, WHITE);
			char delim[] = { *s, '\0' };
			switch (delim[0]) {
			case '[': delim[0] = ']'; break;
			case '{': delim[0] = '}'; break;
			default: DIE(1,"unexpected range start symbol '%s'\n",
			             delim);
			}
			s++;
			t = s + strcspn(s, delim);
			*t = '\0';
			// fprintf(stderr, "debug: range %s: '%s'\n", delim, s);
			vec<kay::Q> nums;
			t = s;
			while (true) {
				s = t + strspn(t, WHITE_COMMA);
				t = s + strcspn(s, WHITE_COMMA);
				if (s == t)
					break;
				*t++ = '\0';
				nums.push_back(kay::Q_from_str(s));
				s = t + strspn(t, WHITE_COMMA);
			}
			component range;
			switch (*delim) {
			case ']':
				assert(nums.size() == 2);
				range.type = component::REAL;
				range.range = ival { move(nums[0]), move(nums[1]) };
				break;
			case '}':
				assert(!nums.empty());
				range.type = component::INT;
				for (const kay::Q &q : nums)
					if (q.get_den() != 1) {
						range.type = component::REAL;
						break;
					}
				range.range = list { move(nums) };
				break;
			default:
				unreachable();
			}
			d.emplace_back(move(name), move(range));
		}
		return d;
	}
};

}

domain smlp::parse_simple_domain(FILE *f)
{
	return simple_domain_parser(f).get();
}

sptr<form2> smlp::domain_constraint(const str &var, const component &c)
{
	return c.range.match(
	[](const entire &) { return true2; },
	[&](const list &lst) {
		sptr<term2> v = make2t(name { var });
		vec<sptr<form2>> args;
		args.reserve(lst.values.size());
		for (const kay::Q &q : lst.values)
			args.emplace_back(make2f(prop2 { EQ, v, make2t(cnst2 { q }) }));
		return disj(move(args));
	},
	[&](const ival &iv) {
		sptr<term2> v = make2t(name { var });
		return conj({
			make2f(prop2 { GE, v, make2t(cnst2 { iv.lo }) }),
			make2f(prop2 { LE, v, make2t(cnst2 { iv.hi }) }),
		});
	});
}

sptr<form2> smlp::domain_constraints(const domain &d)
{
	vec<sptr<form2>> c;
	for (const auto &[var,rng] : d)
		c.emplace_back(domain_constraint(var, rng));
	return conj(move(c));
}
