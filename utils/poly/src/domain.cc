/* SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2022 Franz Brausse <franz.brausse@manchester.ac.uk>
 * Copyright 2022 The University of Manchester
 */

#include "domain.hh"

using namespace smlp;

namespace {

struct domain_parser {

	FILE *f;
	str line;

	explicit domain_parser(FILE *f) : f(f) {}

	domain_parser(const domain_parser &) = delete;
	domain_parser & operator=(const domain_parser &) = delete;

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
			s = t + strspn(t, WHITE);
			t = s + strcspn(s, WHITE);
			if (t == s)
				continue;
			if (*s == ':')
				*s = '_';
			str name(s, t);
			// fprintf(stderr, "debug: name: '%s'\n", name.c_str());
			s = t + strspn(t, WHITE);
			t = s + strcspn(s, WHITE);
			// fprintf(stderr, "debug: ignore: '%.*s'\n", (int)(t - s), s);
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
				range = ival { move(nums[0]), move(nums[1]) };
				break;
			case '}':
				assert(!nums.empty());
				range = list { move(nums) };
				break;
			}
			d.emplace_back(move(name), move(range));
		}
		return d;
	}
};

}

domain smlp::parse_domain(FILE *f)
{
	return domain_parser(f).get();
}

form2 smlp::domain_constraint(const str &var, const component &rng)
{
	return rng.match<form2>(
	[&](const list &lst) {
		vec<sptr<form2>> args;
		args.reserve(lst.values.size());
		for (const kay::Q &q : lst.values)
			args.emplace_back(make2f(prop2 {
				EQ,
				make2e(name { var }),
				make2e(cnst2 { q })
			}));
		return lbop2 { lbop2::OR, move(args) };
	},
	[&](const ival &iv) {
		vec<sptr<form2>> args;
		args.emplace_back(make2f(prop2 {
			GE,
			make2e(name { var }),
			make2e(cnst2 { iv.lo })
		}));
		args.emplace_back(make2f(prop2 {
			LE,
			make2e(name { var }),
			make2e(cnst2 { iv.hi }),
		}));
		return lbop2 { lbop2::AND, move(args) };
	});
}
