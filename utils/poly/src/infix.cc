
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
		if (strchr("+-*", *begin))
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
				decltype(uop::op) op = tok == "+" ? uop::UADD : uop::USUB;
				next();
				return uop { op, make1e(low()) };
			}
			if (tok == "(") {
				next();
				expr e = add();
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

	expr mul()
	{
		// fprintf(stderr, "mul: tok: '%s', s: '%.*s'...\n", tok.c_str(), 20, s);
		expr r = low();
		if (t == SPECIAL && (tok == "*")) {
			next();
			r = bop {
				bop::MUL,
				make1e(move(r)),
				make1e(mul()),
			};
		}
		return r;
	}

	expr add()
	{
		// fprintf(stderr, "add: tok: '%s', s: '%.*s'...\n", tok.c_str(), 20, s);
		expr r = mul();
		// fprintf(stderr, "add2: tok: '%s', s: '%.*s'...\n", tok.c_str(), 20, s);
		if (t == SPECIAL && (tok == "+" || tok == "-")) {
			decltype(bop::op) op = tok == "+" ? bop::ADD : bop::SUB;
			next();
			r = bop {
				op,
				make1e(move(r)),
				make1e(add())
			};
		}
		return r;
	}

	expr get()
	{
		assert(*this);
		return add();
	}
};

}

static str read_all(FILE *f)
{
	str r;
	static char BUF[4096];
	for (size_t rd; (rd = fread(BUF, 1, sizeof(BUF), f)) > 0;)
		r.append(BUF, rd);
	return r;
}

expr smlp::parse_infix(FILE *f, bool python_compat)
{
	return infix_parser(read_all(f), python_compat).get();
}
