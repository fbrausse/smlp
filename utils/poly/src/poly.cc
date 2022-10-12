
#include "common.hh"

#include <kay/numbers.hh>

#include <z3++.h>

#define DIE(code,...) do { fprintf(stderr, __VA_ARGS__); exit(code); } while (0)

using namespace smlp;

namespace {

struct expr;

struct name { str id; };
struct call { sptr<expr> func; vec<expr> args; };
struct bop { enum { ADD, SUB, MUL, } op; sptr<expr> left, right; };
struct uop { enum { UADD, USUB, } op; sptr<expr> operand; };
struct cnst { str value; };

static const char *bop_s[] = { "+", "-", "*" };
static const char *uop_s[] = { "+", "-" };

struct expr : sumtype<name,call,bop,uop,cnst>
            , std::enable_shared_from_this<expr> {

	using sumtype<name,call,bop,uop,cnst>::sumtype;

	void dump_pe(FILE *f) const
	{
		match(
		[&](const name &n) { fprintf(f, "N %s\n", n.id.c_str()); },
		[&](const cnst &c) { fprintf(f, "V %s\n", c.value.c_str()); },
		[&](const call &c) {
			fprintf(f, "C %zu\n", c.args.size());
			c.func->dump_pe(f);
			for (const expr &a : c.args)
				a.dump_pe(f);
		},
		[&](const bop &o) {
			static const char ops[] = { '+', '-', '*', };
			fprintf(f, "%c\n", ops[o.op]);
			o.left->dump_pe(f);
			o.right->dump_pe(f);
		},
		[&](const uop &o) {
			static const char ops[] = { 'x', 'y', };
			fprintf(f, "%c\n", ops[o.op]);
			o.operand->dump_pe(f);
		}
		);
	}
};

template <typename... Ts>
static inline sptr<expr> make1e(Ts &&... ts)
{
	return std::make_shared<expr>(std::forward<Ts>(ts)...);
}

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
		assert(tok.size() == 1);
		switch (tok[0]) {
		case '+': return bop { bop::ADD, gets(), gets() };
		case '-': return bop { bop::SUB, gets(), gets() };
		case '*': return bop { bop::MUL, gets(), gets() };
		case 'x': return uop { uop::UADD, gets() };
		case 'y': return uop { uop::USUB, gets() };
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
		DIE(1,"unhandled operator '%s' in expression\n",tok.c_str());
	}
};

struct ival { kay::Q lo, hi; };
struct list { vec<kay::Q> values; };

struct component : sumtype<ival,list> {

	using sumtype<ival,list>::sumtype;

	friend bool is_real(const component &c)
	{
		if (c.get<ival>())
			return true;
		for (const kay::Q &q : c.get<list>()->values)
			if (q.get_den() != 1)
				return true;
		return 1 || false; /* always real: Z3 falls back to a slow method otherwise */
	}
};

struct domain : vec<pair<str,component>> {
};

using domain_ref = typename domain::const_iterator;

enum cmp_t { LE, LT, GE, GT, EQ, NE, };

static const char *cmp_s[] = { "<=", "<", ">=", ">", "==", "!=" };
static const char *cmp_smt2[] = { "<=", "<", ">=", ">", "=", "distinct" };

static inline bool is_strict(cmp_t c) { return (int)c & 0x1; }
static inline bool is_order(cmp_t c) { return !((int)c & 0x4); }
static inline cmp_t operator-(cmp_t c)
{
	switch (c) {
	case LE: return GE;
	case LT: return GT;
	case GE: return LE;
	case GT: return LT;
	case EQ:
	case NE:
		return c;
	}
	unreachable();
}
static inline cmp_t operator~(cmp_t c)
{
	switch (c) {
	case LE: return GT;
	case LT: return GE;
	case GE: return LT;
	case GT: return LE;
	case EQ: return NE;
	case NE: return EQ;
	}
	unreachable();
}

struct expr2;
struct form2;

struct prop2 { cmp_t cmp; sptr<expr2> left, right; };
struct lbop2 { enum { AND, OR } op; vec<form2> args; };
struct lneg2 { sptr<form2> arg; };

static const char *lbop_s[] = { "and", "or" };

struct form2 : sumtype<prop2,lbop2,lneg2>
             , std::enable_shared_from_this<form2> {

	using sumtype<prop2,lbop2,lneg2>::sumtype;
};

struct ite2 { form2 cond; sptr<expr2> yes, no; };
struct bop2 { decltype(bop::op) op; sptr<expr2> left, right; };
struct uop2 { decltype(uop::op) op; sptr<expr2> operand; };
struct cnst2 { sumtype<kay::Z,kay::Q,str> value; };

struct expr2 : sumtype<name,bop2,uop2,cnst2,ite2>
             , std::enable_shared_from_this<expr2> {

	using sumtype<name,bop2,uop2,cnst2,ite2>::sumtype;
};

template <typename... Ts>
static inline sptr<expr2> make2e(Ts &&... ts)
{
	return std::make_shared<expr2>(std::forward<Ts>(ts)...);
}

template <typename... Ts>
static inline sptr<form2> make2f(Ts &&... ts)
{
	return std::make_shared<form2>(std::forward<Ts>(ts)...);
}

static expr2 unroll(const expr &e, const hmap<str,fun<expr2(vec<expr2>)>> &funs)
{
	return e.match<expr2>(
	[&](const name &n) { return n; },
	[&](const cnst &c) {
		if (c.value == "None")
			return cnst2 { kay::Z(0) };
		if (c.value.find('.') == str::npos &&
		    c.value.find('e') == str::npos &&
		    c.value.find('E') == str::npos)
			return cnst2 { kay::Z(c.value) };
		return cnst2 { kay::Q_from_str(str(c.value).data()) };
	},
	[&](const bop &b) {
		return bop2 {
			b.op,
			make2e(unroll(*b.left, funs)),
			make2e(unroll(*b.right, funs)),
		};
	},
	[&](const uop &u) {
		return uop2 {
			u.op,
			make2e(unroll(*u.operand, funs)),
		};
	},
	[&](const call &c) {
		vec<expr2> args;
		args.reserve(c.args.size());
		for (const expr &e : c.args)
			args.push_back(unroll(e, funs));
		auto f = funs.find(c.func->get<name>()->id);
		assert(f != funs.end());
		return f->second(move(args));
	}
	);
}

struct problem {

	domain dom;
	form2 p;
};

static void dump_smt2(FILE *f, const expr2 &e);
static void dump_smt2(FILE *f, const form2 &e);

template <typename T>
static void dump_smt2_n(FILE *f, const char *op, const vec<T> &args)
{
	fprintf(f, "(%s", op);
	for (const T &t : args) {
		fprintf(f, " ");
		dump_smt2(f, t);
	}
	fprintf(f, ")");
}

static void dump_smt2_bin(FILE *f, const char *op, const expr2 &l, const expr2 &r)
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

static void dump_smt2(FILE *f, const form2 &e)
{
	e.match(
	[&](const prop2 &p) { dump_smt2(f, p); },
	[&](const lbop2 &b) { dump_smt2_n(f, lbop_s[b.op], b.args); },
	[&](const lneg2 &n) { dump_smt2_un(f, "not", *n.arg); }
	);
}

static void dump_smt2(FILE *f, const expr2 &e)
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
		dump_smt2(f, i.cond);
		fprintf(f, " ");
		dump_smt2(f, *i.yes);
		fprintf(f, " ");
		dump_smt2(f, *i.no);
		fprintf(f, ")");
	}
	);
}

static inline form2 domain_constraint(const str &var, const component &rng)
{
	return rng.match<form2>(
	[&](const list &lst) {
		vec<form2> args;
		args.reserve(lst.values.size());
		for (const kay::Q &q : lst.values)
			args.emplace_back(prop2 {
				EQ,
				make2e(name { var }),
				make2e(cnst2 { q })
			});
		return lbop2 { lbop2::OR, move(args) };
	},
	[&](const ival &iv) {
		vec<form2> args;
		args.emplace_back(prop2 {
			GE,
			make2e(name { var }),
			make2e(cnst2 { iv.lo })
		});
		args.emplace_back(prop2 {
			LE,
			make2e(name { var }),
			make2e(cnst2 { iv.hi }),
		});
		return lbop2 { lbop2::AND, move(args) };
	});
}

static void dump_smt2(FILE *f, const domain &d)
{
	for (const auto &[var,rng] : d) {
		fprintf(f, "(declare-const %s %s)\n", var.c_str(),
		        is_real(rng) ? "Real" : "Int");
		dump_smt2_un(f, "assert", domain_constraint(var, rng));
		fprintf(f, "\n");
	}
}

static void dump_smt2(FILE *f, const char *logic, const problem &p)
{
	fprintf(f, "(set-logic %s)\n", logic);
	dump_smt2(f, p.dom);
	fprintf(f, "(assert ");
	dump_smt2(f, p.p);
	fprintf(f, ")\n");
	fprintf(f, "(check-sat)\n");
}

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

struct file {

	FILE *f;

	file(const char *path, const char *mode)
	: f(fopen(path, mode))
	{}

	~file() { if (f) fclose(f); }

	operator FILE *() const { return f; }
};

struct z3_solver {

	z3::context ctx;
	z3::solver slv;
	hmap<str,z3::expr> symbols;

	explicit z3_solver(const domain &d)
	: slv(ctx)
	{
		for (const auto &[var,rng] : d) {
			const char *s = var.c_str();
			symbols.emplace(var, is_real(rng) ? ctx.real_const(s)
			                                  : ctx.int_const(s));
			add(domain_constraint(var, rng));
		}
	}

	void add(const form2 &f)
	{
		slv.add(interp(f));
	}

	z3::expr interp(const expr2 &e)
	{
		return e.match(
		[&](const name &n){
			auto it = symbols.find(n.id);
			assert(it != symbols.end());
			return it->second;
		},
		[&](const cnst2 &c){
			return c.value.match(
			[&](const str &s) -> z3::expr { abort(); },
			[&](const kay::Z &v){ return ctx.int_val(v.get_str().c_str()); },
			[&](const kay::Q &v){ return ctx.real_val(v.get_str().c_str()); }
			);
		},
		[&](const ite2 &i){
			return ite(interp(i.cond), interp(*i.yes), interp(*i.no));
		},
		[&](const bop2 &b){
			z3::expr l = interp(*b.left);
			z3::expr r = interp(*b.right);
			switch (b.op) {
			case bop::ADD: return l + r;
			case bop::SUB: return l - r;
			case bop::MUL: return l * r;
			}
			unreachable();
		},
		[&](const uop2 &u){
			z3::expr a = interp(*u.operand);
			switch (u.op) {
			case uop::UADD: return a;
			case uop::USUB: return -a;
			}
			unreachable();
		}
		);
	}

	z3::expr interp(const form2 &f)
	{
		return f.match(
		[&](const prop2 &p) {
			z3::expr l = interp(*p.left), r = interp(*p.right);
			switch (p.cmp) {
			case LE: return l <= r;
			case LT: return l <  r;
			case GE: return l >= r;
			case GT: return l >  r;
			case EQ: return l == r;
			case NE: return l != r;
			}
			unreachable();
		},
		[&](const lbop2 &b) {
			z3::expr_vector a(ctx);
			for (const form2 &f : b.args)
				a.push_back(interp(f));
			switch (b.op) {
			case lbop2::OR: return z3::mk_or(a);
			case lbop2::AND: return z3::mk_and(a);
			}
			unreachable();
		},
		[&](const lneg2 &n) { return !interp(*n.arg); }
		);
	}
};

static str read_all(FILE *f)
{
	str r;
	static char BUF[4096];
	for (size_t rd; (rd = fread(BUF, 1, sizeof(BUF), f)) > 0;)
		r.append(BUF, rd);
	return r;
}

struct infix_parser {

	enum type {
		NONE, INT, FLOAT, SPECIAL, IDENT,
	};

	str the_str;
	char *s;
	type t;
	str tok;

	infix_parser(str s) : the_str(s), s(the_str.data()) { next(); }

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
			double d = atof(tok.c_str());
			next();
			char buf[64];
			sprintf(buf, "%.15g", d);
			return cnst { buf };
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

static domain parse_domain_file(const char *path)
{
	file f(path, "r");
	if (!f)
		DIE(1,"error opening domain file path: %s: %s\n",
		    path,strerror(errno));
	return domain_parser(f).get();
}

static expr parse_expression_file(const char *path, bool infix)
{
	file f(path, "r");
	if (!f)
		DIE(1,"error opening expression file path: %s: %s\n",
		    path,strerror(errno));
	if (!infix)
		return pe_parser(f).get();
	return infix_parser(read_all(f)).get();
}

[[noreturn]]
static void usage(const char *program_name, int exit_code)
{
	FILE *f = exit_code ? stderr : stdout;
	fprintf(f, "usage: %s [-OPTS] [--] DOMAIN-FILE EXPR-FILE OP CNST\n",
	        program_name);
	if (!exit_code)
		fprintf(f,"\
\n\
Options [defaults]:\n\
  -F IFORMAT  determines the format of the EXPR-FILE; can be one of: 'infix',\n\
              'prefix' [infix]\n\
  -h          displays this help message\n\
  -n          dry run, do not solve the problem [no]\n\
  -p          dump the expression in Polish notation to stdout [no]\n\
  -s          dump the problem in SMT-LIB2 format to stdout [no]\n\
\n\
The DOMAIN-FILE is a text file containing the bounds for all variables in the\n\
form 'NAME -- RANGE' where NAME is the name of the variable and RANGE is either\n\
an interval of the form '[a,b]' or a list of specific values '{a,b,c,d,...}'.\n\
Empty lines are skipped.\n\
\n\
The EXPR-FILE contains a polynomial expression in the variables specified by the\n\
DOMAIN-FILE. The format is either an infix notation or the prefix notation also\n\
known as Polish notation. The expected format can be specified through the -F\n\
switch.\n\
\n\
The problem to be solved is specified by the two parameters OP CNST where OP is\n\
one of '<=', '<', '>=', '>', '==' and '!='. Remember quoting the OP on the shell\n\
to avoid unwanted redirections. CNST is a rational constant in the same format\n\
as those in the EXPR-FILE (if any).\n\
\n\
Developed by Franz Brausse <franz.brausse@manchester.ac.uk>.\n\
License: Apache 2.0; part of SMLP.\n\
");
	exit(exit_code);
}

int main(int argc, char **argv)
{
	bool solve = true;
	bool dump_pe = false;
	bool dump_smt2 = false;
	bool infix = true;

	for (int opt; (opt = getopt(argc, argv, ":F:hnps")) != -1;)
		switch (opt) {
		case 'F':
			if (!strcmp(optarg, "infix"))
				infix = true;
			else if (!strcmp(optarg, "prefix"))
				infix = false;
			else
				DIE(1,"\
error: option '-F' only supports 'infix' and 'prefix'\n");
			break;
		case 'h': usage(argv[0], 0);
		case 'n': solve = false; break;
		case 'p': dump_pe = true; break;
		case 's': dump_smt2 = true; break;
		case ':': DIE(1,"error: option '-%c' requires an argument\n",
		              optopt);
		case '?': DIE(1,"error: unknown option '-%c'\n",optopt);
		}
	if (argc - optind != 4)
		usage(argv[0], 1);

	domain d = parse_domain_file(argv[optind]);
	expr e = parse_expression_file(argv[optind+1], infix);

	hmap<str,fun<expr2(vec<expr2>)>> funs;
	funs["Match"] = [](vec<expr2> args) {
		assert(args.size() >= 2);
		const name *var = args.front().get<name>();
		expr2 r = move(args.back());
		for (int i=args.size()-3; i >= 1; i-=2)
			r = ite2 {
				prop2 {
					EQ,
					make2e(*var),
					make2e(move(args[i]))
				},
				make2e(move(args[i+1])),
				make2e(move(r)),
			};
		return r;
	};
	expr2 e2 = unroll(e, funs);

	size_t c;
	for (c=0; c<ARRAY_SIZE(cmp_s); c++)
		if (std::string_view(cmp_s[c]) == argv[optind+2])
			break;
	if (c == ARRAY_SIZE(cmp_s))
		DIE(1,"OP '%s' unknown\n",argv[optind+2]);

	expr2 rhs = unroll(cnst { argv[optind+3] }, funs);

	if (dump_pe)
		e.dump_pe(stdout);
	assert(d.size() == 10);
	/*
	list *l = d["_post"].get<list>();
	assert(l);*/

	problem p = {
		move(d),
		prop2 {
			(cmp_t)c,
			make2e(move(e2)),
			make2e(move(rhs)),
		},
	};

	const char *logic = "QF_NRA";
	for (const auto &[_,rng] : p.dom)
		if (!is_real(rng))
			logic = "QF_NIRA";

	if (dump_smt2)
		::dump_smt2(stdout, logic, p);

	if (solve) {
		z3_solver s(p.dom);
		s.slv.set("logic", logic);
		s.add(p.p);
		z3::check_result r = s.slv.check();
		switch (r) {
		case z3::sat: fprintf(stderr, "sat\n"); break;
		case z3::unsat: fprintf(stderr, "unsat\n"); break;
		case z3::unknown: fprintf(stderr, "unknown\n"); break;
		}
	}
}
