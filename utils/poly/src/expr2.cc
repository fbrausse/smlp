
#include "expr2.hh"

using namespace smlp;

expr2 smlp::unroll(const expr &e, const hmap<str,fun<expr2(vec<expr2>)>> &funs)
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
