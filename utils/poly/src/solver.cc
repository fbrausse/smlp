
#include "solver.hh"
#include "ext-solver.hh"
#include "ival-solver.hh"
#include "nn.hh"

#ifdef SMLP_ENABLE_Z3_API
# include "z3-solver.hh"
#endif

using namespace smlp;

interruptible *interruptible::is_active;

// template <typename T>
str smlp::smt2_logic_str(const domain &dom, const sptr<form2> &e)
{
	bool reals = false;
	bool ints = false;
	for (const auto &[_,rng] : dom)
		switch (rng.type) {
		case type::INT: ints = true; break;
		case type::REAL: reals = true; break;
		}
	str logic = "QF_";
	if (ints || reals) {
		logic += is_nonlinear(e) ? 'N' : 'L';
		if (ints)
			logic += 'I';
		if (reals)
			logic += 'R';
		logic += 'A';
	} else
		logic += "UF";
	return logic;
}

pair<const Module *,uptr<solver>> smlp::mk_solver0_(bool incremental, const char *logic)
{
	const char *ext = ext_solver_cmd ? ext_solver_cmd->c_str() : nullptr;
	const char *inc = inc_solver_cmd ? inc_solver_cmd->c_str() : nullptr;
	const char *cmd = (inc && ext ? incremental : !ext) ? inc : ext;
	if (cmd)
		return { &mod_ext, std::make_unique<ext_solver>(cmd, logic) };
#ifdef SMLP_ENABLE_Z3_API
	return { &mod_z3, std::make_unique<z3_solver>(logic) };
#endif
	MDIE(mod_smlp,1,"no solver specified and none are built-in, require "
	                "external solver via -S or -I\n");
}

uptr<solver> smlp::mk_solver0(bool incremental, const char *logic)
{
	return mk_solver0_(incremental, logic).second;
}

uptr<solver> smlp::mk_solver(bool incremental, const char *logic)
{
	if (intervals >= 0) {
		vec<pair<const Module *,uptr<solver>>> slvs;
		slvs.emplace_back(&mod_ival, std::make_unique<ival_solver>(intervals, logic));
		slvs.emplace_back(&mod_crit, std::make_unique<crit_solver>());
		slvs.emplace_back(mk_solver0_(incremental, logic));
		return std::make_unique<solver_seq>(move(slvs));
	}
	return smlp::mk_solver0(incremental, logic);
}

solver::all_solutions_iter_owned
smlp::all_solutions(const domain &dom, const sptr<form2> &f)
{
	uptr<solver> s = mk_solver0(true, smt2_logic_str(dom, f).c_str());
	s->declare(dom);
	s->add(f);
	return solver::all_solutions_iter_owned { move(s) };
}

sptr<form2> pre_problem::interpret_input_bounds(bool bnds_dom, bool inject_reals)
{
	if (inject_reals) {
		assert(bnds_dom || empty(input_bounds));
		/* First convert all int-components that are unbounded in the
		 * domain to lists where we have bounds; do not remove the
		 * in_bnds constraints as they are useful for some solvers
		 * (like Z3, which falls back to a slow method with mixed reals
		 * and integers). */
		for (const auto &[n,i] : input_bounds) {
			component *c = dom[n];
			assert(c);
			if (c->type != type::INT)
				continue;
			if (!c->range.get<entire>())
				continue;
			list l;
			using namespace kay;
			for (Z z = ceil(i.lo); z <= floor(i.hi); z++)
				l.values.emplace_back(z);
			c->range = move(l);
		}
		/* Next, convert all lists to real */
		for (auto &[v,c] : dom)
			if (c.range.get<list>())
				c.type = type::REAL;
	}

	if (bnds_dom)
		for (auto it = begin(input_bounds); it != end(input_bounds);) {
			component *c = dom[it->first];
			assert(c);
			if (c->range.get<entire>()) {
				c->range = it->second;
				it = input_bounds.erase(it);
			} else {
				++it;
			}
		}

	vec<sptr<form2>> c;
	for (const auto &[n,i] : input_bounds) {
		sptr<term2> v = make2t(name { n });
		c.emplace_back(conj({
			make2f(prop2 { GE, v, make2t(cnst2 { i.lo }) }),
			make2f(prop2 { LE, v, make2t(cnst2 { i.hi }) }),
		}));
	}
	return conj(move(c));
}

