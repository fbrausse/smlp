from fractions import Fraction
import smlp

path = '../../../problems/alex/2022_07-2/x/a'
pp = smlp.parse_nn(path + '/model_gen_data_12.json',
                   path + '/model_complete_data_12.h5',
                   path + '/../new.spec',
                   path + '/data_bounds_data_12.json')

# pp = smlp.parse_poly(dom_path, expr_path)

# -- end of differences between NN and polynomial models --

io_bounds_dom = False
inject_reals  = False

alpha = pp.interpret_input_bounds(io_bounds_dom, inject_reals)

# TODO: If bounds on named outputs pp.funcs are given, scale them.

# TODO: parse objective specification, e.g. 'Pareto(obj1, obj2, ...)' or 'obj1'
#       -> C++ parse_obj_spec

obj = smlp.Var('RL__8e_09Hz') # TODO: use pp.funcs['RL__8e_09Hz']

# optimize T in obj_range such that (assuming direction is >=):
#
# E x . eta x /\
# A y . theta x y -> alpha y -> (beta y /\ obj y >= T)
#
# domain constraints from 'dom' have to hold for x and y.

lemmas = []

#slv = smlp.solver(True) # SMT solver instance
#
#slv_data = smlp.data_solver(path)
#slv_data.declare(domain)
#slv_data.add(constraint)
#slv_data.check() # solve
#
#class MySolver:
#	def add(constraint):
#		pass
#	def declare(domain):
#		pass
#	def check():
#		pass



def find_candidate(solver, pp : smlp.pre_problem, T : Fraction):
	cand_found = solver.check()
	if isinstance(cand_found, smlp.unknown):
		return None
	else:
		return cand_found

def find_counter_example(T : Fraction, cand):
	solver = smlp.solver(False)
	solver.declare(pp.dom)
	solver.add(not pp.theta(cand))
	solver.add(not alpha)
	solver.add(...)
	return solver.check() # returns UNSAT or a single SAT model

def generalize_counter_example(coex):
	return coex

def optimize_EA(pp : smlp.pre_problem, T : Fraction):
	candidate_solver = smlp.solver(incremental=True)
	candidate_solver.declare(pp.dom)
	candidate_solver.add(pp.eta)
	candidate_solver.add(alpha)
	# obj = Minimum(obj_i)
	candidate_solver.add(obj > smlp.Cnst(T))
	while True:
		ca = find_candidate(candidate_solver, pp, T)
		if isinstance(ca, smlp.sat):
			ce = find_counter_example(T, ca.model)
			if isinstance(ce, smlp.sat):
				lemma = generalize_counter_example(ce.model) # TODO: instantiate pp.theta on ( . , ce.model)
				candidate_solver.add(lemma)
				continue
			elif isinstance(ce, smlp.unsat):
				return ca.model
		else:
			return None

def pareto_EA(pp : smlp.pre_problem):
	pass

n = 10
thresholds = [Fraction(i, n) for i in range(0, n+1)]

for T in thresholds:
	optimize_EA(T)
	break
