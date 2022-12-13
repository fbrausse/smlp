from .libsmlp import *

import fractions

#__doc__ = ""

true = true()
true.__doc__ = "The constant true formula."
false = false()
false.__doc__ = "The constant false formula."
zero = zero()
zero.__doc__ = "The numeric constant 0."
one = one()
one.__doc__ = "The numeric constant 1."

term2.__repr__ = lambda self: '<' + self.__module__ + '.term2 ' + str(self) + '>'
form2.__repr__ = lambda self: '<' + self.__module__ + '.form2 ' + str(self) + '>'

def Cnst(c):
	"""
	Creates a constant term2 from the Python object c. Supported types of c:
	int, float, fractions.Fraction and fractions.Decimal.
	Raises a TypeError in case the type of c does not match one of these.
	Raises a ValueError in case the representation of c could not be parsed.
	This latter case should never happen.
	"""
	# sys.set_int_max_str_digits()
	return libsmlp._mk_cnst(Q(c))

def And(*args):
	"""
	Given any number of form2 formulas such as list or tuple, returns a new
	form2 formula that corresponds to the conjunction of all given formulas.
	The result of a call with an empty 'args' list is the constant true
	formula.
	"""
	return libsmlp._mk_and(args)

def Or(*args):
	"""
	Given any number of form2 formulas such as list or tuple, returns a new
	form2 formula that corresponds to the conjunction of all given formulas.
	The result of a call with an empty 'args' list is the constant false
	formula.
	"""
	return libsmlp._mk_or(args)

def free_vars(e):
	return set(libsmlp._free_vars(e))

def options(opts : dict = None):
	"""
	Set and query options. Supported options:

	- ext_solver_cmd, same as cmd-line param -S, type str
	- inc_solver_cmd, same as cmd-line param -I, type str
	- intervals, same as cmd-line param -i, type int
	- log_color, similar to cmd-line param -c, type int
	- alg_dec_prec_approx, type int
	"""
	return libsmlp._options(opts)

def solver(incremental : bool, smt2_logic_str : str = None):
	return libsmlp._mk_solver(incremental, smt2_logic_str)

def cnst_fold(t, subst : dict = {}):
	return libsmlp._cnst_fold(t, subst)

def component(ty : libsmlp.type, *, interval=None, grid=None):
	assert interval is None or grid is None
	if interval is not None:
		if isinstance(interval, dict):
			lo = interval['min']
			hi = interval['max']
		else:
			lo, hi = interval
		return libsmlp._mk_component_ival(ty, Q(lo), Q(hi))
	if grid is not None:
		return libsmlp._mk_component_list(ty, [Q(v) for v in grid])
	return libsmlp._mk_component_entire(ty)

Q.__repr__ = lambda self: '<' + self.__module__ + '.Q ' + str(self) + '>'
Q.__hash__ = lambda self: fractions.Fraction(self.numerator, self.denominator).__hash__()

def Q(c, *args):
	"""
	Creates a rational constant from the Python object c. Supported types of c:
	int, float, fractions.Fraction, fractions.Decimal and libsmlp.Q.
	Raises a TypeError in case the type of c does not match one of these.
	Raises a ValueError in case the representation of c could not be parsed.
	This latter case should never happen.
	"""
	assert len(args) <= 1
	if len(args) == 1:
		return Q(fractions.Fraction(c, args[0]))
	# sys.set_int_max_str_digits()
	if isinstance(c, int):
		r = libsmlp._mk_Q_Z(repr(c))
	elif isinstance(c, float):
		r = libsmlp._mk_Q_F(c)
	elif isinstance(c, fractions.Fraction):
		r = libsmlp._mk_Q_Q(repr(c.numerator), repr(c.denominator))
	elif isinstance(c, fractions.Decimal):
		return Q(fractions.Fraction(c))
	elif isinstance(c, libsmlp.Q):
		return c
	else:
		raise TypeError("unsupported type of c in Q(c): " + repr(type(c)))
	if not r:
		raise ValueError("cannot interpret " + repr(c) + " as a rational constant")
	return r.value

def domain(components : dict):
	return libsmlp._mk_domain(components)
