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

__version__ = libsmlp._version()

def Cnst(c) -> term2:
	"""
	Either creates a constant term2 or form2 from the Python object c, or
	destructs a constant term2 or form2 into its value.

	For creating term2, supported types of c are:
	int, float, fractions.Fraction, fractions.Decimal and libsmlp.Q.
	Raises a TypeError in case the type of c does not match one of these.
	Raises a ValueError in case the representation of c could not be parsed.
	This latter case should never happen.

	For creating form2, supported values of c are:
	True and False. These cases are equivalent to calling And() and Or()
	with an empty argument list, respectively.
	"""
	# sys.set_int_max_str_digits()
	if isinstance(c, term2) or isinstance(c, form2):
		return libsmlp._dt_cnst(c)
	if c is True:
		return And()
	if c is False:
		return Or()
	return libsmlp._mk_cnst(Q(c))

def And(*args) -> form2:
	"""
	Given any number of form2 formulas such as list or tuple, returns a new
	form2 formula that corresponds to the conjunction of all given formulas.
	The result of a call with an empty 'args' list is the constant true
	formula.
	"""
	return libsmlp._mk_and(args)

def Or(*args) -> form2:
	"""
	Given any number of form2 formulas such as list or tuple, returns a new
	form2 formula that corresponds to the conjunction of all given formulas.
	The result of a call with an empty 'args' list is the constant false
	formula.
	"""
	return libsmlp._mk_or(args)

def destruct(e : form2|term2) -> dict:
	"""
	Destructure the given term2 or form2 instance `e`. The result is a dict
	with the following entries:

	- 'id': always, one of:
	  - term2: 'var', 'add', 'sub', 'mul', 'uadd', 'usub', 'const', 'ite'
	  - form2: 'prop', 'and', 'or', 'not'

	- 'args': operands to this operation as a tuple (or list in case of
	          'and' and 'or') of term2 and/or form2 objects (all except
	          'var', 'const'),

	- 'name': name of symbol ('var' only)

	- 'type': type of term2 constant, one of: 'Z', 'Q', 'A' ('const' only)

	- 'cmp': comparison predicate, one of: '<=', '<', '>=', '>', '==', '!='
	         ('prop' only)
	"""
	return libsmlp._dt(e)

def free_vars(e) -> set:
	return set(libsmlp._free_vars(e))

def options(opts : dict = None) -> dict:
	"""
	Set and query options. Supported options:

	- ext_solver_cmd, same as cmd-line param -S, type str
	- inc_solver_cmd, same as cmd-line param -I, type str
	- intervals, same as cmd-line param -i, type int
	- log_color, similar to cmd-line param -c, type int
	- alg_dec_prec_approx, type int
	"""
	return libsmlp._options(opts)

def solver(incremental : bool, smt2_logic_str : str = None) -> solver:
	return libsmlp._mk_solver(incremental, smt2_logic_str)

def cnst_fold(t, subst : dict = {}):
	return libsmlp._cnst_fold(t, subst)

def component(ty : libsmlp.type, *, interval=None, grid=None) -> component:
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

del domain.append
del domain.extend

def domain(comps : dict) -> domain:
	return libsmlp._mk_domain(comps)

Q.__repr__ = lambda self: '<' + self.__module__ + '.Q ' + str(self) + '>'
Q.__hash__ = lambda self: fractions.Fraction(self.numerator, self.denominator).__hash__()

def Q(c, *args) -> Q:
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
	if r is None:
		raise ValueError("cannot interpret " + repr(c) + " as a rational constant")
	return r

def _R_approx(self, precision : int):
	assert isinstance(precision, int)
	r = self._approx(precision)
	if r is None:
		raise ValueError('cannot approximate smlp.R to precision ' + str(precision) + ' of type ' + type(precision))
	return r
libsmlp.R.approx = _R_approx

def approx(value, /, *, type=float, precision=None):
	"""
	Approximates 'value'. 'value' must be one of: smlp.Q, smlp.A, smlp.R.
	The following parameters can optionally be specified:
	- precision: request a specific precision; required when type != float;
	  default: None
	- type: type of the result, defaults to 'float'; must be one of:
	  float, smlp.Q, smlp.R

	The default, when 'type=float', is to approximate 'value' as a floating
	point number. This can have unexpected results when the quantity cannot
	be represented in the range Python's 'float' supports, e.g. a non-zero
	'value' could be approximated by 0.0 or by float("inf"), depending on
	its magnitude.

	Note that the 'precision' parameter is ignored for type=float.

	When type != float, the precision parameter is required to be set to an
	integer. In this case approx() returns an approximation to 'value'
	which is accurate up to absolute error 2 ** precision. For instance,
	approx(value, precision=-5) will return a result such that |value -
	approx(value, precision=-5)| <= 2 ** -5 = 1/32 holds.
	"""
	assert type is float or precision is not None
	if isinstance(value, libsmlp.A):
		value = value.to_Q() if value.known_Q() else value.to_R()
	assert isinstance(value, libsmlp.Q) or isinstance(value, libsmlp.R)

	if type is float:
		def Q2F(v):
			return float(fractions.Fraction(v.numerator,
			                                v.denominator))

		if isinstance(value, libsmlp.Q):
			return Q2F(value)
		n = -3
		zero = Q(0)
		one = Q(1)
		precision = None
		while True:
			if n < -1074:
				return 0.0
			v = value.approx(n)
			d = abs(v) - Q(1, 2 ** -n)
			if d >= one:
				precision = -54
				break
			df = Q2F(d)
			if d > zero:
				precision = _lbound_log2(value) - 54
				break
			n *= 2
		return Q2F(value.approx(precision))
	assert isinstance(precision, int)
	if isinstance(value, libsmlp.Q):
		return value
	return value.approx(precision)

libsmlp.component.__repr__ = lambda self: (
	'<' + self.__module__ + '.component of type ' + repr(self.type) +
	'>'
)
libsmlp._domain_entry.__iter__ = lambda self: (self.name, self.comp).__iter__()

def parse_poly(domain_path : str, expr_path : str, *,
               python_compat : bool = False,
               dump_pe : bool = False,
               infix : bool = True) -> pre_problem:
	return libsmlp._parse_poly(domain_path, expr_path, python_compat,
	                           dump_pe, infix)

def parse_nn(gen_path : str, hdf5_path : str, spec_path : str,
             io_bounds_path : str, *,
             obj_bounds_path : str = None,
             clamp_inputs : bool = False,
             single_obj : bool = False) -> pre_problem:
	return libsmlp._parse_nn(gen_path, hdf5_path, spec_path, io_bounds_path,
	                         obj_bounds_path, clamp_inputs, single_obj)

