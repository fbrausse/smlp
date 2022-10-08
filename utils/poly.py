#!/usr/bin/env python3

Z3 = True

import sys
from copy import copy
from enum import Enum

if Z3:
	import z3

class Result(Enum):
	SAT = 'sat'
	UNSAT = 'unsat'
	UNKNOWN = 'unknown'

	def __str__(self):
		return self.value

SAT = Result.SAT
UNSAT = Result.UNSAT
UNKNOWN = Result.UNKNOWN

def exit_code(result):
	if result == SAT:
		return 10
	if result == UNSAT:
		return 20
	if result == UNKNOWN:
		return 0
	raise NotImplementedError

# non-empty closed intervals
class Interval:
	def __init__(self, lo, up):
		assert lo.__class__ is up.__class__
		assert lo <= up
		self.lo = lo
		self.up = up

	def __add__(self, other):
		if isinstance(other, Interval):
			return Interval(self.lo + other.lo, self.up + other.up)
		else:
			return Interval(self.lo + other, self.up + other)

	def __radd__(self, other):
		return self.__add__(other)

	def __pos__(self):
		return self

	def __neg__(self):
		return Interval(-self.up, -self.lo)

	def __sub__(self, other):
		return self + (-other)

	def __rsub__(self, other):
		return other + (-self)

	def contains(self, v):
		assert not isinstance(v, Interval)
		return self.lo <= v <= self.up

	def __mul__(self, other):
		a = self
		b = other
		if b is a and a.contains(0):
			return Interval(0, max(a.lo * a.lo, a.up * a.up))
		elif isinstance(b, Interval):
			ll = a.lo * b.lo
			lu = a.lo * b.up
			ul = a.up * b.lo
			uu = a.up * b.up
			return Interval(min(ll, lu, ul, uu), max(ll, lu, ul, uu))
		elif b < 0:
			return Interval(a.up * b, a.lo * b)
		elif b == 0:
			return Interval(0, 0)
		else:
			return Interval(a.lo * b, a.up * b)

	def __rmul__(self, other):
		return self.__mul__(other)

	def __lt__(self, other):
		a = self
		b = other
		lo = False
		up = True
		if isinstance(b, Interval):
			if a.up < b.lo:
				lo = True
			elif a.lo > b.up:
				up = False
		else:
			if a.up < b:
				lo = True
			elif a.lo > b:
				up = False
		return Interval(lo, up)

	def __gt__(self, other):
		return ~(self <= other)

	def __le__(self, other):
		a = self
		b = other
		lo = False
		up = True
		if isinstance(b, Interval):
			if a.up <= b.lo:
				lo = True
			elif a.lo > b.up:
				up = False
		else:
			if a.up <= b:
				lo = True
			elif a.lo > b:
				up = False
		return Interval(lo, up)

	def __ge__(self, other):
		return ~(self < other)

	def __eq__(self, other):
		raise NotImplementedError

	def __ne__(self, other):
		raise NotImplementedError

	def __invert__(self): # unary operator ~
		if isinstance(self.lo, bool):
			return Interval(not self.up, not self.lo)
		raise NotImplementedError

	def __bool__(self):
		if isinstance(self.lo, bool) and self.lo is self.up:
			return self.lo
		raise NotImplementedError

	def __len__(self):
		return self.up - self.lo

	def __str__(self):
		return str([self.lo, self.up])

	def __repr__(self):
		return 'Interval(%s, %s)' % (self.lo, self.up)

def union(*ivals):
	assert len(ivals) > 0 # Interval is non-empty
	a = copy(ivals[0])
	for b in ivals[1:]:
		if a.lo > b.lo:
			a.lo = b.lo
		if a.up < b.up:
			a.up = b.up
	return a

def as_result(r):
	if Z3 and isinstance(r, z3.CheckSatResult):
		return Result(str(r))
	if isinstance(r, Interval) and isinstance(r.lo, bool):
		if r.lo is True:
			return SAT
		if r.up is False:
			return UNSAT
		return UNKNOWN
	raise NotImplementedError

def _log(*args):
	print(*args, file=sys.stderr)

if Z3:
	def build_match(var, otherwise, args):
		if len(args) == 0:
			return 0 if otherwise is None else otherwise
		o = build_match(var, otherwise, args[2:])
		#_log(type(var), type(args[0]), type(o))
		return z3.If(var == args[0], args[1], o)
else:
	def build_match(var, otherwise, args):
		for i in range(0, len(args) // 2):
			if var == args[2*i]:
				return args[2*i+1]
		return otherwise

def Match(var, *args):
	assert len(args) % 2 == 1
	assert args[-1] is None
	return build_match(var, args[-1], args[:-1])

def _prep(contents):
	return (contents.replace(' ', '')
	                .replace('\n', '')
	                .replace(':', '_')
	                .replace(',.)', ',None)'))

def parse_dom_file(f):
	domain = {}
	for line in f:
		line = line.strip()
		if not len(line):
			continue
		toks = line.split(maxsplit=2)
		assert len(toks) == 3, (toks, line)
		bnds = eval(compile(toks[2], '<string>', 'eval'), {}, {})
		var = _prep(toks[0])
		rng = Interval(*bnds)
		domain[var] = rng
	return domain

def parse_expr_file(f):
	s = _prep(f.read())
	#_log(s)
	return compile(s, '<string>', 'eval')

if __name__ == '__main__':
	with open(sys.argv[1]) as f:
		domain = parse_dom_file(f)

	with open(sys.argv[2]) as f:
		c = parse_expr_file(f)

	f = { 'Match': Match } # functions (global)
	T = 0.95

	if Z3:
		solver = z3.Solver()
		solver.set('logic', 'QF_NIRA')
		v = {}
		for k,rng in domain.items():
			var = z3.Real(k)
			v[k] = var
			solver.add(z3.And(rng.lo <= var, var <= rng.up))
		v['_post'] = z3.Real('_post')
		solver.add(z3.Or(v['_post'] == 5, v['_post'] == 8))
		Y = eval(c, f, v)
		solver.add(Y >= T)
		print(solver.to_smt2())
		res = solver.check()
	else:
		v = domain             # variables (local)
		_log('vars:', str(v))
		values = []
		for post in [5,8]:
			y = eval(c, f, v | { '_post': post })
			values.append(y)
			_log('post=%s:' % post, y)

		Y = union(*values)
		_log('Y:', Y)
		res = Y >= T

	res = as_result(res)
	_log('Y >= %s:' % T, res)
	sys.exit(exit_code(res))
