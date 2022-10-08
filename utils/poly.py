#!/usr/bin/env python3

import sys, copy, z3

Z3 = True

# non-empty closed intervals
class Interval:
	def __init__(self, lo, up):
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

	def __str__(self):
		return str([self.lo, self.up])

	def __repr__(self):
		return 'Interval(%s, %s)' % (self.lo, self.up)

def union(*ivals):
	assert len(ivals) > 0 # Interval is non-empty
	a = copy.copy(ivals[0])
	for b in ivals[1:]:
		if a.lo > b.lo:
			a.lo = b.lo
		if a.up < b.up:
			a.up = b.up
	return a

if Z3:
	def build_match(var, otherwise, args):
		if len(args) == 0:
			return 0 if otherwise is None else otherwise
		o = build_match(var, otherwise, args[2:])
		#print(type(var), type(args[0]), type(o))
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

def prep(contents):
	return (contents.replace(' ', '')
	                .replace('\n', '')
	                .replace(':', '_')
	                .replace(',.)', ',None)'))

def parse_dom(line):
	toks = line.split(maxsplit=2)
	assert len(toks) == 3, (toks, line)
	ival = toks[2]
	return (prep(toks[0]), Interval(*eval(compile(ival, '<string>', 'eval'), {}, {})))

if __name__ == '__main__':
	domain = {}
	with open(sys.argv[1]) as f:
		for line in f:
			line = line.strip()
			if len(line):
				var, rng = parse_dom(line)
				domain[var] = rng

	with open(sys.argv[2]) as f:
		s = prep(f.read())
		#print(s)

	c = compile(s, '<string>', 'eval')
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
		print('vars:', str(v))
		values = []
		for post in [5,8]:
			y = eval(c, f, v | { '_post': post })
			values.append(y)
			print('post=%s:' % post, y)

		Y = union(*values)
		print('union:', Y)
		res = ('sat' if Y.lo >= T else
		       'unsat' if Y.up < T else
		       'unknown')

	print('Y >= %s:' % T, res, file=sys.stderr)
