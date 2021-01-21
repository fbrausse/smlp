
from .solverpool import Smtlib2

from functools import partial as bind

import time, logging

def par_dag(a, dag):
	for i,b in enumerate(a.children):
		# no need to replace constants
		if len(b.children) == 0:
			continue
		dag.setdefault(id(b), []).append((i,a))
		par_dag(b, dag)

def daggify(a):
	dag = {}
	par_dag(a, dag)
	#print(repr(a), '->', dag, file=sys.stderr)

	class Let:
		def __init__(self, sym, tm, a):
			self.sym = sym
			self.tm = tm
			self.a = a

		def __repr__(self):
			return '(let ((%s %s)) %s)' % (self.sym, self.tm, self.a)

	# ordering important!
	n = 0
	for v in dag.values():
		if len(v) <= 1:
			continue
		i0, a0 = v[0]
		assert all(a1.children[i].ty == a0.children[i0].ty for i,a1 in v)
		letr = Term(a0.children[i0].ty, '?l%d' % n)
		n += 1
		logging.info('replacing %s with %s in %s', a0.children[i0], letr, v)
		a = Let(letr.sym, a0.children[i0], a)
		for i,a1 in v:
			a1.children[i] = letr

	return a

def QF_NRA(cnst_decls, cnst_defs, asserts, need_model, timeout=None):
	a = time.perf_counter()
	r = Smtlib2('QF_NRA', cnst_decls, cnst_defs,
	            [repr(daggify(a)) for a in asserts], need_model,
	            timeout=timeout)
	logging.info('QF_NRA took %s sec', time.perf_counter() - a)
	return r


def binary(ty, sym, left, right):
	assert type(left.ty) == type(right.ty)
	return term(ty, sym, left, right)

def numeric(ty, sym, *children):
	assert ty in (Int, Real)
	assert all(c.ty == ty for c in children)
	return term(ty, sym, *children)

def term(ty, sym, *children):
	assert all(type(c) == Term for c in children)
	t = Term(ty, sym, *children)

#	if ty == Type.BOOL:
#		t.__and__ = lambda o: binary(Type.BOOL, 'and', t, o)
#		t.__or__  = lambda o: binary(Type.BOOL, 'or', t, o)
#		t.__xor__ = lambda o: binary(Type.BOOL, 'xor', t, o)
#	if ty in (Type.INT, Type.REAL):
#		t.__gt__  = lambda o: binary(Type.BOOL, '>', t, o)
#		t.__lt__  = lambda o: binary(Type.BOOL, '<', t, o)
#		t.__le__  = lambda o: binary(Type.BOOL, '<=', t, o)
#		t.__ge__  = lambda o: binary(Type.BOOL, '>=', t, o)
#		t.__add__ = lambda o: numeric(ty, '+', t, o)
#		t.__sub__ = lambda o: numeric(ty, '-', t, o)
#		t.__mul__ = lambda o: numeric(ty, '*', t, o)
#	if ty == Type.REAL:
#		t.__truediv__ = lambda o: numeric(ty, '/', t, o)

	return t

class Boolean:
	def eq(o): return bind(binary, Boolean, '=') if o == Boolean else NotImplemented
	def _and(o): return bind(binary, Boolean, 'and') if o == Boolean else NotImplemented

class Real:
	def gt(o) : return bind(binary, Boolean, '>') if o == Real else NotImplemented
	def add(o): return bind(numeric, Real, '+') if o == Real else NotImplemented
	def sub(o): return bind(numeric, Real, '-') if o == Real else NotImplemented
	def mul(o): return bind(numeric, Real, '*') if o == Real else NotImplemented

class Int:
	def gt(o) : return bind(binary, Boolean, '>') if o == Int else NotImplemented
	def add(o): return bind(numeric, Int, '+') if o == Int else NotImplemented
	def sub(o): return bind(numeric, Int, '-') if o == Int else NotImplemented
	def mul(o): return bind(numeric, Int, '*') if o == Int else NotImplemented

class Term:
	def __init__(self, ty, sym : str, *children):
		self.ty = ty
		self.sym = sym
		self.children = list(children)

	def __repr__(self):
		return ('(%s%s)' % (self.sym, ''.join(' %s' % c for c in self.children))
		        if len(self.children) > 0 else self.sym)

	def __bool__(self): return NotImplemented

	#def __eq__      (self, o): return NotImplemented
	#def __ne__      (self, o): return binary(Type.BOOL, 'distinct', self, o)

	def __and__     (self, o): return self.ty._and(o.ty)(self, o)
	#def __or__      (self, o): return NotImplemented
	#def __xor__     (self, o): return NotImplemented
	def __gt__      (self, o): return self.ty.gt(o.ty)(self, o)
	#def __lt__      (self, o): return NotImplemented
	#def __le__      (self, o): return NotImplemented
	#def __ge__      (self, o): return NotImplemented
	def __add__     (self, o): return self.ty.add(o.ty)(self, o)
	def __sub__     (self, o): return self.ty.sub(o.ty)(self, o)
	def __mul__     (self, o): return self.ty.mul(o.ty)(self, o)
	def __truediv__ (self, o): return self.ty.div(o.ty)(self, o)

__all__ = [s.__name__ for s in (Term,Int,Real,Boolean,QF_NRA,term,numeric,binary)]
