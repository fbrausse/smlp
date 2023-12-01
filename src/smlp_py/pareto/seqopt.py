
from enum import Enum
from collections import namedtuple

from z3 import And

from common import timed

class Interval:
	def __init__(self, lo=None, hi=None):
		assert lo is None or hi is None or type(lo) is type(hi)
		self.lo = lo
		self.hi = hi

	def center(self):
		r = self.radius
		assert r is not None
		return self.lo + r

	@property
	def width(self):
		if self.lo is None or self.hi is None:
			return None
		return self.hi - self.lo if self.lo <= self.hi else 0

	@property
	def radius(self):
		w = self.width
		return None if w is None else w / 2


def center(lo, hi):
	assert lo is not None
	assert hi is not None
	assert lo <= hi
	return lo + (hi - lo) / 2

def choose_in_unbounded(lo=None, hi=None, choose_in=center):
	if lo is None and hi is None:
		return 0
	else if lo is None:
		return hi - 1
	else if hi is None:
		return lo + 1
	else:
		return choose_in(lo, hi)

def dedekind_approx(in_upper, choose_in=center, lo=None, hi=None):
	while True:
		yield lo, hi
		m = choose_in_unbounded(lo, hi, choose_in=choose_in)
		if in_upper(m):
			lo = m
		else:
			hi = m

def intersect_seq(seq):
	lo = None
	hi = None
	for l, h in seq:
		if l is not None:
			lo = l if lo is None else max(l, lo)
		if h is not None:
			hi = h if hi is None else min(h, hi)
		yield lo, hi

def is_bounded(lo, hi, eps):
	if lo is None or hi is None:
		return False
	assert lo <= hi
	return hi - lo <= eps


class Seqmax:
	def __init__(self, lb, ub, objs, timeout=None):
		self.lb = lb
		self.ub = ub
		self.objs = objs
		self.timeout = timeout

	def step(self, solver, eps):
		if len(self.objs) == 0:
			return False # done
		assert self.lb is not None
		if is_bounded(self.lb, self.ub, eps):
			return True # need refinement: drop one in self.objs
		if self.timeout < 0:
			raise SolverTimeoutError(self.timeout)
		th = (self.ub - self.lb) / 2
		r, t = timed(lambda: solver.solve(And(*[o >= th for o in self.objs]),
		                                  timeout=self.timeout))
		if self.timeout is not None:
			self.timeout -= t
		if r is None or not r:
			self.ub = th
		else:
			self.lb = th
		return self.step(solver, eps)

def left_path(objs):
	return objs[1:]

def seqmax(lb, ub, objs, solver, eps, choose_path = left_path):
	sm = Seqmax(lb, ub, objs)
	try:
		while sm.step(solver, eps):
			# drop one in self.objs
			assert sm.lb >= lb
			sm.objs = choose_path(sm.objs)
			sm.ub = ub
	except SolverTimeoutError as e:
		pass
	
