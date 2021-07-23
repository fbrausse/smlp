#!/usr/bin/python
#
# This file is part of smlprover.
#
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

import asyncio, sys
from subprocess import PIPE

from z3 import *

async def _run2(which, args):
	#print('running',args)
	try:
		proc = await asyncio.create_subprocess_exec(*args.split(' '), stdin=PIPE,
		                                            stdout=PIPE, stderr=PIPE)
	except FileNotFoundError as e:
		print('error executing "%s":' % which, e, file=sys.stderr)
		proc = None
	return which, proc

async def _run(proc, interp_model):
	out, err = await proc.communicate()
	r, m = out.decode('utf-8').split('\n', maxsplit=1)
	#print('out:',out)
	#print('err:',err)
	#print('code',proc.returncode)
	if proc.returncode != 0:
		return unknown, 'return code: %d' % proc.returncode
	if err:
		return unknown, 'error: ' + err.decode('utf-8')
	if r == 'sat':
		assert proc.returncode == 0
		return sat, interp_model(m)
	elif r == 'unsat':
		return unsat, None
	elif r == 'unknown':
		return unknown, None
	else:
		print('process', p, "returned unusable output '%s'" % r,
			  "and stderr '%s'" % err.decode('utf-8'))
		assert False
	#return out, err, proc.returncode, which, interp_model

def _run_until_1st(cmds):
	loop = asyncio.get_event_loop()
	tasks = [asyncio.ensure_future(c) for k,c,p in cmds]
	pending = list(tasks)
	done_ok = []
	done_fail = []
	while len(done_ok) == 0 and len(pending) > 0:
		now_done, pending = loop.run_until_complete(asyncio.wait(pending,
		                                                         return_when=asyncio.FIRST_COMPLETED))
		for t in now_done:
			ch, m = t.result()
			if ch == unknown:
				done_fail.append(t)
			else:
				done_ok.append(t)

	res = None
	for (k,c,p),t in zip(cmds, tasks):
		if t in pending:
			t.cancel()
			try:
				p.terminate()
			except ProcessLookupError as e:
				print('%s:' % k, e)
			#print('terminated',p,'->',p.returncode)
		elif t in done_ok:
			if res is None:
				res = k, t.result()
			elif res[1] != t.result():
				print('error:', res[0], 'returned', res[1], 'while', k, 'returned', t.result(),
				      file=sys.stderr)
		else:
			assert t in done_fail
	#assert res is not None
	return res


class Cmd:
	def __init__(self, command, interp_model, preamble=''):
		self._command = command
		self._preamble = preamble
		self._interp_model = interp_model

	def spawn(self, which):
		return _run2(which, self._command)

	def fmt_in(self, proc, inst):
		proc.stdin.write((self._preamble + inst.sexpr() +
		                  '(check-sat)(get-model)'
		                 ).encode('utf-8'))
		proc.stdin.close()

	def solve(self, proc):
		return _run(proc, self._interp_model)

class SMTSolver(Cmd):
	def __init__(self, command, interp_model, logic=None, seed=None):
		super().__init__(command, lambda m: interp_model(m, lambda s: smt2_interp_numeral(s, False)[0]),
		                 (('(set-logic ' + logic + ')' if logic is not None else '') +
		                  ('(set-option :random-seed %s)' % seed if seed is not None else '')))

class Z3Py:
	def __init__(self):
		self._solver = Solver()
		self._which = None
		self._vars = None

	async def spawn(self, which):
		self._solver.reset()
		self._which = which
		self._vars = None
		class Shim:
			def terminate(self):
				pass
		return which, Shim()

	def fmt_in(self, proc, inst):
		self._solver.add(inst.z3ast())
		self._vars = inst.vars()

	def solve(self, proc, loop=None):
		def z3py(s):
			ch = s.check()
			if ch == sat:
				m = s.model()
				m = { k: m[v] for k,v in self._vars.items() }
			elif ch == unknown:
				m = s.reason_unknown()
			else:
				m = None
			return ch, m
		if loop is None:
			loop = asyncio.get_event_loop()
		return loop.run_in_executor(None, z3py, self._solver)


# inst.sexpr() -> str, inst.vars() -> dict, inst.z3ast() -> Z3.AstVector
def run_solvers(inst, solvers : dict):
	procs = {}
	#print('solvers:', solvers)
	#print({ k: v for k,v in solvers.items() })
	loop = asyncio.get_event_loop()
	for p in loop.run_until_complete(asyncio.wait([v.spawn(k) for k,v in solvers.items()]))[0]:
		k, proc = p.result()
		if proc is not None:
			solvers[k].fmt_in(proc, inst)
			procs[k] = proc

	#print(procs)

	res = _run_until_1st([(k, solvers[k].solve(p), p) for k,p in procs.items()])
	if res is None:
		return None
	#print(res)
	which, (ch, m) = res
	if ch == sat:
		variables = inst.vars()
		for k,s in m.items():
			assert m[k].is_int() if variables[k].is_int() else m[k].is_real()
			#IntVal if variables[k].is_int() else RealVal
	#print(m)
	return which, ch, m

def smt2_interp_numeral(s, neg=False):
	#print('interp "%s"' % s)
	if s[0] == '(':
		t = s[1:]
		if t.startswith('- '):
			return smt2_interp_numeral(t[2:], not neg)
		else:
			assert t.startswith('/ ')
			v, t = smt2_interp_numeral(t[2:], neg)
			if t[0] == ')':
				t = t[1:]
			assert t[0] == ' '
			w, t = smt2_interp_numeral(t[1:], False)
			#print('v:', type(v), 'w:', type(w))
			return Q(v.as_string(), w.as_string()), t
	else:
		has_dot = False
		for i in range(len(s)+1):
			if i == len(s):
				break
			if s[i] == '.':
				has_dot
				continue
			if s[i] not in '0123456789':
				break
		#print('fst:',s[:i],'snd:',s[i:],'i:',i)
		return (RealVal if has_dot else IntVal)(('-' + s[:i]) if neg else s[:i]), s[i:]

def _smt2_interp_model(m, interp_numeral):
	# output is of the form '(model\n(define-fun x () Real (/ (- 1) 2))\n(define-fun i () Int (- 1))\n)\n'
	#print('smt2 interp model:', m, file=sys.stderr)
	r={}
	for i,a in enumerate(m[:-2].split('(define-fun ')):
		if i == 0:
			continue
		a, b = a.replace('\n', ' ').replace('\t', ' ').strip().split(' ', maxsplit=1)
		assert b.startswith('() ')
		b = b[3:]
		assert b.startswith('Int') or b.startswith('Real')
		b = (b[3:] if b.startswith('Int') else b[4:]).strip()
		#print('a:',a,'b:',b[:-1], file=sys.stderr)
		#print('a: "%s", b: "%s"' % (a,b))
		r[a] = interp_numeral(b[:-1])
	return r

def _yices_interp_model(m, interp_numeral):
	# output is of the form '(= x (/ 1 2))\n(= i 1)\n'
	r={}
	#print('m: "%s"' % m, 'type', type(m))
	for a in m.split('(= '):
		if not len(a):
			continue
		#print('a',a)
		x, v = a.split(' ', maxsplit=1)
		r[x] = interp_numeral(v[:-2]) # remove ')\n'
	return r

def vars_in_inst(I):
	# TODO: use an AstMap to avoid exponential traversals
	variables = {}
	def collect(t):
		for c in t.children():
			collect(c)
		if is_const(t) and not is_int_value(t) and not is_rational_value(t):
			variables[t.decl().name()] = t
	for t in I:
		collect(t)
	#print('variables:',variables)
	return variables

SOLVERS = {
	'z3py': Z3Py(), #_z3py_run,
	'z3': SMTSolver('z3 -smt2 -in', _smt2_interp_model, 'QF_LIRA'),
	'cvc4': SMTSolver('cvc4 -L smt2 -m --rewrite-divk', _smt2_interp_model, 'QF_LIRA'),
	'yices': SMTSolver('yices-smt2', _yices_interp_model, 'QF_LIRA'),
	'ksmt': SMTSolver('ksmt', _smt2_interp_model),
}

class InstStrWrapper:
	def __init__(self, **kwargs):
		self._inst = kwargs.get('inst')
		self._I = kwargs.get('I')
		self._vars = kwargs.get('vars')
		if self._inst is None and self._I is None:
			raise ValueError('param "inst" or "I" must be given')

	def sexpr(self):
		if self._inst is None:
			self._inst = self._I.sexpr()
		return self._inst

	def z3ast(self):
		if self._I is None:
			self._I = parse_smt2_string(self._inst)
		return self._I

	def vars(self):
		if self._vars is None:
			self._vars = vars_in_inst(self.z3ast())
		return self._vars

if __name__ == '__main__':
	try:
		with open(sys.argv[1], 'r') as f:
			inst = f.read()
	except FileNotFoundError:
		inst = """
		(declare-fun x () Real)
		(declare-fun i () Int)
		(assert (distinct x 0.0))
		(assert (= (* x 2) (to_real i)))
		"""

	solvers = { k: v for k,v in SOLVERS.items() if k in sys.argv[2:] } if len(sys.argv) > 2 else SOLVERS

	#s = Solver()
	#s.add(I)
	#print(run_solvers(inst, vars_in_inst(s.assertions()), solvers))
	print(run_solvers(InstStrWrapper(inst=inst), solvers))
