#!/usr/bin/env python3
#
# This file is part of smlprover.
#
# Copyright 2019 Konstantin Korovin
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

import datetime, argparse, csv, operator

from tensorflow.keras.models import load_model
from tensorflow import keras
from z3 import *
from numpy import random, isfinite

import pandas as pd
import numpy as np

from fractions import Fraction
from decimal import Decimal
import itertools, json

from common import *
from smlp.util import die
from checkdata import CD
from solvepar import SOLVERS, run_solvers

import skopt
from scipy.interpolate import interp1d


def parse_args(argv):
	p = argparse.ArgumentParser(prog=argv[0])
	p.add_argument('nn_model', metavar='NN_MODEL', type=str,
	               help='Path to NN model in .h5 format')
	p.add_argument('-a', '--more-constraints', type=str, default=None,
	               metavar='CONSTRAINTS',
	               help='additional constraints; Python term containing features '+
	                    'or numeric constants; incompatible with BO and -d')
	p.add_argument('-b', '--bounds', type=float, nargs='?', const=0.0,
	               help='bound variables; optional parameter is a factor to '+
	                    'increase the range (max-min) by '+
	                    '[default: none; otherwise, if BOUNDS is missing, 0]')
	p.add_argument('-B', '--data-bounds', default=None, type=str, metavar='DBOUNDS',
	               help='path to data_bounds file to amend the bounds determined from SPEC')
	p.add_argument('-C', '--check-safe', type=int, default=0,
	               help='Number of random samples to check for each SAFE config '+
	                    'found [default: 0]')
	p.add_argument('-d', '--data', type=str,
	               help='path to DATA.csv; check DATA for counter-examples to '+
	                    'found regions')
	p.add_argument('-D', '--delta', type=str,
	               help='exclude (1+DELTA)*radius region for non-grid components')
	p.add_argument('-g', '--model-gen', type=str, required=True,
	               help='the model_gen*.json file containing the training / '+
	                    'preprocessing parameters')
	p.add_argument('-G', '--grid', type=str,
	               help='Path to grid.istar file')
	#p.add_argument('-n', type=int, default=1,
	#               help='Number of counter examples to generate [default: 1]')
	p.add_argument('-n', type=int, default=1,
	               help='number of safe regions to generate in total (that is, '+
	                    'including those already in SAFE) [default: 1]')
	p.add_argument('-N', '--no-exists', default=False, action='store_true',
	               help='only check GRID, no solving of existential part')
	p.add_argument('-o', '--output', type=str, default=None,
	               help='Path to output .smt2 instance [default: none]')
	p.add_argument('-O', '--objective', type=str, default='delta',
	               help='Objective function in terms of labelled outputs '+
	                    '[default: "delta"]')
	p.add_argument('-P', '--partial-grid', type=str, help='Path to partial grid CSV')
	p.add_argument('-r', '--response-bounds', type=str, default=None,
	               help='Path to bounds.csv for response bounds to interpret T and ST in '+
	                    '[default: use DATA_BOUNDS]')
	p.add_argument('-s', '--spec', type=str, required=True,
	               help='Path to JSON spec of input features')
	p.add_argument('-S', '--safe', type=str,
	               help='Path to output found safe configurations to as CSV')
	p.add_argument('-t', '--threshold', type=str,
	               default=json.dumps([float(Decimal('0.05')*n) for n in range(0,20)]),
	               help='Threshold to restrict output feature to be larger-equal '+
	                    'than [default: search in 0.05 grid between 0 and 0.95]')
	p.add_argument('-T', '--safe_threshold', type=str, default=None,
	               help='Center threshold [default: THRESHOLD+SAFE_OFFSET]. ' +
	                    'Overrides any SAFE_OFFSET.')
	p.add_argument('-U', '--center_offset', type=str, default='0',
	               help='Center threshold offset of threshold [default: 0]')
	p.add_argument('-v', '--verbose', default=0, action='count',
	               help='Increase verbosity')
	p.add_argument('-x', '--trace-exclude', type=str, metavar='TRACE',
	               help='exclude all unsafe i* from trace file')
	p.add_argument('-X', '--trace-exclude-safe', default=False, action='store_true',
	               help='exclude also found safe i* from the trace file')
	p.add_argument('-z', '--bo-cex', default=None, type=int,
	               help='use BO_CEX >= 10 iterations of BO to find '
	                    'counter-examples [default: no]')
	p.add_argument('-Z', '--bo-cad', default=None, type=int,
	               help='use BO_CAD iterations of BO to find a candidate '
	                    'prior to falling back to Z3 [default: no]')

	args = p.parse_args(argv[1:])
	args.threshold = json.loads(args.threshold, parse_float=Decimal)
	if args.safe_threshold is None:
		ce = Decimal(args.center_offset)
		if isinstance(args.threshold, list):
			args.safe_threshold = [t + ce for t in args.threshold]
		else:
			args.safe_threshold = args.threshold + ce
	assert isinstance(args.threshold, list) == isinstance(args.safe_threshold, list)
	if isinstance(args.threshold, list):
		assert len(args.threshold) == len(args.safe_threshold)
		d = args.safe_threshold[0] - args.threshold[0]
		assert all(st - th == d for st, th in zip(args.safe_threshold, args.threshold))
		args.center_offset = d
	else:
		args.center_offset = Decimal(args.safe_threshold) - args.threshold
	return args


def to_real(e):
	assert is_arith(e)
	if is_real(e):
		return e
	elif is_int(e):
		return ToReal(e)
	else:
		raise ValueError('e is neither of sort Real nor Int')



def sequential_nn_to_terms(model, inputs, ctx=None,
	layer_funcs = {
		'Dense':
			lambda vars, weights, bias:
				sum(to_real(v) * w for v,w in zip(vars, weights)) + bias,
	},
	activations = {
		'relu'  : lambda expr, ctx=None: If(expr >= 0, expr, 0, ctx=ctx),
		'linear': lambda expr, ctx=None: expr,
	}
):
	assert isinstance(model, keras.Sequential)

	last_layer = inputs   # expression references into Z3's AST
	for layer in model.layers:
		assert isinstance(layer, keras.layers.Dense)
		layer_func = layer_funcs[type(layer).__name__]
		layer_activation = layer.get_config()["activation"]
		try:
			activation = activations[layer_activation]
		except KeyError:
			log(0, "add_model:activation:"+layer_activation+" is not supported")
			quit(code=1)
		weights, biases = layer.get_weights()
		# weights are [[w1_1,..,w1_6],[w20_1,..,w20_6]]
		last_layer = [activation(layer_func(last_layer, w, b), ctx=ctx)
		              for w,b in zip(weights.transpose(), biases.transpose())]

	return last_layer


# TODO: rounding to floats is undirected, therefore b0rked
def double_check_model(model, spec, is_safe, threshold_f, center, n, norm, rng=None):
	if rng is None:
		rng = random.default_rng()
	if is_safe:
		# run N random samples in center ball through keras prediction and check
		# whether out >= threshold
		abs_radii = get_radii(spec, center)
		log(3, 'radii for', center, ':', abs_radii)
		samples = rng.uniform([float(c-r) for c,r in zip(center, abs_radii)],
		                      [float(c+r) for c,r in zip(center, abs_radii)],
		                      (n,len(center)))
	else:
		# run just center point through keras prediction and check whether
		# out < threshold
		samples = [[float(c) for c in center]]
	if norm is not None:
		samples = [[norm[i](v) for i,v in enumerate(s)] for s in samples]
	pred = model.predict(samples)
	if any(not isfinite(s[0]) for s in pred):
		print('WARNING: some predictions of random samples in ball evaluate to NaN!', file=sys.stderr)
	return all((threshold_f(s[0])) == is_safe for s in pred if isfinite(s[0]))


def run(s, in_vars):
	which, ch, m = run_solvers(s.sexpr(), { v.decl().name(): v for v in in_vars },
	                           { (k,s) for k,s in SOLVERS if k in ('z3','cvc4','yices') })
	class MyModel:
		def __init__(self, m):
			self._m = m
		def eval_real(self):
			pass
		def __iter__(self):
			pass
		def __getitem__(self, key):
			pass
	return ch, MyModel(m)


def in_interval(v,a,b):
	return And(*[Fraction(a) <= v, v < Fraction(b)])

def in_threshold_area(v,t):
	if isinstance(t, str):
		return Or(*[in_interval(v, *s.split(':')) if ':' in s else Fraction(s) <= v
		            for s in t.split(',')])
	return t <= v




trace_out = csv.writer(sys.stdout, dialect='unix', quoting=csv.QUOTE_MINIMAL)

class Command:
	def __init__(self, label, *args):
		assert isinstance(label, str)
		self.label = label
		self.args = args

	def __call__(self, solver, *args):
		self.result = self.run(solver, *args)
		trace_out.writerow(self.fmt(self.result))
		return self.result

	def fmt(self, result):
		if result is None:
			return [ self.label, *self.args ]
		return [ self.label, result, *self.args ]


class BoundInput(Command):
	# which is either 'min' or 'max'
	def __init__(self, in_vars, spec, bounds, which):
		sub_args = []
		for v,s in zip(in_vars, spec):
			if s['label'] in bounds:
				b = bounds[s['label']]
				sub_args.append(b[which] if which in b else None)
			else:
				sub_args.append(None)
		super().__init__(which, *sub_args)
		self._in_vars = in_vars
		self._spec = spec

	def run(self, solver):
		for v,s,b in zip(self._in_vars, self._spec, self.args):
			if b is not None:
				if self.label == 'min':
					log(2, 'bounded:', b, " <= '%s'" % s['label'])
					solver.add(b <= v)
				else:
					assert self.label == 'max'
					log(2, 'bounded:', "'%s' <=" % s['label'], b)
					solver.add(v <= b)
		return None


class Cat(Command):
	LABEL = 'c'
	def __init__(self, in_vars, cati, catv):
		super().__init__(Cat.LABEL, *catv)
		self._in_vars = in_vars
		self._cati = cati

	def run(self, solver):
		log(1, '--------', self.args, '--------')
		solver.add([self._in_vars[i] == w for i,w in zip(self._cati, self.args)])
		return None


class Solve:
	def __init__(self, output, obj_term, in_vars):
		self._output = output
		self._obj_term = obj_term
		self._in_vars = in_vars

	def solve(self, solver):
		if self._output is not None:
			with open(self._output, 'w') as f:
				f.write(solver.sexpr())
		res, self.t = timed(solver.check)
		log(3, 'Z3 stats:', solver.statistics())
		if res == unknown:
			log(1, 'unknown because', solver.reason_unknown()) # Ctrl-C -> 'cancelled'
			snd = solver.reason_unknown()
		elif res == sat:
			snd = solver.model()
		else:
			snd = None
		return res, snd

	def fmt(self, result):
		s, m = result
		row = [ self.label, s, *self.args, str(self.t) ]
		if s == sat:
			row += [ m[v] for v in self._in_vars ]
			row.append(m.eval(self._obj_term))
		return row


class Grid(Command):
	LABEL = 'g'
	def __init__(self, in_vars, spec):
		super().__init__(Grid.LABEL)
		self._in_vars = in_vars
		self._spec = spec

	def run(self, solver):
		for s,v in zip(self._spec, self._in_vars):
			if 'safe' in s and s['type'] != 'input':
				solver.add(Or([v == w for w in s['safe']]))
		return None


def cnst_ctor(spec_entry):
	if spec_entry['range'] == 'int':
		return IntVal
	elif spec_entry['range'] == 'float':
		return RealVal
	elif all(type(v) == int for v in spec_entry['range']):
		return IntVal
	else:
		return RealVal


class List(Command):
	LABEL = 'l'
	def __init__(self, in_vars, spec, lst):
		super().__init__(List.LABEL)
		self._in_vars = in_vars
		self._spec = spec
		self._list = lst

	def run(self, solver):
		solver.add(Or([
			And([((v == cnst_ctor(s)(row[1][s['label']])) if s['label'] in row[1] else True)
			     for s,v in zip(self._spec, self._in_vars)])
			for row in self._list.iterrows()
		]))


class SafePoint(Solve, Command):
	LABEL = 'a'
	def __init__(self, in_vars, spec, obj_term, safe_threshold, output):
		Command.__init__(self, SafePoint.LABEL, safe_threshold)
		Solve.__init__(self, (output + '-' + SafePoint.LABEL if output is not None else None),
		               obj_term, in_vars)
		self._in_vars = in_vars
		self._obj_term = obj_term
		self._spec = spec

	def run(self, solver, candidate_idx):
		st = self.args[0]
		log(1, 'solving >=', st, '...')
		solver.add(in_threshold_area(self._obj_term, st))
		out = self.solve(solver)
		log(1, 'candidate', candidate_idx, '->', out[0], 'in', self.t, 'sec')
		return out

def contains(bounds, x):
	assert(len(bounds) == len(x))
	return all(b['min'] <= xi and xi <= b['max'] for b,xi in zip(bounds,x))

class Rad(Command):
	LABEL = 'r'
	def __init__(self, in_vars, spec, i_star, delta=None):
		def py2z3(v):
			if type(v) == int:
				return IntVal(v)
			else:
				assert type(v) == Fraction
				return RealVal(v)

		if delta is None:
			delta = 0
		else:
			delta = float(delta)

		def get_lo(s):
			return operator.le

		def get_hi(s):
			return operator.lt if s['range'] == 'int' else operator.le

		def rel_err(v,c,e,s): # only for c != 0
			d = to_real(v - c) / to_real(c)
			return And(get_lo(s)(-e, d), get_hi(s)(d, e))

		def abs_err(v,c,e,s):
			if e == 0:
				return v == c
			d = (v - c)
			return And(get_lo(s)(-e, d), get_hi(s)(d, e))

		self._c = []
		sub_args = []
		self.bnds = {} # strictness is lost
		for s,v in zip(spec, in_vars):
			if s['type'] == 'categorical' or s['type'] == 'input':
				continue
			try:
				def is_zero(val):
					if val.is_int():
						return val.as_long() == 0
					assert val.is_real()
					return val.numerator_as_long() == 0
				if 'rad-rel' in s and not is_zero(i_star[v]):
					e = s['rad-rel'] * (1+delta)
					t, c, rng = ('rel', rel_err(v, i_star[v], e, s), '%g%%' % (e*100))
					a = i_star.eval((1-e) * to_real(i_star[v])).as_fraction()
					b = i_star.eval((1+e) * to_real(i_star[v])).as_fraction()
					self.bnds[s['label']] = {
						'min': min(a, b),
						'max': max(a, b),
					}
				else:
					try:
						t, e = 'abs', s['rad-abs']
					except KeyError:
						t, e = 'rel', s['rad-rel']
					ef = e * (1+delta)
					e = e * (1+to_real(cnst_ctor(s)(delta)))
					c, rng = (abs_err(v, i_star[v], e, s), '%g' % ef)
					self.bnds[s['label']] = {
						'min': i_star.eval((-e) + to_real(i_star[v])).as_fraction(),
						'max': i_star.eval((+e) + to_real(i_star[v])).as_fraction(),
					}
			except KeyError:
				raise
			sub_args.append(t + str(e))
			self._c.append((c, rng))
		super().__init__(Rad.LABEL, *sub_args)
		self._i_star = i_star
		self._in_vars = in_vars
		self._spec = spec

	def run(self, solver):
		def approx(val, prec):
			if val.is_int():
				return val
			assert val.is_real()
			return val.as_decimal(prec)
		for v, s, (c, rng) in zip(self._in_vars, self._spec, self._c):
			log(2, "restricting var '%s' to" % s['label'],
				approx(self._i_star[v], 7), '+/-', rng)
			solver.add(c)
		return None

	def constraints(self):
		for c, rng in self._c:
			yield c


class CounterExample(Solve, Command):
	LABEL = 'b'
	def __init__(self, obj_term, threshold, output, in_vars, guard):
		Command.__init__(self, CounterExample.LABEL, threshold)
		Solve.__init__(self,
		               None if output is None else output + '-' + CounterExample.LABEL,
		               obj_term, in_vars)
		self._obj_term = obj_term
		self._guard = guard

	def run(self, solver):
		threshold = self.args[0]
		log(1, 'solving safe <', threshold, 'with eps ...')
		assertion = self._obj_term < threshold
		if self._guard is not None:
			assertion = z3.Implies(self._guard, assertion)
		solver.add(assertion)
		out = self.solve(solver)
		log(1, '->', out[0], 'in', self.t, 'sec')
		return out


class Vars(Command):
	LABEL = 'v'
	def __init__(self, spec):
		super().__init__(Vars.LABEL, *[s['label'] for s in spec])

	def run(self, solver):
		pass


class Exclude(Command):
	LABEL = 'x'
	def __init__(self, spec, in_vars, i_star, radius=None):
		super().__init__(Exclude.LABEL,
		                 *[i_star[v] if s['type'] != 'input' else None
		                   for s,v in zip(spec, in_vars)
		                   if s['type'] != 'categorical'])
		self._in_vars = in_vars
		self._radius = radius

	def run(self, solver, candidate_idx):
		if self._radius is None:
			log(1, 'excluding candidate', candidate_idx)
			solver.add(Or([v != w for v,w in zip(self._in_vars, self.args)
			               if w is not None]))
		else:
			log(1, 'excluding candidate', candidate_idx,
			       'via region around counter example'
			       #[Not(c) for c in self._radius.constraints()]
			   )
			solver.add(Not(And(*[c for c in self._radius.constraints()])))


class Simple(Command):
	def __init__(self, func_name, *args):
		super().__init__(func_name, *args)

	def run(self, solver):
		return getattr(solver, self.label)(*self.args)

def toz3compat(x):
	return {
		float: float,
		int: int,
		np.int64: int,
		np.float64: float,
	}[type(x)](x)

# Required for Rad() constructor
class MockModel(dict):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		s = Solver()
		res = s.check() # required to get a model for the empty formula
		assert res == sat # no model before check
		self._m = s.model() # can evaluate constants

	def eval(self, expr):
		#log(2, 'MockModel eval %s' % expr)
		return self._m.eval(expr)

class Instance:
	def __init__(self, spec_path, model, gen, data_bounds, use_input_bounds,
	             input_bounds, resp_bounds, T_resp_bounds, data_path=None,
	             bo_cex = None, more_constraints : str = None):

		with open(spec_path, 'r') as f:
			all_spec = json.load(f, parse_float=Fraction)
			self.spec = [s for s in all_spec if s['type'] in ['categorical','knob','input']]
			del all_spec

		assert(type(self.spec) == list)

		assert(max(len(w) for w in model.layers[0].get_weights()[0].transpose()) ==
		       len(self.spec))

		self.model = model
		self.gen = gen
		self.data_bounds = data_bounds
		self.use_input_bounds = use_input_bounds
		self.input_bounds = input_bounds
		self.T_resp_bounds = T_resp_bounds
		self.resp_bounds = resp_bounds
		self.more_constraints = more_constraints

		self.counter_ex_finders = []
		if data_path is not None:
			objective = None
			if 'objective' in self.gen: # default now
				objective = self.gen['objective']
			elif len(self.gen['response']) == 1:
				objective = self.gen['response'][0]
			if objective is not None:
				cd = CD(objective, spec_path, data_path,
						[s['label'] for s in self.spec if s['type'] != 'categorical'],
						bnds=(None if self.T_resp_bounds is None else
						      [self.T_resp_bounds[r][v]
						       for r in self.gen['response']
						       for v in ('min','max')]))
				def ex_data_counter_example(asgn, threshold):
					n_fail, n_ok = cd.check([v.as_long() if s['range'] == 'int' else v.as_fraction()
					                         for s,v in zip([s for s in self.spec
					                                         if s['type'] != 'categorical'], asgn)],
					                        threshold)
					log(2,'data in ball: %d fail, %d ok' % (n_fail, n_ok))
					y = None # TODO: counter-example value
					return sat if n_fail > 0 else unsat, None, y
				self.counter_ex_finders.append(('data',
					lambda solver, i_star, obj_term, in_vars, threshold, output=None, constraints=None: # constraints get ignored
						ex_data_counter_example([
							i_star[v] for s,v in zip(self.spec, in_vars) if s['type'] != 'categorical'
						], threshold)))
			else:
				log(1, 'WARNING: > 1 responses unsupported for data-check, skipping...')

		self.excl_regions = []

		def skopt_counter_example(solver, i_star, obj_term, in_vars, threshold, output = None, constraints = None): # constraints get ignored
			r = Rad(in_vars, self.spec, i_star)
			try:
				opt = self._init_skopt(r.bnds)
				t0 = datetime.datetime.now()
				for i in range(bo_cex):
					x = opt.ask()
					y = self._skopt_eval(x)
					log(1, 'BO-cex #%d suggested x: %s -> y: %s' % (i,x,y))
					#log(2, 'BO-ce #%d types x: %s -> y: %s' % (i,[type(x[i]) for i in range(len(x))], type(y)))
					# np.int64, float and np.float64
					if y < threshold:
						trace_out.writerow(['B',sat,threshold,(datetime.datetime.now()-t0).total_seconds(),i,*x,toz3compat(y)])
						return sat, self.mock_model(in_vars, x), toz3compat(y)
					opt.tell(x, y)
				trace_out.writerow(['B',unsat,threshold,(datetime.datetime.now()-t0).total_seconds(),bo_cex])
				return unsat, None, None
			except ValueError as e:
				log(0, 'BO-cex exception, returning', unknown, e)
				trace_out.writerow(['B',unknown,threshold,str(e)])
				return unknown, e, None

		if bo_cex is not None:
			self.counter_ex_finders.append(('BO', skopt_counter_example))


		def ex_model_counter_example(solver, i_star, obj_term, in_vars, threshold, output, constraints):
			#Simple('push')(solver)
			r = Rad(in_vars, self.spec, i_star)
			r(solver)
			eps_res, eps_model = CounterExample(obj_term, threshold, output, in_vars, constraints)(solver)
			y = None
			#Simple('pop')(solver)
			if eps_res == sat:
				log(1,'Found counter-example') or log(2,'Found counter-example:',
				                                        str(eps_model).replace('\n', ''))
				y = eps_model.eval(obj_term).as_fraction()
			elif eps_res == unsat:
				log(1, 'Found no counter-example:', str(i_star).replace('\n', ''))
			return eps_res, eps_model, y

		self.counter_ex_finders.append(('NN',
			lambda solver, i_star, obj_term, in_vars, threshold, output=None, constraints=None:
				ex_model_counter_example(solver, i_star, obj_term, in_vars,
				                         threshold, output, constraints)))


	def record_excl_region(self, r : Rad, y):
		assert(type(y) == float or type(y) == Fraction)
		self.excl_regions.append((r, y))
		log(1, 'new excluded region around',
		       str(self.excl_regions[-1][0]._i_star).replace('\n',''), 'with min value',
		       self.excl_regions[-1][1])

	def was_excluded_y(self, x):
		return [y for r,y in self.excl_regions if contains(r.bnds.values(), x)]

	def mock_model(self, in_vars, x):
		return MockModel([(v, self.cnst_ctor(i)(toz3compat(x[i])))
		                  for i,v in enumerate(in_vars)])

	def var_ctor(self, i):
		if self.spec[i]['range'] == 'int':
			return Int
		elif self.spec[i]['range'] == 'float':
			return Real
		elif all(type(v) == int for v in self.spec[i]['range']):
			return Int
		else:
			return Real


	def cnst_ctor(self, i):
		if self.spec[i]['range'] == 'int':
			return IntVal
		elif self.spec[i]['range'] == 'float':
			return RealVal
		elif all(type(v) == int for v in self.spec[i]['range']):
			return IntVal
		else:
			return RealVal


	@property
	def cati(self):
		return [i for i,s in enumerate(self.spec) if s['type'] == 'categorical']

	def denorm_for(self, b):
		return input_scaler(self.gen, b).denorm

	def norm_for(self, b):
		return input_scaler(self.gen, b).norm

	def in_vars(self, ctx=None):
		return [self.var_ctor(j)(self.spec[j]['label'], ctx=ctx)
		        for j in range(max(len(w) for w in self.model.layers[0].get_weights()[0].transpose()))]

	def _scale_nn_for_threshold(self, nn_terms):
		assert len(nn_terms) == len(self.gen['response'])
		if len(self.gen['response']) == 1 and self.gen['response'][0] == self.gen['objective']:
			# objective function is identity (modulo normalization)
			obj_term = nn_terms[0]
			if self.gen['pp']['response'] == 'min-max' and self.T_resp_bounds is self.resp_bounds:
				obj_scaler = lambda x: x
				log(1, 'obj_scaler: id')
			else:
				def obj_scaler(x):
					r = self.gen['objective']
					y = response_scaler(self.gen, self.resp_bounds[r]).denorm(x)
					return MinMax(*(self.T_resp_bounds[r][m] for m in ('min','max'))).norm(y)
				log(1, 'obj_scaler: post denorm/norm on "%s":' % self.gen['objective'],
				       'denorm:', self.resp_bounds[self.gen['objective']],
				       '/ norm: ', self.T_resp_bounds[self.gen['objective']])
		else:
			obj_term = eval(compile(self.gen['objective'], '<string>', 'eval'), {}, {
				r: s.denorm(t)
				for r, t, s in zip(self.gen['response'], nn_terms,
				                   response_scalers(self.gen, self.resp_bounds))
			})
			obj_scaler = MinMax(*obj_range(self.gen, self.T_resp_bounds)).norm
			log(1, 'obj_scaler: post pre denorm on', self.gen['response'],
				   '/ post norm on ', self.gen['objective'])
		return obj_scaler(obj_term)


	def _init_solver(self, ctx=None):
		in_vars = self.in_vars(ctx=ctx)
		normalized_inputs = []
		for j in range(len(in_vars)):
			nv = Real('n_' + self.spec[j]['label'], ctx=ctx)
			normalized_inputs.append(nv)

		nn_terms = sequential_nn_to_terms(self.model, normalized_inputs, ctx=ctx)
		obj_term = self._scale_nn_for_threshold(nn_terms)
		# obj_term is normed for comparison with thresholds

		solver = Solver(ctx=ctx)
		Vars(self.spec)(solver)
		for i,v in enumerate(in_vars):
			solver.add(self.denorm_for(self.data_bounds[self.spec[i]['label']])(normalized_inputs[i]) == v)
		for which in ('min','max'):
			BoundInput(in_vars, self.spec,
			           self.input_bounds if self.use_input_bounds else {},
			           which)(solver)

		constraints = None
		if self.more_constraints is not None:
			unnorm_resp = {r: s.denorm(t) for r, t, s in
			               zip(self.gen['response'], nn_terms,
			                   response_scalers(self.gen, self.data_bounds))}
			if log(2, 'unnorm_resp:'):
				for k,v in unnorm_resp.items():
					log(2, '  %s =' % k, v.sexpr())
			namespace = unnorm_resp
			namespace.update({e['label']: v for e,v in zip(self.spec, in_vars)})
			namespace['And'] = z3.And
			namespace['Or'] = z3.Or
			constraints = eval(compile(self.more_constraints, '<string>', 'eval'),
				{}, # globals
				namespace # locals
			)
		log(2, 'constraints:', constraints.sexpr())

		return solver, obj_term, in_vars, constraints

	def _intersect(self, a, b):
		if a is None:
			return b
		if b is None:
			return a
		return {
			'min': max(a['min'], b['min']),
			'max': min(a['max'], b['max']),
		}

	def _ensure_bounded_proper(self, label, bounds):
		bs = bounds.get(label)
		gs = (self.input_bounds if self.use_input_bounds else {}).get(label)
		ii = self._intersect(bs, gs)
		assert ii is not None
		assert ii['min'] < ii['max']
		return ii

	def _skopt_eval(self, x):
		n = len(x)
		assert(n == len(self.in_vars()))
		xn = [self.norm_for(self.data_bounds[self.spec[i]['label']])(x[i])
		      for i in range(n)]
		ym = sequential_nn_to_terms(self.model, xn,
			layer_funcs = {
				'Dense':
					lambda vars, weights, bias:
						sum(float(v) * w for v,w in zip(vars, weights)) + bias,
			},
			activations = {
				'relu'  : lambda expr, ctx=None: expr if expr >= 0 else 0,
				'linear': lambda expr, ctx=None: expr,
			}
		)
		yt = self._scale_nn_for_threshold(ym)
		return yt

	def _init_skopt(self, bounds):
		n = len(self.in_vars())

		n_initial_points = 10
		base_estimator = "GP" # {'GP', 'RF', 'ET', 'GBRT'}
		acq_func = "gp_hedge" # {'LCB', 'EI', 'PI', 'gp_hedge', 'EIps', 'PIps'}
		dims = []
		for i,s in zip(range(n), self.spec):
			assert s['type'] not in ('categorical', 'input')
			z3_ctor = self.var_ctor(i)
			if z3_ctor == Int:
				dt = skopt.space.space.Integer
			elif z3_ctor == Real:
				dt = skopt.space.space.Real
			else:
				assert z3_ctor in (Int, Real)
			#log(3, 'dim %d (%s) bounds: %s' % (i,s['label'],bounds.get(s['label'])))
			bnds = self._ensure_bounded_proper(s['label'], bounds)
			log(3, 'dim %d (%s) bounds: %s' % (i,s['label'],bnds))
			dims.append(dt(float(bnds['min']), float(bnds['max']), name=s['label']))

		mk_opt = lambda: skopt.Optimizer(
			dims,
			n_initial_points=n_initial_points,
			base_estimator=base_estimator,
			acq_func=acq_func,
			acq_optimizer="auto",
			acq_func_kwargs={},
			acq_optimizer_kwargs={},
			n_jobs=1,
		)
		#self.skopt_interp = interp1d(
		#	param_values, param_values, kind="nearest", fill_value="extrapolate"
		#)
		class ShortMemOpt:
			def __init__(self, mk_opt, mem_n=50, reset_every=150):
				self._mk_opt = mk_opt
				self._opt = mk_opt()
				self._i = -mem_n
				self._reset = reset_every
				self._mem = [None for i in range(mem_n)]

			def ask(self):
				return self._opt.ask()

			def tell(self, x, y):
				self._mem = self._mem[1:] + [(x,y)]
				self._i += 1
				if self._i <= self._reset:
					self._opt.tell(x, y)
				else:
					self._opt = self._mk_opt()
					n = 0
					for p in self._mem:
						if p is not None:
							self._opt.tell(*p)
							n += 1
					self._i = 0
					log(1, 'BO cad reset to', n, 'points')

		return ShortMemOpt(mk_opt)


	# yields a sequence of candidate solutions i_star
	def exists(self, catv, excluded, excluded_safe, center_threshold,
	           output=None, bo_cad = None, ctx=None, extra_eta=None):
		solver, obj_term, in_vars, constraints = timed(
			lambda: self._init_solver(ctx=ctx),
			'a init_solver()',
			lambda *args: log(2, *args))
		#for catv in itertools.product(*[spec[i]['range'] for i in cati]):
		Cat(in_vars, self.cati, catv)(solver)

		Grid(in_vars, self.spec)(solver)

		if bo_cad is None:
			bo_cad = 0
		else:
			assert catv == (), 'categorical variables not supported in combination with BO_CAD'
			assert not excluded, 'excluded not supported in combination with BO_CAD'
			assert not excluded_safe, 'excluded_safe not supported in combination with BO_CAD'
			opt = self._init_skopt({})

		if catv in excluded:
			log(1, 'excluding', excluded[catv])
			solver.add(*[Or(*[v != w for v,w in zip(in_vars, e)])
			             for e in excluded[catv]])
		if catv in excluded_safe:
			log(1, 'excluding safe', excluded_safe[catv])
			solver.add(*[Or(*[v != w for v,w in zip(in_vars, e)])
			             for e in excluded_safe[catv]])

		if extra_eta is not None:
			extra_eta(in_vars, self.spec, solver)

		if constraints is not None:
			solver.add(constraints)

		for candidate_idx in itertools.count(): # infinite loop
			safe_res = None
			x = None

			t_bo0 = datetime.datetime.now()
			for i in range(bo_cad):
				x = opt.ask()
				ey = self.was_excluded_y(x)
				y = self._skopt_eval(x) if len(ey) == 0 else min(ey)
				log(1, 'BO-cad #%d CT >= %s suggested x: %s -> y: %s' % (i,center_threshold,x,y))
				if y >= center_threshold:
					safe_res = sat
					i_star = self.mock_model(in_vars, x)
					dt = (datetime.datetime.now() - t_bo0).total_seconds()
					log(1, 'BO candidate %d search succeded at iteration %d in %s seconds' %
					    (candidate_idx, i, dt))
					trace_out.writerow(['A',sat,center_threshold,dt,i,*x,y])
					break
				opt.tell(x, -y)
				x = None
			if bo_cad and safe_res is None:
				dt = (datetime.datetime.now() - t_bo0).total_seconds()
				log(1, 'BO candidate %d search failed at iteration %d in %s seconds' %
				       (candidate_idx, bo_cad, dt))
				trace_out.writerow(['A',unsat,center_threshold,dt,bo_cad])

			if safe_res is None:
				# first, get model i_star for positive formulation: out >= Thresh
				safe_res, i_star = SafePoint(in_vars, self.spec, obj_term,
				                             center_threshold, output)(solver, candidate_idx)
				if safe_res == sat:
					y = i_star.eval(obj_term)
			if safe_res == unknown and (i_star == 'cancelled' or i_star == 'interrupted from keyboard'):
				raise KeyboardInterrupt()
			if safe_res != sat:
				return safe_res

			log(1, 'candidate', candidate_idx,
			       'for CT=%s is' % center_threshold,
			       [i_star[v] for v in in_vars],
			       'with value', y, 'in_vars', in_vars)
			had_cex = {}

			def cex_cb(mk_rad, y, had_cex = had_cex):
				had_cex[0] = True # just to indicate we've been called
				excl = mk_rad(in_vars)
				if excl._radius is not None:
					self.record_excl_region(excl._radius, y)
				return excl(solver, candidate_idx)

			yield i_star, cex_cb, y
			Exclude(self.spec, in_vars, i_star)(solver, candidate_idx)

			if 0 in had_cex:
				# mk_rad(in_vars) in cex_cb() above just appended the region excluding x
				if x is not None:
					opt.tell(x, -self.excl_regions[-1][1])
				log(1, 'candidate', candidate_idx,
				       'for CT=%s result:' % center_threshold,
				       'excluded by region around',
				       [self.excl_regions[-1][0]._i_star[v]
				        for v in self.excl_regions[-1][0]._in_vars],
				       'with value', self.excl_regions[-1][1])
				del had_cex[0]
			else:
				log(1, 'candidate', candidate_idx,
				       'for CT=%s result:' % center_threshold,
				       'no counter-example')


	def is_safe(self, catv, i_star, threshold, output=None, check_safe=0, ctx=None):
		solver, obj_term, in_vars, constraints = timed(
			lambda: self._init_solver(ctx=ctx),
			'b init_solver()',
			lambda *args: log(2, *args))
		Cat(in_vars, self.cati, catv)(solver)
		eps_res = unknown
		for k,f in self.counter_ex_finders:
			log(1, "'%s' searching CEX < %s" % (k,threshold))
			(eps_res, eps_model, y), t = timed(lambda: f(solver, i_star, obj_term,
			                                             in_vars, threshold, output,
			                                             constraints))
			log(2,"'%s' counter-example returned" % k, eps_res, 'in', t, 'seconds')
			if eps_res == unknown:
				log(1, "'%s' returned unknown, reason:" % k, eps_model)
				if eps_model == 'cancelled' or eps_model == 'interrupted from keyboard':
					raise KeyboardInterrupt()
				else:
					log(1, 'unrecognized reason for "unknown":', str(eps_model))
				continue
			if eps_res == sat:
				break
		if eps_res == unknown:
			raise ValueError('existence of counter-examples is "unknown": ' + str(eps_model))
		if check_safe:
			# double_check_model does not know about args.response_bounds, yet
			assert self.T_resp_bounds is self.resp_bounds
			assert len(self.gen['response']) == 1 and self.gen['response'][0] == self.gen['objective']
			assert double_check_model(self.model, self.spec, eps_res == unsat,
			                          lambda r: r >= threshold,
			                          [(i_star if eps_res == unsat else eps_model
			                           ).eval(to_real(v)).as_fraction() for v in in_vars],
			                          check_safe,
			                          [self.norm_for(self.data_bounds[s['label']])
			                           for s in self.spec])

		if eps_res == unsat:
			log(1, 'Found SAFE config:', str(i_star).replace('\n', ''))
			return True, None, None
		if eps_res == sat:
			return False, eps_model, y
		assert False


	def exists_safe(self, catv, excluded, excluded_safe, center_threshold, threshold,
	                grid_path = None, only_grid = False, output=None, check_safe=0,
	                delta = None, bo_cad = None, ctx=None, partial_grid_path = None):
		exs = []
		if grid_path is not None:
			exs.append(self.csv_spec_rows(grid_path)) # needs update to return 2nd element like self.exists(...) does
		if not only_grid:
			exs.append(self.exists(catv, excluded, excluded_safe, center_threshold, output, bo_cad))
		elif partial_grid_path is not None:
			exs.append(self.exists(catv, excluded, excluded_safe, center_threshold, output, None,
			                       extra_eta = lambda v, s, S: List(v, s, pd.read_csv(partial_grid_path))(S)))

		for i_star, cex_cb, y in itertools.chain(*exs):
			def handle_safe(is_safe, cex_model, y):
				if not is_safe and delta is not None:
					cex_cb(lambda in_vars:
						Exclude(self.spec, in_vars, cex_model,
						        radius=Rad(in_vars, self.spec, cex_model, delta=delta)),
						y)
				return is_safe

			if isinstance(threshold, list):
				assert only_grid
				th = bisect_left(threshold,
				                 lambda th: handle_safe(*self.is_safe(catv, i_star, th, output,
				                                                      check_safe, ctx)))
				log(1, 'Found config', str(i_star).replace('\n', ''),
					   'first UNSAFE for threshold idx', th, '=', threshold[th])
				assert th > 0
				yield i_star, threshold[th-1], y
			else:
				# TODO: disable "data" is_safe() check for those from csv_spec_rows?
				is_safe = timed(lambda: handle_safe(*self.is_safe(catv, i_star, threshold,
				                                                  output, check_safe, ctx)),
				                'is_safe()', lambda *args: log(1, *args))
				if is_safe:
					log(1, 'Found SAFE config for th=%s:' % threshold,
					       str(i_star).replace('\n', ''))
					yield i_star, threshold, y
				else:
					log(1, 'config', str(i_star).replace('\n', ''), 'is not safe')


	def find_safe_threshold(self, has_n, thresholds, maxN):
		ti = bisect_left(thresholds, lambda th: len(has_n.search_n(th, 1)) >= 1)
		log(1, "bisected safe threshold to be just below index", ti, "into",
		       thresholds, " -> th =", thresholds[ti-1] if ti > 0 else "ERROR")
		if ti > 0:
			assert len(has_n.res[thresholds[ti-1]]) >= 1
			return has_n.search_n(thresholds[ti-1], maxN)
		else:
			return None


	def csv_spec_rows(self, path):
		def cex_cb(mk_rad, y, had_cex):
#			had_cex[0] = True # just to indicate we've been called
#			excl = mk_rad(in_vars)
#			if excl._radius is not None:
#				self.record_excl_region(excl._radius, y)
#			return excl(solver, candidate_idx)
			pass

		with open(path, 'r') as f:
			r = csv.reader(f)
			header = { l: i for i,l in enumerate(next(r)) }
			seq = [(self.in_vars()[i], self.cnst_ctor(i), header[s['label']])
			       for i,s in enumerate(self.spec)]
			for row in r:
				log(1, 'grid point', { v: c(row[j]) for v,c,j in seq })
				yield MockModel({ v: c(row[j]) for v,c,j in seq }), cex_cb, None

		log(1, 'grid exhausted')


class has_n:
	def __init__(self, itr):
		self.srch = {}
		self.res = {}
		self._itr = itr
		self.interrupted = None

	def search_n(self, th, n):
		if th not in self.res and self.interrupted is not None:
			raise self.interrupted
		if th not in self.srch:
			self.srch[th] = self._itr(th)
		if th not in self.res:
			self.res[th] = list()
		try:
			if self.interrupted is None:
				while len(self.res[th]) < n:
					self.res[th].append(next(self.srch[th]))
		except StopIteration:
			pass
		except KeyboardInterrupt as e:
			self.interrupted = e
			if len(self.res[th]) == 0:
				del self.res[th]
				raise
		return self.res[th]

# Returns i s.t. all(pred(v) for v in a[lo:i]) and
#                all(not pred(v) for v in a[i:hi])
def bisect_left(a, pred, lo=0, hi=None):
	if lo < 0:
		raise ValueError('lo must be >= 0')
	if hi is None:
		hi = len(a)
	while lo < hi:
		m = (lo+hi) // 2
		log(1, 'bisect_left[%s ; %s) m = %s, TH[m] = %s' % (lo,hi,m,a[m]))
		if pred(a[m]):
			lo = m+1
		else:
			hi = m
	return lo


# Returns i s.t. all(pred(v) for v in a[lo:i]) and
#                all(not pred(v) for v in a[i:hi])
def bisect_left_discrete(pred, lo, hi):
	while lo < hi:
		m = (lo+hi) // 2
		if pred(m):
			lo = m+1
		else:
			hi = m
	return lo


def excluded_by_trace(inst, trace_path, trace_exclude_safe):
	excluded = {}
	with open(trace_path, 'r') as f:
		c = None
		last_found = None
		last_safe = False
		for l in csv.reader(f, dialect='unix'):
			if l[0] == Cat.LABEL:
				c = tuple(l[1:])
				excluded.setdefault(c, [])
			if l[0] == Exclude.LABEL:
				excluded[c].append(l[1:])
				log(1, 'excluding-x', l[1:], 'for', c)
			if l[0] == SafePoint.LABEL and l[1] == 'sat':
				last_safe = False
				last_found = l[4:-1]
				assert len(last_found) == len(inst.in_vars())
			if l[0] == CounterExample.LABEL:
				last_safe = l[1] == 'sat'
			if trace_exclude_safe and last_safe:
				excluded[c].append(last_found)
				log(1, 'excluding-X', last_found, 'for', c)
	return excluded


def init_bounds(spec, model_gen, data_bounds_json_path=None, bounds_factor=None,
                T_resp_bounds_csv_path=None):
	# init 'bounds' for any feature: determines search space
	#   a) from grid in spec,
	#   b) from args.data_bounds (optional file via "-B")
	#   c) optionally grown by a user-specified factor,
	# 'data_bounds' for input features from args.data_bounds: used for scaling
	#   NN's inputs to training data range,
	# 'resp_bounds' for response/output features contained in
	#   model_gen['response'] from args.data_bounds: used for scaling NN's
	#   outputs to training data range, and
	# 'T_resp_bounds' to either 'resp_bounds' or from file ("-r"): gives the
	#   scale the thresholds T and ST are interpreted in
	bounds = {
		s['label']: { 'min': min(s['safe']), 'max': max(s['safe']) } if 'safe' in s else {}
		for s in spec #if s['type'] != 'input' # if s['type'] == 'knob'
	}
	resp_bounds = None
	if data_bounds_json_path is not None:
		with open(data_bounds_json_path, 'r') as f:
			data_bounds = json.load(f, parse_float=Fraction)
		for k,b in data_bounds.items():
			if k in bounds:
				if 'min' in b:
					bounds[k]['min'] = (min(bounds[k]['min'], b['min'])
					                    if 'min' in bounds[k]
					                    else b['min'])
				if 'max' in b:
					bounds[k]['max'] = (max(bounds[k]['max'], b['max'])
					                    if 'max' in bounds[k]
					                    else b['max'])
			if k in model_gen['response']:
				if resp_bounds is None:
					resp_bounds = {}
				resp_bounds[k] = { m: v for m,v in b.items() if m in ('min','max') }
	if bounds_factor is not None:
		for s in spec:
			if s['label'] in bounds:
				b = bounds[s['label']]
				if 'min' in b and 'max' in b:
					d = b['max'] - b['min']
					b['min'] -= d * bounds_factor / 2
					b['max'] += d * bounds_factor / 2;
	if model_gen['pp']['features'] == 'min-max':
		if data_bounds_json_path is None:
			die(1, 'error: NN expects normalized inputs, require bounds via param "-B"')
		for s in spec:
			if s['type'] == 'input':
				continue
			try:
				b = bounds[s['label']]
			except KeyError:
				die(1, "error: no bounds provided for variable '%s'" % s['label'])
			try:
				b['min'], b['max'] = b['min'], b['max']
			except KeyError:
				die(1, "error: bounds for variable '%s' do not include both, "
				       "'min' and 'max'" % s['label'])

	# NN model produces output normed with resp_bounds
	# T_resp_bounds relate to the thresholds T and ST
	if T_resp_bounds_csv_path is None:
		T_resp_bounds = resp_bounds
	else:
		T_resp_bounds = pd.read_csv(T_resp_bounds_csv_path, index_col=0)
		if all(T_resp_bounds[f][m] == resp_bounds[f][m]
			   for f in model_gen['response']
			   for m in ('min','max')):
			log(1, 'T_resp_bounds = resp_bounds')
			T_resp_bounds = resp_bounds

	return bounds, data_bounds, resp_bounds, T_resp_bounds


def main(argv):
	args = parse_args(argv)
	global log

	def log(lvl, *v):
		if lvl > args.verbose:
			return False
		print(*v, file=sys.stderr, flush=True)
		return True

	with open(args.spec, 'r') as spec_file:
		all_spec = json.load(spec_file, parse_float=Fraction)
		spec = [s for s in all_spec if s['type'] in ['categorical','knob','input']]
		del all_spec

	assert(type(spec) == list)

	with open(args.model_gen, 'r') as f:
		model_gen = json.load(f)


	# init 'bounds' for any feature: determines search space
	#   a) from grid in spec,
	#   b) from args.data_bounds (optional file via "-B")
	#   c) optionally grown by a user-specified factor,
	# 'data_bounds' for input features from args.data_bounds: used for scaling
	#   NN's inputs to training data range,
	# 'resp_bounds' for response/output features contained in
	#   model_gen['response'] from args.data_bounds: used for scaling NN's
	#   outputs to training data range, and
	# 'T_resp_bounds' to either 'resp_bounds' or from file ("-r"): gives the
	#   scale the thresholds T and ST are interpreted in
	bounds, data_bounds, resp_bounds, T_resp_bounds = init_bounds(spec, model_gen,
	                                                              args.data_bounds,
	                                                              args.bounds,
	                                                              args.response_bounds)

#	bounds = {
#		s['label']: { 'min': min(s['safe']), 'max': max(s['safe']) } if 'safe' in s else {}
#		for s in spec #if s['type'] != 'input' # if s['type'] == 'knob'
#	}
#	resp_bounds = None
#	if args.data_bounds is not None:
#		with open(args.data_bounds, 'r') as f:
#			data_bounds = json.load(f, parse_float=Fraction)
#		for k,b in data_bounds.items():
#			if k in bounds:
#				if 'min' in b:
#					bounds[k]['min'] = (min(bounds[k]['min'], b['min'])
#					                    if 'min' in bounds[k]
#					                    else b['min'])
#				if 'max' in b:
#					bounds[k]['max'] = (max(bounds[k]['max'], b['max'])
#					                    if 'max' in bounds[k]
#					                    else b['max'])
#			if k in model_gen['response']:
#				if resp_bounds is None:
#					resp_bounds = {}
#				resp_bounds[k] = { m: v for m,v in b.items() if m in ('min','max') }
#	if args.bounds is not None:
#		for s in spec:
#			if s['label'] in bounds:
#				b = bounds[s['label']]
#				if 'min' in b:
#					b['min'] -= b['min'] * args.bounds
#				if 'max' in b:
#					b['max'] += b['max'] * args.bounds
#	if model_gen['pp']['features'] == 'min-max':
#		if args.data_bounds is None:
#			print('error: NN expects normalized inputs, require bounds via param "-B"',
#				  file=sys.stderr)
#			return 1
#		for s in spec:
#			if s['type'] == 'input':
#				continue
#			try:
#				b = bounds[s['label']]
#			except KeyError:
#				print("error: no bounds provided for variable '%s'" % s['label'],
#				      file=sys.stderr)
#				return 1
#			try:
#				b['min'], b['max'] = b['min'], b['max']
#			except KeyError:
#				print(("error: bounds for variable '%s' do not include both, "+
#				       "'min' and 'max'") % s['label'],
#				      file=sys.stderr)
#				return 1
#
#	# NN model produces output normed with resp_bounds
#	# T_resp_bounds relate to the thresholds T and ST
#	if args.response_bounds is None:
#		T_resp_bounds = resp_bounds
#	else:
#		T_resp_bounds = pd.read_csv(args.response_bounds, index_col=0)
#		if all(T_resp_bounds[f][m] == resp_bounds[f][m]
#			   for f in model_gen['response']
#			   for m in ('min','max')):
#			log(1, 'T_resp_bounds = resp_bounds')
#			T_resp_bounds = resp_bounds
	log(1, 'bounds:', bounds)
	log(1, 'data_bounds:', data_bounds)
	log(1, 'resp_bounds:', resp_bounds)
	log(1, 'T_resp_bounds:', T_resp_bounds)

	inst = Instance(args.spec, load_model(args.nn_model), model_gen, data_bounds,
	                args.bounds is not None, bounds, resp_bounds, T_resp_bounds,
	                args.data, args.bo_cex, args.more_constraints)

	excluded = {} # dict from catv -> [[x1,x2,...,xn], ...]
	if args.trace_exclude is not None and os.path.exists(args.trace_exclude):
		excluded = excluded_by_trace(inst, args.trace_exclude, args.trace_exclude_safe)

	excluded_safe = {}
	safe_n = {}
	if args.safe is not None and os.path.exists(args.safe):
		for s in pd.read_csv(args.safe).iterrows():
			c = tuple(np2py(s[1][i]) for i in inst.cati)
			excluded_safe.setdefault(c, [])
			excluded_safe[c].append([np2py(w, lenient=True) for w in s[1].values])
			safe_n.setdefault(c, 0)
			safe_n[c] += 1
			del c
	log(1, 'excluded safe', excluded_safe)


	safe_path = os.devnull if args.safe is None else args.safe
	ex = os.path.exists(safe_path)
	try:
		with open(safe_path, 'at', newline='', buffering=1) as f:
			w = csv.writer(f, dialect='unix', quoting=csv.QUOTE_MINIMAL)
			if not ex:
				if isinstance(args.threshold, list):
					w.writerow([s['label'] for s in spec] + ['center_obj','thresh'])
				else:
					w.writerow([s['label'] for s in spec] + ['center_obj'])


			for catv in itertools.product(*[spec[i]['range'] for i in inst.cati]):
				safe_n.setdefault(catv, 0)
				if safe_n[catv] >= args.n:
					break

				s = has_n(lambda th: inst.exists_safe(catv, excluded, excluded_safe,
				                                      th + args.center_offset, th,
				                                      args.grid, args.no_exists,
				                                      args.output, args.check_safe,
				                                      args.delta, args.bo_cad,
				                                      partial_grid_path=args.partial_grid))
				if True:
					res = inst.find_safe_threshold(s, args.threshold, args.n)
					#assert res is not None
					if res is None:
						continue
					for i_star, th, y in res:
						def enc(n):
							return (n.as_string()
							        if n.is_real() and not n.is_int_value()
							        else str(n.as_long()))
						if isinstance(args.threshold, list):
							w.writerow([enc(i_star[v]) for v in inst.in_vars()] + [enc(y),th])
						else:
							w.writerow([enc(i_star[v]) for v in inst.in_vars()] + [enc(y)])
						safe_n[catv] += 1
						if safe_n[catv] >= args.n:
							break
					if True:
						continue

				for i_star, th in inst.exists_safe(catv, excluded, excluded_safe,
				                                   args.safe_threshold,
				                                   args.threshold,
				                                   args.grid, args.no_exists,
				                                   args.output, args.check_safe,
				                                   args.delta, args.bo_cad,
				                                   partial_grid_path=args.partial_grid):
					def enc(n):
						return (n.as_string()
						        if n.is_real() and not n.is_int_value()
						        else str(n.as_long()))
					if isinstance(args.threshold, list):
						w.writerow([enc(i_star[v]) for v in inst.in_vars()] + [th])
					else:
						w.writerow([enc(i_star[v]) for v in inst.in_vars()])
					safe_n[catv] += 1
					if safe_n[catv] >= args.n:
						break

	except KeyboardInterrupt:
		log(1,'cancelled')
		return 1

	return (0 if all(safe_n[catv] >= args.n
	                 for catv in itertools.product(*[spec[i]['range']
	                                                 for i in inst.cati]))
	        else 0)


if __name__ == "__main__":
	sys.exit(main(sys.argv))
