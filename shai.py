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

from fractions import Fraction
from decimal import Decimal
import itertools, json

from common import *

def parse_args(argv):
	p = argparse.ArgumentParser(prog=argv[0])
	p.add_argument('nn_model', metavar='NN_MODEL', type=str,
	               help='Path to NN model in .h5 format')
	p.add_argument('-b', '--bounds', type=float, nargs='?', const=0.0,
	               help='bound variables  [default: none; otherwise, if BOUNDS '+
	                    'is missing, 0]')
	p.add_argument('-B', '--data-bounds', default=None, type=str, metavar='DBOUNDS',
	               help='path to data_bounds file to amend the bounds determined from SPEC')
	p.add_argument('-C', '--check-safe', type=int, default=1000,
	               help='Number of random samples to check for each SAFE config '+
	                    'found [default: 1000]')
	p.add_argument('-d', '--data', type=str,
	               help='path to DATA.csv; check DATA for counter-examples to '+
	                    'found regions')
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


class Rad(Command):
	LABEL = 'r'
	def __init__(self, in_vars, spec, i_star):
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

		def py2z3(v):
			if type(v) == int:
				return IntVal(v)
			else:
				assert type(v) == Fraction
				return RealVal(v)

		self._c = []
		sub_args = []
		for s,v in zip(spec, in_vars):
			if s['type'] == 'category' or s['type'] == 'input':
				continue
			try:
				def is_zero(val):
					if val.is_int():
						return val.as_long() == 0
					assert val.is_real()
					return val.numerator_as_long() == 0
				if 'rad-rel' in s and not is_zero(i_star[v]):
					e = s['rad-rel']
					t, c, rng = ('rel', rel_err(v, i_star[v], e, s), '%g%%' % (e*100))
				else:
					try:
						t, e = 'abs', s['rad-abs']
					except KeyError:
						t, e = 'rel', s['rad-rel']
					c, rng = (abs_err(v, i_star[v], e, s), '%g' % e)
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


class CounterExample(Solve, Command):
	LABEL = 'b'
	def __init__(self, obj_term, threshold, output, in_vars):
		Command.__init__(self, CounterExample.LABEL, threshold)
		Solve.__init__(self,
		               None if output is None else output + '-' + CounterExample.LABEL,
		               obj_term, in_vars)
		self._obj_term = obj_term

	def run(self, solver):
		threshold = self.args[0]
		log(1, 'solving safe <', threshold, 'with eps ...')
		solver.add(self._obj_term < threshold)
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
	def __init__(self, spec, in_vars, i_star):
		super().__init__(Exclude.LABEL,
		                 *[i_star[v] if s['type'] != 'input' else None
		                   for s,v in zip(spec, in_vars)
		                   if s['type'] != 'category'])
		self._in_vars = in_vars

	def run(self, solver, candidate_idx):
		log(1, 'excluding candidate', candidate_idx)
		solver.add(Or([v != w for v,w in zip(self._in_vars, self.args)
		               if w is not None]))


class Simple(Command):
	def __init__(self, func_name, *args):
		super().__init__(func_name, *args)

	def run(self, solver):
		return getattr(solver, self.label)(*self.args)


class Instance:
	def __init__(self, spec_path, model, gen, data_bounds, use_input_bounds,
	             input_bounds, resp_bounds, T_resp_bounds, data_path=None):

		with open(spec_path, 'r') as f:
			all_spec = json.load(f, parse_float=Fraction)
			self.spec = [s for s in all_spec if s['type'] in ['category','knob','input']]
			del all_spec

		assert(type(self.spec) == list)

		assert max(len(w) for w in model.layers[0].get_weights()[0].transpose()) == len(self.spec)

		self.model = model
		self.gen = gen
		self.data_bounds = data_bounds
		self.use_input_bounds = use_input_bounds
		self.input_bounds = input_bounds
		self.T_resp_bounds = T_resp_bounds
		self.resp_bounds = resp_bounds

		self.counter_ex_finders = []
		if data_path is not None:
			objective = None
			if 'objective' in self.gen: # default now
				objective = self.gen['objective']
			elif len(self.gen['response']) == 1:
				objective = self.gen['response'][0]
			if objective is not None:
				cd = CD(objective, spec_path, data_path,
						[s['label'] for s in self.spec if s['type'] != 'category'],
						bnds=(None if self.T_resp_bounds is None else
						      [self.T_resp_bounds[r][v]
						       for r in self.gen['response']
						       for v in ('min','max')]))
				def ex_data_counter_example(asgn, threshold):
					n_fail, n_ok = cd.check([v.as_long() for v in asgn], # TODO: Fraction support
					                        threshold)
					log(2,'data in ball: %d fail, %d ok' % (n_fail, n_ok))
					return sat if n_fail > 0 else unsat, None
				self.counter_ex_finders.append(('data',
					lambda solver, i_star, obj_term, in_vars, threshold, output=None:
						ex_data_counter_example([
							i_star[v] for s,v in zip(self.spec, in_vars) if s['type'] != 'category'
						], threshold)))
			else:
				log(1, 'WARNING: > 1 responses unsupported for data-check, skipping...')

		def ex_model_counter_example(solver, i_star, obj_term, in_vars, threshold, output):
			#Simple('push')(solver)
			Rad(in_vars, self.spec, i_star)(solver)
			eps_res, eps_model = CounterExample(obj_term, threshold, output, in_vars)(solver)
			#Simple('pop')(solver)
			if eps_res == sat:
				log(1,'Found counter-example') or log(2,'Found counter-example:',
				                                        str(eps_model).replace('\n', ''))
			elif eps_res == unsat:
				log(1, 'Found no counter-example:', str(i_star).replace('\n', ''))
			return eps_res, eps_model

		self.counter_ex_finders.append(('NN',
			lambda solver, i_star, obj_term, in_vars, threshold, output=None:
				ex_model_counter_example(solver, i_star, obj_term, in_vars,
				                         threshold, output)))


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
		return [i for i,s in enumerate(self.spec) if s['type'] == 'category']

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

		return solver, obj_term, in_vars


	# yields a sequence of candidate solutions i_star
	def exists(self, catv, excluded, excluded_safe, center_threshold,
			   output=None, ctx=None, extra_eta=None):
		solver, obj_term, in_vars = timed(lambda: self._init_solver(ctx=ctx),
		                                  'a init_solver()',
		                                  lambda *args: log(2, *args))
		#for catv in itertools.product(*[spec[i]['range'] for i in cati]):
		Cat(in_vars, self.cati, catv)(solver)

		Grid(in_vars, self.spec)(solver)

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

		for candidate_idx in itertools.count(): # infinite loop
			# first, get model i_star for positive formulation: out >= Thresh
			safe_res, i_star = SafePoint(in_vars, self.spec, obj_term,
			                             center_threshold, output)(solver, candidate_idx)
			if safe_res == unknown and (i_star == 'cancelled' or i_star == 'interrupted from keyboard'):
				raise KeyboardInterrupt()
			if safe_res != sat:
				return safe_res
			yield i_star
			Exclude(self.spec, in_vars, i_star)(solver, candidate_idx)


	def is_safe(self, catv, i_star, threshold, output=None, check_safe=0, ctx=None):
		solver, obj_term, in_vars = timed(lambda: self._init_solver(ctx=ctx),
		                                  'b init_solver()',
		                                  lambda *args: log(2, *args))
		Cat(in_vars, self.cati, catv)(solver)
		eps_res = unsat
		for k,f in self.counter_ex_finders:
			(eps_res, eps_model), t = timed(lambda: f(solver, i_star, obj_term,
			                                          in_vars, threshold, output))
			log(2,"'%s' counter-example returned" % k, eps_res, 'in', t, 'seconds')
			if eps_res != unsat:
				break
		if eps_res == unknown:
			log(1, 'unknown reason:', eps_model)
			if eps_model == 'cancelled' or eps_model == 'interrupted from keyboard':
				raise KeyboardInterrupt()
			else:
				raise ValueError('unrecognized reason for "unknown": ' + str(eps_model))
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
			return True, None
		if eps_res == sat:
			return False, eps_model
		assert False


	def exists_safe(self, catv, excluded, excluded_safe, center_threshold, threshold,
	                grid_path = None, only_grid = False, output=None, check_safe=0, ctx=None,
			partial_grid_path = None):
		exs = []
		if grid_path is not None:
			exs.append(self.csv_spec_rows(grid_path))
		if not only_grid:
			exs.append(self.exists(catv, excluded, excluded_safe, center_threshold, output))
		elif partial_grid_path is not None:
			exs.append(self.exists(catv, excluded, excluded_safe, center_threshold, output,
			                       extra_eta = lambda v, s, S: List(v, s, pd.read_csv(partial_grid_path))(S)))

		for i_star in itertools.chain(*exs):
			if isinstance(threshold, list):
				assert only_grid
				th = bisect_left(threshold,
				                 lambda th: self.is_safe(catv, i_star, th, output,
				                                         check_safe, ctx)[0])
				log(1, 'Found config', str(i_star).replace('\n', ''),
					   'first UNSAFE for threshold idx', th, '=', threshold[th])
				assert th > 0
				yield i_star, threshold[th-1]
			else:
				# TODO: disable "data" is_safe() check for those from csv_spec_rows?
				is_safe, why = timed(lambda: self.is_safe(catv, i_star, threshold,
				                                          output, check_safe, ctx),
				                     'is_safe()', lambda *args: log(1, *args))
				if is_safe:
					log(1, 'Found SAFE config:', str(i_star).replace('\n', ''))
					yield i_star, threshold
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
		with open(path, 'r') as f:
			r = csv.reader(f)
			header = { l: i for i,l in enumerate(next(r)) }
			seq = [(self.in_vars()[i], self.cnst_ctor(i), header[s['label']])
			       for i,s in enumerate(self.spec)]
			for row in r:
				log(1, 'grid point', { v: c(row[j]) for v,c,j in seq })
				yield { v: c(row[j]) for v,c,j in seq }
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
		spec = [s for s in all_spec if s['type'] in ['category','knob','input']]
		del all_spec

	assert(type(spec) == list)

	with open(args.model_gen, 'r') as f:
		model_gen = json.load(f)

	bounds = {
		s['label']: { 'min': min(s['safe']), 'max': max(s['safe']) } if 'safe' in s else {}
		for s in spec #if s['type'] != 'input' # if s['type'] == 'knob'
	}
	resp_bounds = None
	if args.data_bounds is not None:
		with open(args.data_bounds, 'r') as f:
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
	if args.bounds is not None:
		for s in spec:
			if s['label'] in bounds:
				b = bounds[s['label']]
				if 'min' in b:
					b['min'] -= b['min'] * args.bounds
				if 'max' in b:
					b['max'] += b['max'] * args.bounds
	if model_gen['pp']['features'] == 'min-max':
		if args.data_bounds is None:
			print('error: NN expects normalized inputs, require bounds via param "-B"',
				  file=sys.stderr)
			sys.exit(1)
		for s in spec:
			if s['type'] == 'input':
				continue
			try:
				b = bounds[s['label']]
			except KeyError:
				print("error: no bounds provided for variable '%s'" % s['label'],
				      file=sys.stderr)
				sys.exit(1)
			try:
				b['min'], b['max'] = b['min'], b['max']
			except KeyError:
				print(("error: bounds for variable '%s' do not include both, "+
				       "'min' and 'max'") % s['label'],
				      file=sys.stderr)
				sys.exit(1)

	# NN model produces output normed with resp_bounds
	# T_resp_bounds relate to the thresholds T and ST
	if args.response_bounds is None:
		T_resp_bounds = resp_bounds
	else:
		T_resp_bounds = pd.read_csv(args.response_bounds, index_col=0)
		if all(T_resp_bounds[f][m] == resp_bounds[f][m]
			   for f in model_gen['response']
			   for m in ('min','max')):
			log(1, 'T_resp_bounds = resp_bounds')
			T_resp_bounds = resp_bounds
	log(1, 'bounds:', bounds)
	log(1, 'data_bounds:', data_bounds)
	log(1, 'resp_bounds:', resp_bounds)
	log(1, 'T_resp_bounds:', T_resp_bounds)

	inst = Instance(args.spec, load_model(args.nn_model), model_gen, data_bounds,
	                args.bounds is not None, bounds, resp_bounds, T_resp_bounds,
	                args.data)

	excluded = {} # dict from catv -> [[x1,x2,...,xn], ...]
	if args.trace_exclude is not None and os.path.exists(args.trace_exclude):
		excluded = excluded_by_trace(inst, args.trace_exclude, args.trace_exclude_safe)

	excluded_safe = {}
	safe_n = {}
	if args.safe is not None and os.path.exists(args.safe):
		for s in pd.read_csv(args.safe).iterrows():
			c = tuple(np2py(s[1][i]) for i in inst.cati)
			excluded_safe.setdefault(c, [])
			excluded_safe[c].append([np2py(w) for w in s[1].values])
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
					w.writerow([s['label'] for s in spec] + ['thresh'])
				else:
					w.writerow([s['label'] for s in spec])


			for catv in itertools.product(*[spec[i]['range'] for i in inst.cati]):
				safe_n.setdefault(catv, 0)
				if safe_n[catv] >= args.n:
					break

				s = has_n(lambda th: inst.exists_safe(catv, excluded, excluded_safe,
				                                      th + args.center_offset, th,
				                                      args.grid, args.no_exists,
				                                      args.output, args.check_safe,
				                                      partial_grid_path=args.partial_grid))
				if True:
					res = inst.find_safe_threshold(s, args.threshold, args.n)
					#assert res is not None
					if res is None:
						continue
					for i_star, th in res:
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
					if True:
						continue

				for i_star, th in inst.exists_safe(catv, excluded, excluded_safe,
				                                   args.safe_threshold,
				                                   args.threshold,
				                                   args.grid, args.no_exists,
				                                   args.output, args.check_safe):
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
		sys.exit(1)

	sys.exit(0 if all(safe_n[catv] >= args.n
	                  for catv in itertools.product(*[spec[i]['range']
	                                                  for i in inst.cati]))
	         else 0)


if __name__ == "__main__":
	main(sys.argv)
