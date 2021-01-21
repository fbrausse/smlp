#!/usr/bin/env python3

#from smlp.comm import client # , run1
#from smlp.solverpool import *

from smlp import *

import asyncio, argparse, sys, logging, shlex, functools, time

import code, traceback, signal

def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)

def listen():
    signal.signal(signal.SIGUSR1, debug)  # Register handler

HOST = None
PORT = 1337

def distribute(pool, config=None):
	if config is None:
		config = {}
	host = config.get('host', HOST)
	port = config.get('port', PORT)
	return server(host, port, pool)

def parse_args(argv):
	class LogLevel(argparse.Action):
		def __call__(self, parser, namespace, values, option_string=None):
			l = getattr(logging, values.upper(), None)
			if not isinstance(l, int):
				raise ValueError('Invalid log level: %s' % values)
			logging.basicConfig(level=l)

	class ClientCommand(argparse.Action):
		def __init__(self, *args, **kwds):
			super().__init__(*args, **kwds)

		def __call__(self, parser, namespace, values, option_string=None):
			setattr(namespace, self.dest, shlex.split(values))

	p = argparse.ArgumentParser(prog=argv[0])
	p.add_argument('-c', '--client', default=None, metavar='CMD', type=str,
	               action=ClientCommand, help='start client mode')
	p.add_argument('-H', '--host', default=HOST, type=str)
	p.add_argument('-P', '--port', default=PORT, type=int)
	p.add_argument('-v', '--log-level', metavar='LVL', type=str, action=LogLevel)
	args = p.parse_args(argv[1:])
	return args

class Instance:
	def __init__(self):
		self.parent = None
		self.dom = None
		self.codom = None
		self.obj = None

class SMLP:
	def __init__(self, config_decls, input_decls, eta, theta, phi):
		self.configs = config_decls
		self.inputs  = input_decls
		self.eta     = eta
		self.theta   = theta
		self.phi     = phi

	def candidate(self, real_pred):
		# TODO: return SMT instance: Ep Eq eta /\ theta /\ phi(real_pred)
		pass

	def counterex(self, model, real_pred):
		# TODO: return SMT instance: Eq theta[model|p] /\ not phi(real_pred)
		pass

	def exclude(self, model):
		return SMLP(self.eta and not model, self.theta, self.phi)

async def enumerate_sol(solver, smlp):
	while True:
		sol = await solver.solve(smlp.exists())
		if isinstance(sol, UNSAT):
			break
		yield sol
		smlp = smlp.exclude(sol)

# asynchronous generator
async def threshold1(solver, smlp, th, prec):
	ex = smlp
	al = smlp
	while True:
		hi = await solver.solve(ex.candidate(lambda x: x >= th + prec))
		if isinstance(hi, UNSAT):
			break
		lo = await solver.solve(al.at(hi).exists(lambda x: x >= th))
		if isinstance(lo, UNSAT):
			yield hi
		ex = ex.exclude(hi)



#async def solve_shai(solver, spec, path):
#	pass

async def solve_specific(solver):
	x = numeric(Real, 'x')
	y = numeric(Real, 'y')
	xy = y * x
	xyy = y * xy
	a, b = await asyncio.gather(
		solver.solve((1,), ('',), QF_NRA({}, {}, [], False)),
		solver.solve((0,), ('test',),
		             QF_NRA({ d.sym: d.ty.__name__ for d in (x,y) }, {},
		                    [(x > y)  #'(> x y)'
		                    , (xy < xyy + xyy) #'(< (* y x) y)'
		                    ], True #, timeout=1
		                   )),
		return_exceptions=True
	)
	c = await solver.solve((0,), ('',), QF_NRA({}, {}, [], False))
	return 42

async def main():
	try:
		args = parse_args(sys.argv)
	except ValueError as e:
		print('error:', e, file=sys.stderr)
		return 1

	if args.client is not None:
		await client(args.host, args.port, args.client)
	else:
		#pool = TestPool(loop)
		#sol  = solve_instance(pool)
		#coro = server(args.host, args.port, StoredPool(pool, dict()))
		#coro = asyncio.gather(coro, sol)
		await run_solver(args.host, args.port, solve_specific)

	return 0

if __name__ == "__main__":
	# python-3.8: creating 10^6 futures from asyncio.get_event_loop():
	#             time: 1.6sec, memory: 152M
	#             [None for i in range(1000000)]:
	#             .15sec, 22.5M
	#          -> per .create_future(): 1.45µs, 136 bytes

	listen() # for debugging

	##task = asyncio.ensure_future(coro) # loop.create_task(coro)
	#task = coro

	#run1(task, loop=loop)
	sys.exit(run1(main()))
