#!/usr/bin/env python3

from smlp.comm import *

import asyncio, argparse, sys, logging, shlex, heapq

from typing import Mapping, Sequence, Tuple, Any
from dataclasses import dataclass, field

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

# forwarding Pool; uses a database object to cache/persist results
class StoredPool(Pool):
	def __init__(self, parent, db):
		self._parent = parent
		self._db = db

	async def pop(self):
		while True:
			pr = await self._parent.pop()
			if pr is None or pr.id not in self._db:
				return pr
			self._parent.push(pr.id, self._db[pr.id])

	def push(self, pr, result):
		if result is not None:
			self._db[pr.id] = result
		self._parent.push(pr, result)

	def wait_empty(self):
		return self._parent.wait_empty()


class UNSAT:
	pass

class SAT:
	def __init__(self, model=None):
		self.model = model

# assumes LC_ALL=*.UTF-8
def smtlib2_instance(logic : str,
                     cnst_decls : Mapping[str, str], # name -> type
                     cnst_defs : Mapping[str, Tuple[str,str]], # name -> (type, term)
                     assert_terms : Sequence[str], # term
                     need_model : bool,
                     timeout : int=None) -> bytes:
	r = ''
	r += '(set-option :print-success false)\n'
	if timeout is not None:
		r += '(set-option :timeout %s)\n' % timeout
	if need_model:
		r += '(set-option :produce-models true)\n'
	r += '(set-logic %s)\n' % logic
	for n,ty in cnst_decls.items():
		r += '(declare-fun %s () %s)\n' % (n,ty)
	for n,(ty,tm) in cnst_defs.items():
		r += '(define-fun %s () %s %s)\n' % (n,ty,tm)
	for tm in assert_terms:
		r += '(assert %s)\n' % tm
	r += '(check-sat)\n'
	if need_model:
		r += '(get-model)\n'
	r += '(exit)'
	return r.encode()

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

if __name__ == "__main__":
	# python-3.8: creating 10^6 futures from asyncio.get_event_loop():
	#             time: 1.6sec, memory: 152M
	#             [None for i in range(1000000)]:
	#             .15sec, 22.5M
	#          -> per .create_future(): 1.45µs, 136 bytes

	listen()

	class MsgHandleRecv:
		def handle_recv(msg_id, res):
			log.debug('request got reply: %s', res)
			assert res.version == VERSION
			assert res.msg_id == msg_id
			assert res.HasField('reply')
			assert not res.HasField('request')
			handle_reply(res.reply)

	class CmdHandleReply:
		def handle_reply(rep):
			assert rep.type == rep.Type.SMTLIB_REPLY
			handle_command(rep.cmd)

	class SaneHandleCmd:
		def handle_sane_command(conn, script, cmd):
			if cmd.status == 0:
				handle_stdout(cmd.stdout)
			else:
				raise CalledProcessError(returncode=cmd.status,
				                         cmd=(conn, script),
				                         output=cmd.stdout,
				                         stderr=cmd.stderr)

	class SmtlibHandleStdout:
		def handle_stdout(result):
			sat_res, _, model = result.partition(b'\n')
			assert sat_res in (b'sat', b'unsat', b'unknown')
			m = smtlib_script_parse(model)
			#logging.info('result for id %s is %s with model %s',
			#             prid, sat_res, m)
			handle_result(sat_res, m)

	class Z3HandleCmd:
		def handle_z3_command(conn, script, cmd):
			if cmd.status == 1 and cmd.stderr == b'':
				try:
					handle_stdout(cmd.stdout)
				except:
					pass
			handle_stdout(handle_command(conn, script, cmd))

	@dataclass(order=True)
	class Item:
		priority : Any
		id       : Any=field(compare=False)
		instance : bytes=field(compare=False)
		# type probably asyncio.Future, but depends on the event
		# loop implementation
		reply    : Any=field(compare=False)

		# TODO: replace this method further down the protocol stack
		def handle_result(self, res):
			self.reply.set_result(res)

	class TestPool(Pool):
		def __init__(self, loop):
			self.heap = [Item(priority=(0,),id=('',),instance=b'',
			                  reply=loop.create_future()),]
			inst = smtlib2_instance('QF_LRA',
			                        { 'x': 'Real', 'y': 'Real' }, {},
			                        ['(> x y)', '(> x y)'], True)
			self.heap.append(Item(priority=(1,), id=('test',),
			                      instance=inst,
			                      reply=loop.create_future()))
			heapq.heapify(self.heap)
			self.empty = loop.create_future()

		async def pop(self):
			while len(self.heap) > 0:
				item = heapq.heappop(self.heap)
				if item.reply.done():
					continue
				return item
			# notify self.wait_empty()
			self.empty.set_result(None)

		def push(self, item, result):
			if item.id == ('test',):
				sat_res, _, model = result.partition(b'\n')
				logging.info('result for id %s is %s with model %s', item.id,
					     sat_res, smtlib_script_parse(model))
				assert sat_res in (b'sat', b'unsat', b'unknown')

		async def wait_empty(self):
			await self.empty

		async def solve(self, prio, prid, instance : bytes):
			fut = self._loop.create_future()
			heapq.heappush(self.heap, Item(priority=prio, id=prid,
			                               instance=instance,
			                               reply=fut))
			stdout = await fut
			return stdout

	try:
		args = parse_args(sys.argv)
	except ValueError as e:
		print('error:', e, file=sys.stderr)
		sys.exit(1)

	loop = asyncio.get_event_loop()

	if args.client is not None:
		coro = client(args.host, args.port, args.client)
	else:
		pool = TestPool(loop)
		pool = StoredPool(pool, dict())
		coro = server(args.host, args.port, pool)

	#task = asyncio.ensure_future(coro) # loop.create_task(coro)
	task = coro

	run1(task, loop=loop)
