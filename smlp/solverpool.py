
from .comm import *

import asyncio, heapq, functools, logging, time

from typing import Mapping, Sequence, Tuple, Any
from dataclasses import dataclass, field

# forwarding Pool; uses a database object to cache/persist results
class StoredPool(Pool):
	def __init__(self, parent, db):
		self._parent = parent
		self._db = db

	def _done_cb(self, prid, reply):
		try:
			self._db[prid] = reply.result()
		except:
			pass

	async def pop(self):
		while True:
			pr = await self._parent.pop()
			if pr.id is not None:
				if pr.id in self._db:
					logging.info('providing cached result from DB for id %s', pr.id)
					pr.reply.set_result(self._db[pr.id])
					continue
				pr.reply.add_done_callback(functools.partial(self._done_cb, pr.id))
			return pr

	def wait_empty(self):
		return self._parent.wait_empty()


class UNSAT:
	pass

class SAT:
	def __init__(self, model=None):
		self.model = model

@dataclass
class Smtlib2:
	logic : str
	cnst_decls : Mapping[str, str] # name -> type
	cnst_defs : Mapping[str, Tuple[str,str]] # name -> (type, term)
	assert_terms : Sequence[str] # term
	need_model : bool
	timeout : int=None

	# assumes LC_ALL=*.UTF-8
	def format(self) -> bytes:
		r = ''

		r += '(set-option :print-success false)\n'
		if self.timeout is not None:
			r += '(set-option :timeout %s)\n' % self.timeout
		if self.need_model:
			r += '(set-option :produce-models true)\n'
		r += '(set-logic %s)\n' % self.logic
		for n,ty in self.cnst_decls.items():
			r += '(declare-fun %s () %s)\n' % (n,ty)
		for n,(ty,tm) in self.cnst_defs.items():
			r += '(define-fun %s () %s %s)\n' % (n,ty,tm)
		for tm in self.assert_terms:
			r += '(assert %s)\n' % tm
		r += '(check-sat)\n'
		if self.need_model:
			r += '(get-model)\n'
		r += '(exit)'

		return r.encode()

class Item:
	# 'reply' type probably asyncio.Future, but depends on the event
	# loop implementation
	def __init__(self, id : Any, instance : Smtlib2, reply : Any):
		self.id = id
		self.instance = instance
		self.reply = reply
		self.reply.handle_smtlib_replies = self.handle_smtlib_replies

	def handle_smtlib_replies(self, parser):
		sat_res = parser.sat_result()
		model = parser.model() if self.instance.need_model else None
		logging.info('result for id %s is %s with model %s',
		             self.id, sat_res, model)
		res = { b'sat'    : lambda: SAT(model),
		        b'unsat'  : lambda: UNSAT(),
		        b'unknown': lambda: None
		      }[sat_res]()

		if res is None:
			self.reply.cancel()
		else:
			self.reply.set_result(res)

@dataclass(order=True)
class PrItem:
	priority : Any
	item     : Item=field(compare=False)

class TestPool(Pool):
	def __init__(self, loop):
		self.heap = []
		self.any   = asyncio.Event()
		self.empty = loop.create_future()

	async def pop(self):
		while True:
			await self.any.wait()
			assert len(self.heap) > 0
			item = heapq.heappop(self.heap).item
			if len(self.heap) == 0:
				self.any.clear()
			if item.reply.done():
				continue
			return item
		# notify self.wait_empty()
		#self.empty.set_result(None)

	async def wait_empty(self):
		await self.empty

	async def solve(self, prio, prid, instance : Smtlib2):
		loop = asyncio.get_event_loop()
		fut = loop.create_future()
		item = Item(prid, instance, fut)
		heapq.heappush(self.heap, PrItem(prio, item))
		self.any.set()
		res = await fut
		return res

async def run_solver(host, port, solve_specific):
	loop = asyncio.get_event_loop()
	pool = TestPool(loop)
	async with await server2(host, port, StoredPool(pool, dict())):
		await solve_specific(pool)

__all__ = [run_solver.__name__
          ,SAT.__name__
          ,UNSAT.__name__
          ,Smtlib2.__name__
          ]
