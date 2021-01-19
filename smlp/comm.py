
import socket, asyncio, selectors, functools, os, signal, logging
import time, platform
from asyncio.subprocess import PIPE
from concurrent.futures import CancelledError as cf_CancelledError

try:
	# new in python-3.7
	from asyncio.exceptions import CancelledError as as_CancelledError
except:
	class as_CancelledError(BaseException):
		pass

from subprocess import CalledProcessError

from .protocol import pb_pb2 as proto

# semantics of asyncio.get_event_loop() (and other async-related things)
# are different for <python-3.6
assert platform.python_version_tuple() >= ('3','6')

VERSION = 1

def fmt_address(sockaddr):
	return '%s:%s' % socket.getnameinfo(sockaddr, 0)

# self.handle_request(conn, id, req)
class Connection:
	def __init__(self, rd, wr, handle_request):
		super().__init__()
		self._pending = {}
		self._wr = wr
		self._reqs = 0
		self.log = logging.getLogger(fmt_address(wr.transport.get_extra_info('peername')))

		loop = asyncio.get_event_loop()

		async def rd_msg_dispatch():
			while not rd.at_eof():
				self.log.debug('rd_msg_dispatch 1')
				n = await rd.read(4)
				if len(n) == 0:
					logging.warning('rd_msg_dispatch: eof, why?')
					break
				self.log.debug('rd_msg_dispatch 2: %s', n)
				n = int.from_bytes(n, 'big')
				msg = await rd.read(n)
				if len(msg) != n:
					self.log.critical('short read, possibly not synchronized comms, aborting...')
					break
				s = proto.Smlp.FromString(msg)
				self.log.debug('rd_msg_dispatch 3: %s', s)
				assert s.version == VERSION
				if s.HasField('reply'):
					fut = self._pending[s.msg_id]
					del self._pending[s.msg_id]
					try:
						fut.handle_recv(s)
					except BaseException as ex:
						fut.set_exception(ex)
				if s.HasField('request'):
					loop.create_task(handle_request(self, s.msg_id, s.request))
			self.log.info('rd_msg_dipatch fini')

		self._rd_msg_task = loop.create_task(rd_msg_dispatch())

	@property
	def rd_msg_task(self):
		return self._rd_msg_task

	def close(self):
		self._rd_msg_task.cancel()
		self._wr.close()

	async def __aenter__(self):
		return self

	async def __aexit__(self, exc_type, exc_value, traceback):
		self.close()
		try:
			await self._wr.wait_closed()
		except AttributeError:
			# < python-3.7
			pass

	def _wr_msg_nowait(self, msg):
		self._wr.write(len(msg).to_bytes(4, 'big'))
		self._wr.write(msg)
		self.log.debug('sent msg %s', msg)

	async def wait_wr_drain(self):
		await self._wr.drain()

	async def _wr_msg(self, msg):
		self._wr_msg_nowait(msg)
		await self.wait_wr_drain()

	async def wait_send_reply(self, msg_id, rep):
		await self._wr_msg(proto.Smlp(version=VERSION, msg_id=msg_id, reply=rep)
		                             .SerializeToString())

	def _next_id(self):
		msg_id = self._reqs
		self._reqs += 1
		return msg_id

	# this is dangerous: can lead to blow up of write buffer if not occasionally
	# .wait_wr_drain()
	def _send_request(self, req, fut=None):
		loop = asyncio.get_event_loop()
		if fut is None:
			fut = loop.create_future()
			fut.handle_reply = fut.set_result
		msg_id = self._next_id()
		self._pending[msg_id] = fut
		s = proto.Smlp(version=VERSION, msg_id=msg_id, request=req)
		self._wr_msg_nowait(s.SerializeToString())

		def _request_handle_reply(msg_id, fut, res):
			self.log.debug('request got reply: %s', res)
			assert res.version == VERSION
			assert res.msg_id == msg_id
			assert res.HasField('reply')
			assert not res.HasField('request')
			fut.handle_reply(res.reply)

		fut.handle_recv = functools.partial(_request_handle_reply, msg_id, fut)

		return fut, msg_id

	async def wait_send_request(self, req, fut=None):
		fut, msg_id = self._send_request(req, fut=fut)
		await self.wait_wr_drain()
		return fut, msg_id

	# returns an asyncio.Future representing the state of the incoming reply
	# to the request with message id `msg_id`
	def get_pending(self, msg_id):
		return self._pending.get(msg_id)

	async def wait_pending(self):
		v = self._pending.values()
		if len(v) > 0:
			await asyncio.wait(v)

	async def request_wait_reply(self, req, fut=None):
		fut, msg_id = await self.wait_send_request(req, fut=fut)
		await fut
		reply = fut.result()
		return reply
	#	#s = fut.result()
	#	#self.log.info('request got reply: %s', s)
	#	#assert s.version == VERSION
	#	#assert s.msg_id == msg_id
	#	#assert s.HasField('reply')
	#	#assert not s.HasField('request')
	#	#return s.reply

class SaneStdoutParser:
	def __init__(self, replies):
		self.replies = replies

	def sat_result(self):
		return next(self.replies)

	def model(self):
		return next(self.replies)

	def remaining(self):
		return list(self.replies)

class YicesStdoutParser(SaneStdoutParser):
	def model(self):
		return list(self.replies)

def handle_smtlib_stdout(fut, prid, parser, stdout):
	logging.debug('stdout for id %s: %s', prid, stdout)
	a = time.perf_counter()
	p = parser(filter(lambda w: w != b'unsupported',
	                  smtlib_script_parse(stdout)))
	fut.handle_smtlib_replies(p)
	assert len(p.remaining()) == 0
	logging.info('parsed replies in %g sec', time.perf_counter()-a)

def handle_sane_command(conn, script, fut, cmd):
	logging.debug('handle_sane_command')
	if cmd.status == 0:
		fut.handle_stdout(cmd.stdout)
	else:
		raise CalledProcessError(returncode=cmd.status,
		                         cmd=(conn, script),
		                         output=cmd.stdout,
		                         stderr=cmd.stderr)

def handle_z3_command(conn, script, fut, cmd):
	logging.debug('handle_z3_command')
	if cmd.status == 1 and cmd.stderr == b'':
		try:
			fut.handle_stdout(cmd.stdout)
			return
		except BaseException as e:
			conn.log.exception('z3 exit code 1 handler: %s', e)
	handle_sane_command(conn, script, fut, cmd)

def handle_reply(fut, rep):
	assert rep.type == rep.Type.SMTLIB_REPLY
	fut.handle_command(rep.cmd)

async def smtlib_script_request(conn, script, fut, handle_command = handle_sane_command):
	req = proto.Request(stdin=script)
	req.type = req.Type.SMTLIB_SCRIPT

	fut.handle_reply = functools.partial(handle_reply, fut)
	fut.handle_command = functools.partial(handle_command, conn, script, fut)

	fut, _ = await conn.wait_send_request(req, fut=fut)

	return fut

async def smtlib_script_wait_request(conn, script, fut=None):
	if fut is None:
		fut = asyncio.get_event_loop().create_future()
		fut.handle_stdout = fut.set_result
	fut = await smtlib_script_request(conn, script, fut=fut)
	return await fut

#	rep = await conn.request_wait_reply(req, fut=fut)
#	assert rep.type == rep.Type.SMTLIB_REPLY
#	if rep.cmd.status == 0:
#		return rep.cmd.stdout
#	raise CalledProcessError(returncode=rep.cmd.status, cmd=(conn, script),
#	                         output=rep.cmd.stdout, stderr=rep.cmd.stderr)

async def smtlib_name(conn):
	return await smtlib_script_wait_request(conn, b'(get-info :name)')

async def smtlib_version(conn):
	return await smtlib_script_wait_request(conn, b'(get-info :version)')


class Pool:
	# returns either a pair (prid,instance) or None to signify end-of-problem
	async def pop(self):
		pass

	async def wait_empty(self):
		pass


class ConnectedWorker:
	def __init__(self, worker):
		self.conn = worker



def smtlib_tokenize(s):
	i = 0
	while i < len(s):
		c = s[i:i+1]
		if c in b' \t\r\n\f\v':
			i += 1
		elif c in b'()':
			yield c
			i += 1
		elif c == b'"':
			j = i + 1
			while True:
				assert j < len(s)
				if s[j:j+1] == b'"':
					break
				j += 1
			j += 1
			yield s[i:j]
			i = j
		else:
			j = i + 1
			while True:
				assert j < len(s)
				if s[j:j+1] in b' \t\r\n\f\v()"':
					break
				j += 1
			yield s[i:j]
			i = j

def smtlib_script_parse_sexpr1(toks):
	try:
		s = []
		while True:
			t = next(toks)
			if t == b')':
				break
			s.append(t if t != b'(' else smtlib_script_parse_sexpr1(toks))
		return s
	except StopIteration:
		assert False

def smtlib_script_parse(s):
	toks = smtlib_tokenize(s)
	for t in toks:
		yield t if t != b'(' else smtlib_script_parse_sexpr1(toks)

class Server:
	def __init__(self, pool):
		self._pool = pool

	# connection is closed on return
	async def feed(self, worker):
		#n, v = await asyncio.gather(smtlib_name(worker), smtlib_version(worker))
		#n = await smtlib_name(worker)
		#v = await smtlib_version(worker)
		#logging.info('client runs %s v%s', n, v)
		pong = await worker.request_wait_reply(proto.Request(type=proto.Request.Type.PING))
		if pong.type != proto.Reply.Type.PONG:
			worker.log.critical('worker does not reply to ping, disconnecting')
			return

		try:
			nv = await smtlib_script_wait_request(worker,
			                                      b'(get-info :name)(get-info :version)')
		except CalledProcessError as e:
			worker.log.critical('worker\'s SMT command fails smtlib2 sanity check, ' +
			                    'disconnecting: ' +
			                    'exited with %d on input %s, stdout: %s, stderr: %s',
			                    e.returncode, e.cmd[1], e.stdout, e.stderr)
			await worker.wait_send_request(
					proto.Request(type=proto.Request.Type.CLIENT_QUIT))
			return
		smtlib_nv = [w for w in smtlib_script_parse(nv)]
		worker.log.info('client runs %s version %s',
		                smtlib_nv[0][1].decode(),
		                smtlib_nv[1][1].decode())

		handle_command = (handle_z3_command if smtlib_nv[0][1] == b'"Z3"'
		                  else handle_sane_command)

		stdout_parser = (YicesStdoutParser if smtlib_nv[0][1] == b'"Yices"'
		                 else SaneStdoutParser)

		while True:
			pr = await self._pool.pop()
			if pr is None:
				break
			worker.log.info('submitting instance %s', pr.id)
			pr.reply.handle_stdout = functools.partial(
				handle_smtlib_stdout, pr.reply, pr.id,
				stdout_parser)
			fut = await smtlib_script_request(worker, pr.instance.format(),
			                                  fut=pr.reply,
			                                  handle_command=handle_command)
			# workers can only handle one smtlib script request at
			# any point in time, so we can just as well wait for its
			# result here before submitting the next instance
			try:
				res = await fut
				worker.log.info('got result %s for instance %s', res, pr.id)
			except:
				worker.log.exception('error computing instance %s', pr.id)

		worker.log.info('pool empty, closing connection')

	async def accepted(self, rd, wr):
		claddr = fmt_address(wr.transport.get_extra_info('peername'))
		logging.info('client %s accepted', claddr)
		try:
			async with Connection(rd, wr, self.handle_request) as worker:
				await self.feed(worker)
				await worker.wait_pending()
			#await wr.wait_closed()
		except ConnectionResetError as e:
			logging.warning('connection lost to client %s', claddr)
		#except as_CancelledError:
		#	raise
		#except:
		#	logging.exception('server accepted exception')
		#	raise
		finally:
			logging.info('done with client %s', claddr)

	def handle_request(self, conn, msg_id, req):
		conn.log.error('unhandled request: %s', req)
		r = proto.Reply()
		r.type = r.Type.ERROR
		r.code = r.Code.UNKNOWN_REQ
		conn.reply(msg_id, r)


class Client:
	def __init__(self, args):
		self.args = args;
		self._working = False

	async def connected(self, rd, wr):
		tp = wr.transport
		logging.info('client %s connected to %s',
		             fmt_address(tp.get_extra_info('sockname')),
		             fmt_address(tp.get_extra_info('peername')))
		await Connection(rd, wr, self.handle_request).rd_msg_task

	async def handle_request(self, conn, msg_id, req):
		conn.log.info('got request %s', req)
		r = proto.Reply()
		a = time.perf_counter()

		if req.type == req.Type.PING:
			r.type = r.Type.PONG
			r.code = r.Code.BUSY if self._working else r.Code.IDLE

		elif req.type == req.Type.CLIENT_QUIT:
			conn.close()
			return

		elif req.type == req.Type.SMTLIB_SCRIPT:

			if self._working:
				r.type = r.Type.ERROR
				r.code = r.Code.BUSY
				r.error_msg = 'busy'

			else:
				self._working = True
				r.type = r.Type.SMTLIB_REPLY
				proc = await asyncio.create_subprocess_exec(*self.args,
				                                            stdin=PIPE,
				                                            stdout=PIPE,
				                                            stderr=PIPE)
				r.cmd.stdout, r.cmd.stderr = await proc.communicate(req.stdin)
				r.cmd.status = proc.returncode
				self._working = False

		else:
			r.type = r.Type.ERROR
			r.code = r.Code.UNKNOWN_REQ
			r.error_msg = 'request not understood'

		conn.log.info('handling request took %gs', time.perf_counter() - a)
		await conn.wait_send_reply(msg_id, r)


# precondition: pool not empty
async def server(host, port, pool):
	server = await asyncio.start_server(Server(pool).accepted, host, port)
	logging.info('server listening on %s',
	             ', '.join(map(lambda s: fmt_address(s.getsockname()),
	                           server.sockets)))
	#async with server:
	#	await server.serve_forever()
	await pool.wait_empty()
	server.close()
	await server.wait_closed()

async def server2(host, port, pool):
	server = await asyncio.start_server(Server(pool).accepted, host, port)
	logging.info('server listening on %s',
	             ', '.join(map(lambda s: fmt_address(s.getsockname()),
	                           server.sockets)))
	return server
	#async with server:
	#	await server.serve_forever()
	await pool.wait_empty()
	server.close()
	await server.wait_closed()

async def client(host, port, args):
	logging.info('client args: %s', args)
	try:
		#import z3
		rd, wr = await asyncio.open_connection(host, port)
		await Client(args).connected(rd, wr)
	except OSError:
		logging.error('error connecting to %s:%d', host, port)
		raise
	except ConnectionRefusedError:
		logging.error('error connecting to %s:%d: connection refused', host, port)
		raise


def run1(task, loop=None):
	def cancel_all_tasks(loop):
		try:
			# >= python-3.7
			all_tasks = asyncio.all_tasks
		except AttributeError:
			all_tasks = asyncio.Task.all_tasks
		for task in all_tasks(loop=loop):
			task.cancel()

	def sighandler(sig, loop):
		logging.critical('got signal %s, terminating...', sig.name)
		cancel_all_tasks(loop)

	if loop is None:
		loop = asyncio.get_event_loop()

	for sig in map(lambda n: getattr(signal, n), ('SIGINT','SIGTERM')):
		loop.add_signal_handler(sig, functools.partial(sighandler, sig, loop))

	try:
		loop.run_until_complete(task)
		ret = 0
	except as_CancelledError:
		# from client
		logging.info('as cancelled, aborting...')
		ret = 3
	except cf_CancelledError:
		# from sighandler
		logging.info('cf cancelled, aborting...')
		ret = 3
	except:
		logging.exception("error")
		ret = 4
	finally:
		# workaround server's accept task not being cleaned up on cancel leading
		# to:
		#   ERROR:asyncio:Task was destroyed but it is pending!
		#   task: <Task pending coro=<Server.accepted() running at async.py:222>
		#          wait_for=<Future pending cb=[<TaskWakeupMethWrapper object at
		#                                        0x7f7c9d02f7c8>()]>>
		cancel_all_tasks(loop)

		loop.run_until_complete(loop.shutdown_asyncgens())
		#loop.close()

	return ret

__all__ = [ Pool.__name__
          , server2.__name__
          , client.__name__
          , run1.__name__
          , smtlib_script_parse.__name__
          , 'VERSION'
          ]
