
import socket, asyncio, selectors, functools, os, signal, argparse, sys, logging
import protocol.pb_pb2 as proto

VERSION = 1

class smtlib_io:
	def __init__(self, rd, wr):
		self.rd = rd
		self.wr = wr
		self.pending = {}
		self.reqs = 0

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.wr.close()

	async def rd_msg_dispatch(self):
		logging.info('rd_msg_dispatch 1')
		n = await self.rd.read(4)
		logging.info('rd_msg_dispatch 2: %s', n)
		msg = await self.rd.read(int.from_bytes(n, 'big'))
		s = proto.Smlp.FromString(msg)
		logging.info('rd_msg_dispatch 3: %s', s)
		fut = self.pending[s.msg_id]
		del self.pending[s.msg_id]
		fut.set_result(s)

	async def wr_msg(self, msg):
		self.wr.write(len(msg).to_bytes(4, 'big'))
		self.wr.write(msg)
		logging.info('wrote msg %s to client', msg)
		await self.wr.drain()

	async def request(self, req):
		loop = asyncio.get_event_loop()
		task = loop.create_task(self.rd_msg_dispatch())
		fut = loop.create_future()
		id = self.reqs;
		self.pending[id] = fut
		s = proto.Smlp(version=VERSION, msg_id=id, request=req)
		self.reqs += 1
		await self.wr_msg(s.SerializeToString())
		await fut
		s = fut.result()
		logging.info('request got reply: %s', s)
		assert s.version == VERSION
		assert s.msg_id == id
		assert s.reply is not None
		return s.reply

	@property
	async def name(self):
		req = proto.Request(smt_command=b'(get-info :name)')
		req.type = req.Type.SMT_COMMAND
		rep = await self.request(req)
		return rep.smt_reply

	@property
	async def version(self):
		req = proto.Request(smt_command=b'(get-info :version)\n')
		req.type = req.Type.SMT_COMMAND
		rep = await self.request(req)
		return rep.smt_reply



# connection is closed on return
async def serve_client(smt):
	name = await smt.name
	vers = await smt.version
	logging.info('client runs %s v%s', name, vers)

async def client_handle_request(req):
	r = proto.Reply()
	r.type = proto.Reply.Type.SMT_REPLY
	r.status = 0
	r.smt_reply = 'myself' if ":name" in req.smt_command else '42.-1'
	return r;

# connection is closed on return
async def client_connected(smt):
	logging.info('connected, appearently')
	while not smt.rd.at_eof():
		n = await smt.rd.read(4)
		logging.info('received n: %s', n)
		if len(n) == 0:
			logging.warning('eof, why?')
			break
		n = int.from_bytes(n, 'big')
		msg = await smt.rd.read(n)
		if len(msg) != n:
			logging.critical('short read, possibly not synchronized comms, aborting...')
			break
		logging.info('received msg: %s', msg)
		s = proto.Smlp.FromString(msg)
		assert s.version == VERSION
		await smt.wr_msg(proto.Smlp(version=VERSION, msg_id=s.msg_id,
		                            reply=await client_handle_request(s.request))
		                           .SerializeToString())
		logging.error("unhandled message '%s'", s)

async def server(host, port):
	async def cc(rd, wr):
		logging.info('client connected')
		try:
			with smtlib_io(rd, wr) as smt:
				await serve_client(smt)
			#await wr.wait_closed()
		except ConnectionResetError as e:
			logging.warning('connection lost')
		finally:
			logging.info('done')

	server = await asyncio.start_server(cc, host, port, family=socket.AF_INET)
	print('server listening on', ', '.join(map(lambda s: '%s:%d' % s.getsockname(),
	                                           server.sockets)))
	#async with server:
	#	await server.serve_forever()

async def client(host, port):
	import z3

	rd, wr = await asyncio.open_connection(host, port)
	with smtlib_io(rd, wr) as smt:
		await client_connected(smt)

HOST = 'localhost'
PORT = 1337

def parse_args(argv):
	class LogLevel(argparse.Action):
		def __call__(self, parser, namespace, values, option_string=None):
			l = getattr(logging, values.upper(), None)
			if not isinstance(l, int):
				raise ValueError('Invalid log level: %s' % values)
			logging.basicConfig(level=l)

	p = argparse.ArgumentParser(prog=argv[0])
	p.add_argument('-c', '--client', default=False, action='store_true',
	               help='start client mode')
	p.add_argument('-H', '--host', default=HOST, type=str)
	p.add_argument('-P', '--port', default=PORT, type=int)
	p.add_argument('-v', '--log-level', metavar='LVL', type=str, action=LogLevel)
	args = p.parse_args(argv[1:])
	return args

def sighandler(loop, sig):
	logging.critical('got signal %s, terminating...', sig.name)
	loop.stop()

if __name__ == "__main__":
	try:
		args = parse_args(sys.argv)
	except ValueError as e:
		print('error:', e, file=sys.stderr)
		sys.exit(1)

	loop = asyncio.get_event_loop()
	for sig in map(lambda n: getattr(signal, n), ('SIGINT','SIGTERM')):
		loop.add_signal_handler(sig, functools.partial(sighandler, loop, sig))

	try:
		if args.client:
			loop.run_until_complete(client(args.host, args.port))
		else:
			loop.run_until_complete(server(args.host, args.port))
			loop.run_forever()
	finally:
		loop.run_until_complete(loop.shutdown_asyncgens())
		loop.close()
