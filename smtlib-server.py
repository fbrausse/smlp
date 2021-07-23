
import selectors
import socket

sel = selectors.DefaultSelector()

clients = []

class smtlib_io:
	

def accept(sock, mask):
	conn, addr = sock.accept()  # Should be ready
	conn.send('(get-info :name)'
	
	print('accepted', conn, 'from', addr)
	conn.setblocking(False)
	sel.register(conn, selectors.EVENT_READ, read)

def read(conn, mask):
	data = conn.recv(1000)  # Should be ready
	if data:
		print('echoing', repr(data), 'to', conn)
		conn.send(data)  # Hope it won't block
	else:
		print('closing', conn)
		sel.unregister(conn)
		conn.close()

sock = socket.socket()
sock.bind(('localhost', 1234))
sock.listen()
sock.setblocking(False)
sel.register(sock, selectors.EVENT_READ, accept)

while True:
	events = sel.select()
	for key, mask in events:
		callback = key.data
		callback(key.fileobj, mask)
