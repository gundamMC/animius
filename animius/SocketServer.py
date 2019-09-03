import json
import socket
import struct
import threading

clients = []


def new_client(c, console, event):
    # check if event is set (then this client is probably the 'fake' one from stop())
    if event.is_set():
        return

    print('Establishing connection with: {0}:{1}'.format(c.address, c.port))

    clients.append(c)
    # check for password
    if c.password != '':
        recvPassword = c.recv_pass()
        print(recvPassword)
        recvPassword = json.loads(recvPassword)
        if recvPassword['command'] == 'login' and recvPassword['arguments']['pwd'] != c.password:
            # wrong password, close connection
            c.close()
            return None

    # password verified and connected
    c.send('', 0, 'success', {})

    queue = console.queue[1]
    sendThread = threading.Thread(target=send_queue, args=(c, queue,))
    sendThread.start()

    while True:

        req = c.recv()
        if req is None or req is "":
            continue
        print(req)

        if req['command'] == 'logout':
            break
        elif req['command'] != 'login':
            console.queue[0].put(req)


def send_queue(client, queue):
    while True:
        if not queue.empty():
            result = queue.get()
            client.send(result['id'], result['status'], result['result'], result['data'])
            print(result)
            queue.task_done()


class _ServerThread(threading.Thread):
    def __init__(self, console, port, local=True, password='', max_clients=10):
        super(_ServerThread, self).__init__()
        self.event = threading.Event()

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.port = port

        if local:
            self.host = '127.0.0.1'
        else:
            self.host = socket.gethostname()
        self.server.bind((self.host, port))

        self.console = console
        self.password = password
        self.max_clients = max_clients

    def run(self):
        # Start Listening
        self.server.listen(self.max_clients)

        while not self.event.is_set():
            # Accept Connection
            conn, addr = self.server.accept()
            c = Client(conn, addr, self.password)
            t = threading.Thread(target=new_client, args=(c, self.console, self.event))
            t.start()

        # close server
        self.server.close()
        print('Server closed')

    def stop(self):
        self.event.set()
        # let the while loop in run() know to stop

        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((self.host, self.port))
        # send a fake client to let run() move on from self.server.accept()


class Request:
    def __init__(self, id, command, arguments):
        self.id = id
        self.command = command
        self.arguments = arguments

    @classmethod
    def initFromReq(cls, req):
        data = json.loads(req)
        return cls(data["id"], data["command"], data["arguments"])


class Response:

    @staticmethod
    def createResp(response_id, status, message, data):
        resp = {"id": response_id,
                "status": status,
                "message": message,
                "data": data
                }
        return json.dumps(resp).encode("utf-8")


class Client:
    def __init__(self, socket, address, password):
        self.address = address[0]
        self.port = address[1]
        self.password = password
        self.socket = socket

    def _send(self, data):
        length = len(data)
        self.socket.sendall(struct.pack('!I', length))
        self.socket.sendall(data)

    def _recvall(self, count):
        buf = b''
        while count:
            newbuf = self.socket.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def _recv(self):
        lengthbuf = self._recvall(4)
        length, = struct.unpack('!I', lengthbuf)
        return self._recvall(length)

    def send(self, id, status, message, data):
        resp = Response.createResp(id, status, message, data)
        print('socket_send', resp)
        self._send(resp)

    def recv(self):
        req = self._recv()
        req = req.decode()
        print('socket_recv', req)
        req = Request.initFromReq(req)
        return req

    def recv_pass(self):
        req = self._recv()
        req = req.decode()
        return req

    def close(self):
        self.socket.close()
