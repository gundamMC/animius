import json
import struct

clients = {}


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
        print(resp)
        self._send(resp)

    def recv(self):
        req = self._recv()
        req = req.decode()
        print(req)
        req = Request.initFromReq(req)
        return req

    def recv_pass(self):
        req = self._recv()
        req = req.decode()
        return req

    def close(self):
        self.socket.close()
