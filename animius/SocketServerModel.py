import json

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
        print(password,4)
    def _send(self, data):
        self.socket.send(data)

    def _recv(self, mtu=65535):
        return self.socket.recv(mtu)

    def send(self, id, status, message, data):
        resp = Response.createResp(id, status, message, data)
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
