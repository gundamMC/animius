import base64
import json

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util import Padding

clients = {}


class Request:
    def __init__(self, id, command, arguments):
        self.id = id
        self.command = command
        self.arguments = arguments

    @classmethod
    def initFromReq(cls, req):
        try:
            data = json.loads(req)
            return cls(data["id"], data["command"], data["arguments"])
        except:
            return None


class Response:

    @staticmethod
    def createResp(response_id, status, message, data):
        resp = {"id": response_id,
                "status": status,
                "message": message,
                "data": data
                }
        return json.dumps(resp).encode("utf-8")


class AEScipher:
    def __init__(self, key, iv):
        self.key = key
        self.iv = iv
        self.blocksize = AES.block_size
        # use AES CBC
        self.cipher = lambda: AES.new(self.key, AES.MODE_CBC, iv=self.iv)

    def getKey(self):
        return base64.b64encode(self.key).decode()

    def getIv(self):
        return base64.b16encode(self.iv).decode()
    @classmethod
    def generateRandom(cls):
        iv = get_random_bytes(16)
        key = get_random_bytes(16)
        return cls(key, iv)

    # AES encrypt
    def encrypt(self, data):
        try:
            data = data
            padded_data = Padding.pad(data, self.blocksize, style='pkcs7')
            encrData = self.cipher().encrypt(padded_data)
            return base64.b64encode(encrData)
        except:
            return None

    # AES decrypt
    def decrypt(self, encrData):
        try:
            decrData = self.cipher().decrypt(base64.b64decode(encrData))
            decrData = Padding.unpad(decrData, self.blocksize, style='pkcs7')
            return decrData
        except:
            return None


class Client:
    def __init__(self, socket, addr):
        self.addr = addr[0]
        self.port = addr[1]
        self.socket = socket
        self.AEScipher = None

    def initAEScipher(self, key, iv):
        try:
            self.AEScipher = AEScipher(key, iv)
            return True
        except:
            return False

    def initRandomAEScipher(self):
        try:
            self.AEScipher = AEScipher.generateRandom()
            return True
        except:
            return False

    def _send(self, data):
        self.socket.send(data)

    def _recv(self, mtu=65535):
        return self.socket.recv(mtu)

    def send(self, id, status, message, data):
        try:
            resp = Response.createResp(id, status, message, data)
            resp = self.AEScipher.encrypt(resp)
            self._send(resp)
            return True
        except:
            return False
            
    def sendWithoutAes(self, id, status, message, data):
        try:
            resp = Response.createResp(id, status, message, data)
            self._send(resp)
            return True
        except:
            return False
            
    def recv(self):
        try:
            req = self.AEScipher.decrypt(self._recv())
            req = Request.initFromReq(req.decode("utf-8"))
            return req
        except:
            return None
