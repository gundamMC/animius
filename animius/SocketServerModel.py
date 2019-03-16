import base64
import json

# from Crypto.Cipher import AES
# from Crypto.Random import get_random_bytes
# from Crypto.Util import Padding

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

# ============================
# Encryption is no longer used
# ============================

# class AEScipher:
#     def __init__(self, key, iv):
#         self.key = key
#         self.iv = bytes([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
#         self.blocksize = AES.block_size
#         # use AES CBC
#         # self.cipher = AES.new(self.key, AES.MODE_CBC, iv=self.iv)
#
#         # use AES CTR
#         # self.cipher = AES.new(self.key, AES.MODE_CTR, nonce=self.iv)
#
#     def getKey(self):
#         return base64.b64encode(self.key).decode('utf-8')
#
#     def getIv(self):
#         return base64.b64encode(self.iv).decode('utf-8')
#
#     @classmethod
#     def generateRandom(cls):
#         iv = get_random_bytes(16)
#         print('iv ', iv)
#         key = get_random_bytes(16)
#         print('key ', key)
#         return cls(key, iv)
#
#     # AES encrypt
#     def encrypt(self, data):
#         # padded_data = Padding.pad(data, self.blocksize, style='pkcs7')
#         encrData = self.cipher.encrypt(data)
#         # encrData = base64.b64encode(encrData)
#
#         # see decrypt for details
#         self.cipher = AES.new(self.key, AES.MODE_CTR, nonce=self.iv)
#
#         return encrData
#
#     # AES decrypt
#     def decrypt(self, encrypted_byte):
#         # decrData = encrypted_byte.decode('utf-8')
#         # decrData = base64.b64decode(decrData)
#         decrData = self.cipher.decrypt(encrypted_byte)
#         print(decrData)
#         decrData = decrData.decode()
#         # decrData = Padding.unpad(decrData, self.blocksize, style='pkcs7')
#
#         # https://stackoverflow.com/questions/54082280/typeerror-decrypt-cannot-be-called-after-encrypt
#         # en/decrypt cipher object is stateful. Create new object for next operation
#         self.cipher = AES.new(self.key, AES.MODE_CTR, nonce=self.iv)
#
#         return decrData


class Client:
    def __init__(self, socket, address, pwd):
        self.address = address[0]
        self.port = address[1]
        self.pwd = pwd
        self.socket = socket
        # self.AEScipher = None

    # def initAEScipher(self, key, iv):
    #     try:
    #         self.AEScipher = AEScipher(key, iv)
    #         return True
    #     except:
    #         return False

    # def initRandomAEScipher(self):
    #     try:
    #         # self.AEScipher = AEScipher.generateRandom()
    #         return True
    #     except:
    #         return False

    def _send(self, data):
        self.socket.send(data)

    def _recv(self, mtu=65535):
        return self.socket.recv(mtu)

    def send(self, id, status, message, data):
        resp = Response.createResp(id, status, message, data)
        # resp = self.AEScipher.encrypt(resp)
        self._send(resp)

    # def sendWithoutAes(self, id, status, message, data):
    #     try:
    #         resp = Response.createResp(id, status, message, data)
    #         self._send(resp)
    #         return True
    #     except:
    #         return False

    def recv(self):
        try:
            req = self._recv()
            # req = self.AEScipher.decrypt(req)
            req = req.decode()
            req = Request.initFromReq(req)
            return req
        except:
            return None

    def recv_pass(self):
        req = self._recv()
        req = req.decode()

        return req

    def close(self):
        self.socket.close()
