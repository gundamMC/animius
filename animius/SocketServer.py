import asyncio
import json
import struct


class SocketServer:
    def __init__(self, console, port, local, pwd, max_clients):
        self.console = console
        self.host = '127.0.0.1' if local else '0.0.0.0'
        self.port = port
        self.pwd = pwd
        self.max_clients = max_clients
        self.server = None

    def start_server(self):
        asyncio.run(self.main())

    def stop_server(self):
        self.server._shutdown_request = True

    @staticmethod
    def create_response(response_id, status, message, data):
        response = {"id": response_id,
                    "status": status,
                    "message": message,
                    "data": data
                    }
        return json.dumps(response).encode("utf-8")

    @staticmethod
    def parse_request(request):
        request = request.decode()
        data = json.load(request)
        return data["id"], data["command"], data["arguments"]

    @staticmethod
    async def await_write(writer, response):
        writer.write(struct.pack('!I', len(response)))
        writer.write(response)
        await writer.drain()

    @staticmethod
    async def await_receive(reader):
        length_buf = reader.read(4)
        length, = struct.unpack('!I', length_buf)
        return await reader.read(length)

    async def await_auth(self, reader):
        auth_info = await SocketServer.await_receive(reader)
        request_id, command, arguments = SocketServer.parse_request(auth_info)
        if command == 'login' and arguments['pwd'] == self.pwd:
            return True
        else:
            return False

    async def handle_connection(self, reader, writer):
        address = writer.get_extra_info('peername')
        valid_session = await self.await_auth(reader)

        while valid_session:
            raw_request = await SocketServer.await_receive(reader)
            request_id, command, arguments = SocketServer.parse_request(raw_request)
            request_id, status, message, data = self.console.thread_pool.submit(
                self.console.handle_network(request_id, command, arguments)
            )

            response = SocketServer.create_response(request_id, status, message, data)
            await SocketServer.await_write(writer, response)

        writer.close()

    async def main(self):
        self.server = await asyncio.start_server(self.handle_connection, self.host, self.port)
        await self.server.serve_forever()

