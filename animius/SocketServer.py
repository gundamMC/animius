import socket
import threading

from .SocketServerModel import Client

clients = {}


def new_client(c, console):
    try:
        print('Establishing connection with: {0}:{1}'.format(c.addr, c.port))
        c.initRandomAEScipher()
        c.sendWithoutAes(0, 200, 'InitAes', {'key': c.AEScipher.getKey(), 'iv': c.AEScipher.getIv()})
        while True:
            req = c.recv()
            response = console.handle_network(req)
            c.send(*response)
    except socket.error as error:
        print('Socket error from {0}: {1]'.format(c.addr, error))
    except Exception as error:
        print('Unexpected exception from {0}: {0}'.format(c.addr, error))
    finally:
        print('Closing connection with {0}:{1}'.format(c.addr, c.port))
        c.close()


def start_server(console, port, local=True, max_clients=10):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if local:
        host = '127.0.0.1'
    else:
        host = socket.gethostname()
    server.bind((host, port))

    # 侦听客户端
    server.listen(max_clients)
    print('Sever started. Listening on {0}:{1}'.format(host, port))

    while True:
        # 接受客户端连接
        conn, addr = server.accept()
        c = Client(conn, addr)
        t = threading.Thread(target=new_client, args=(c, console))
        t.start()
