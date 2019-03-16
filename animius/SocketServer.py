import socket
import threading

from .SocketServerModel import Client

clients = {}


def new_client(c, console, event):

    # check if event is set (then this client is probably the 'fake' one from stop())
    if event.is_set():
        return

    try:
        print('Establishing connection with: {0}:{1}'.format(c.address, c.port))
        # initialize AES
        # c.initRandomAEScipher()
        # send AES keys to client
        # c.sendWithoutAes(0, 200, 'InitAes', {'key': c.AEScipher.getKey(), 'iv': c.AEScipher.getIv()})

        # check for password
        if c.pwd != '':
            recvPwd = c.recv_pass()
            if recvPwd != c.pwd:
                # wrong password, close connection
                c.close()

        # password verified and connected
        c.send('', 0, 'success', {})

        while True:
            req = c.recv()
            response = console.handle_network(req)
            c.send(*response)

    except socket.error as error:
        print('Socket error from {0}: {1]'.format(c.address, error))
    except Exception as error:
        print('Unexpected exception from {0}: {1}'.format(c.address, error))
    finally:
        print('Closing connection with {0}:{1}'.format(c.address, c.port))
        c.close()


def start_server(console, port, local=True, pwd='', max_clients=10):
    thread = _ServerThread(console, port, local, pwd, max_clients)
    thread.start()
    return thread


class _ServerThread(threading.Thread):
    def __init__(self, console, port, local=True, pwd='', max_clients=10):
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
        self.pwd = pwd
        self.max_clients = max_clients

    def run(self):
        # Start Listening
        self.server.listen(self.max_clients)

        while not self.event.is_set():
            # Accept Connection
            conn, addr = self.server.accept()
            c = Client(conn, addr, self.pwd)
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
