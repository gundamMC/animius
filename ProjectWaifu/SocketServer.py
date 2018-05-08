import socket
import ProjectWaifu.Console as Console
import sys

s = socket.socket()
host = "localhost"
port = 8089
s.bind((host, port))
s.listen(1)

file = s.makefile('wb', buffering=None, encoding="utf-8")
sys.stdout = file

while True:
    conn, addr = s.accept()

    #print('New connection from %s:%d' % (addr[0], addr[1]))

    while True:
        try:
            data = conn.recv(1024)
            data = data.decode("utf-8").strip()
            if not data or data == "":
                continue
            elif data == 'exit':
                conn.close()
                break
            else:
                InputArgs = Console.ParseArgs(data)
                Command = InputArgs[0]
                InputArgs = InputArgs[1:]

                method_to_call = getattr(Console, Command, None)

                if method_to_call is None:
                    print("Invalid command")
                    # TODO: TypeError: a bytes-like object is required, not 'str'
                else:
                    method_to_call(InputArgs)

        except socket.timeout:
            # print('server timeout!!' + '\n')
            continue
