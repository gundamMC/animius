import socket
import ProjectWaifu.Console as Console
import ProjectWaifu.Utils as Utils

s = socket.socket()
host = "localhost"
port = 8089
s.bind((host, port))
s.listen(1)

while True:
    conn, addr = s.accept()

    print('New connection from %s:%d' % (addr[0], addr[1]))

    Utils.setSocket(conn)

    while True:
        try:
            data = conn.recv(1024)
            data = data.decode("utf-8").strip()
            if not data or data == "":
                break
            elif data == 'exit':
                conn.close()
                break
            else:
                InputArgs = Console.ParseArgs(data)
                Command = InputArgs[0]
                InputArgs = InputArgs[1:]

                method_to_call = getattr(Console, Command, None)

                if method_to_call is None:
                    Utils.printMessage("Invalid command")
                else:
                    method_to_call(InputArgs)

                Utils.printMessage("end")

        except socket.timeout:
            print('Connection timed out')
            continue
