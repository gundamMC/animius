from .SocketServerModel import client,Request,Response,AEScipher

clients = {}


def new_client(c):
    try:
        #print("%s(%s) 尝试连接" % (c.addr, c.port))
        initAes=c.initRandomAEScipher()
        c.sendAes(0,200,"InitAes",key=initAes["key"],iv=initAes["iv"])
        while True:
            req = c.recv()
        #do something with console
        print(req)
    except socket.errno as e:
        print("Socket error: %s" % str(e))
    except Exception as e:
        print("Other exception: %s" % str(e))
    finally:
        #print("%s(%s) 断开连接" % (c.addr, c.port))
        c.close()


def start_server(port, local=True, cnum=10):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if local:
        host = "127.0.0.1"
    else:
        host = socket.gethostname()
    server.bind((host, port))

    # 侦听客户端
    server.listen(cnum)
    print("服务器已开启")

    while True:
        # 接受客户端连接
        conn, addr = server.accept()
        c = client(conn, addr)
        t = threading.Thread(target=new_client, args=(c,))
        t.start()

if __name__ == "__main__":
    start_server(12345, local=False)
    #print("服务器已关闭")