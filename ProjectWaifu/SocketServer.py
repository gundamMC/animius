import socketserver
import threading


# Used to interact with other clients based on other languages
# E.g. WaifuGUI (gundamMC/WaifuGUI)
class MyTCPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        while True:
            data = self.request.recv(1024)
            if not data:
                break
            data = data.decode("UTF-8").strip()
            if data == "exit":
                print("Socket exited")
                break
            print(data)
            # self.request.send("Test")


server = socketserver.TCPServer(("localhost", 8089), MyTCPHandler)
server_thread = threading.Thread(target=server.serve_forever)
server_thread.start()
# server.shutdown()
