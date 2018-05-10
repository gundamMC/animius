# This is a demo of how a socket client should work
# regardless of the language.

import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 8089))

while True:
    UserInput = input("Input: ")

    if UserInput.lower() == "exit":
        break

    client_socket.send(UserInput.encode("utf-8"))
    respond = ""
    while True:
        respond = client_socket.recv(1024).decode("UTF-8")
        if respond == "end":
            break
        print(respond)

client_socket.send("exit".encode("utf-8"))
client_socket.close()
