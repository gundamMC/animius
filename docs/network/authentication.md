# Network Authentication

Since the server and client are able to communicate across different networks, authentication is required to prevent abusing hardware or stealing model information. 

## Password

When starting a server, the argument `-p` or `--password` is avaliable to set up passwords. Connections without the correct password will be closed.

The password is not saved by the console, so it is expected for the user to pass in the argument every time. This also allows the password to be changed between sessions. If no argument is passed, the server will accept all connections.

## Client

If the server has a password, the client must send the password string encrypted with AES ([encryption](encryption)) upon connection. No JSON format is required. If the password is incorrect, the connection will be closed. Otherwise, the connection will remain open and a 'success' response will be sent.

If the server does not have a password, then the client is not expected to send anything.