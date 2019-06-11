# Functions

## start_server

``` am.start_server(console, port, local=True, password='', max_clients=10) ```

Defined in [animius/SocketServer.py](https://github.com/gundamMC/animius/blob/master/animius/SocketServer.py).

Args:

* *console* (`am.Console`) -- reference to an am.Console object.


* *port* (`int`) -- specific port which the socket server listening on.

* *local* (`boolean`) -- whether or not the server runs on local address. (as known as '127.0.0.1' or 'localhost')

* *password* (`str`) -- password which requires when clients creating a connection with the socket server. (optional)

* *max_clients* (`int`) -- specific number of clients the server can communicate with. (optional)

Returns:

The reference to a thread object which socket server is running on.