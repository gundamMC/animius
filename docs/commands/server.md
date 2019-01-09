# Server

See [Network](../../network/overview) for details.

### startServer

Start a socket server and listen for clients. The server runs on a separate thread so the console will still function.

```
startServer -p 23333 [-l True] [-pwd 'p@ssword'] [-c 10]
```

Keyword Arguments:

* *-p, --port* (``int``) -- Port to listen on

* *-l, --local* (``bool``) -- If the server is running locally (server will listen on 127.0.0.1 if this is true or not set) (Optional)

* *-pwd, --password* (``str``) -- Password of server (Optional)

* *-c, --max_clients* (``int``) -- Maximum number of clients (Optional)

### stopServer

Stop current socket server and close all connections.

```
stopServer
```

No argument required.