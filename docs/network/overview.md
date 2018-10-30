# Network Overview

In addition to using the python console, Project Waifu can also be accessed with a network TCP socket.
This allows for user-friendly or extensive GUIs and clients written in other languages.

## Server

The server has the same requirements as using the python console, with an addition of networking. (still writing the script)

## Client

The client has no requirements other than networking. Python is not needed. To communicate with the server, simply use TCP messages.
Although it is recommened for security, the client and the server are not required to be on the same network.

See [(gundamMC/Waifu-GUI)](https://github.com/gundamMC/waifu-gui) for a C# WPF example.

## Format

The server takes in JSON messages with the following format:

``` JSON
{
  "command": "foo",
  "parameters": {
    "foo": 0.001,
    "boo": 2,
    "bar": "MAX"
  }
}
```

`command` takes in a string that specifies a function defined by the server while the dictionary `parameters` define the keyword arguments.
For instance, the above code represents `#!python foo(foo=0.001, boo=2, bar='MAX')`