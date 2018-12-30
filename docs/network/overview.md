# Network Overview

In addition to using the python console, Animius can also be accessed with a network TCP socket.
This allows for user-friendly or extensive GUIs and clients written in other languages.

## Server

The server has the same requirements as using the python console, with addition of networking. (still writing the script)

## Client

The client has no requirements other than networking. Python is not needed. To communicate with the server, simply use TCP messages.
Although it is recommended for security, the client and the server are not required to be on the same network.

See [(gundamMC/Waifu-GUI)](https://github.com/gundamMC/waifu-gui) for a C# WPF example.

## Commands

The server takes in JSON messages with the following format:

``` JSON
{
  "command": "foo",
  "id": "01:01",
  "arguments": {
    "boo": 2,
    "bar": "MAX"
  }
}
```

`command` takes in a string that specifies a function defined by the server while the dictionary `arguments` define the keyword arguments.
For instance, the above code represents `foo --boo=2 --bar='MAX'`. `id` is simply a string identifier for the client.

## Responses

The server responds in the following format:

``` JSON
{
  "id": "01:01",
  "status": 0,
  "message": "success",
  "data": {
    "foo": 2,
    "boo": "bar"
  }
}
```

`id` is the identifier that the client sends. The server simply returns the same id. `status` is a code that represents the following values

- 0 -> success
- 1 -> failure
- 2 -> argument error
- 3- > error

(An argument error occurs when an argument is missing or has the wrong type, in contrary to an error)

`message` is simply a message that provides additional information on the status. In failure or an error, it would provide the cause of such failure or error. `data` is a dictionary that contains the return values of the command. If the user queries for information on a model, `data` would include the information of the model. Note that `data` is subject to change for each command, while some commands may not even return anything in `data` (an empty dictionary will be used in that case to ensure that all responses contain a `data`).