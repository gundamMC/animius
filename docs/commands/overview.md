# Commands Overview

Commands are used to interact with the console as well as through the network socket. A command has two parts: the command and the arguments. 
In the console, a command could look something like this:

```
createModel --name 'myModel' --type 'SpeakerVerification'
```

The equivalent of this in the network server would be:

``` JSON
{
  "command": "createModel",
  "arguments": {
    "name": "myModel",
    "type": "SpeakerVerification"
  }
}
```

The various commands and their arguments can be found in this section. Network-socket-related commands can be found under the [Network section](https://gundammc.github.io/Animius/network/overview/).
