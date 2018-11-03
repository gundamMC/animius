# Commands Overview

Commands are used to interact with the console as well as through the network socket. A command has two parts: the command and the arguments. 
In the console, a command could look something like this:
```
createModel --name='myModel' --model='SpeakerVerification'
```
The command must start with the command, indi

The equivalent of this in the network socket would be:
``` JSON
{
  "command": "createModel",
  "arguments": {
    "name": "myModel",
    "model": "SpeakerVerification"
  }
}
```

The various commands and their arguments can be found in this section. Network-socket-related commands can be found under the [Network section](https://gundammc.github.io/Project-Waifu/network/overview/).