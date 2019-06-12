# Module: am

Defined in [animius/\_\_init\_\_.py](https://github.com/gundamMC/animius/blob/master/animius/__init__.py).

## Overview

### Modules

[Chatbot](https://gundammc.github.io/animius/python/am.Chatbot)

[IntentNER](https://gundammc.github.io/animius/python/am.IntentNER)

[SpeakerVerification](https://gundammc.github.io/animius/python/am.SpeakerVerification)

[Utils](https://gundammc.github.io/animius/python/am.Utils)

### Classes

class ChatbotData

class CombinedPredictionData

[class Commands](https://gundammc.github.io/animius/python/am/#amcommands)

[class Console](https://gundammc.github.io/animius/python/am#amconsole)

class IntentNERData

class Model

class ModelConfig

class SpeakerVerificationData

class SubtitleParser

class Waifu

## am.Commands

Defined in [animius/Commands.py](https://github.com/gundamMC/animius/blob/master/animius/Commands.py).

Class am.Commands contains the whole details of console commands, such as their arguments or their descriptions.

### \_\_init\_\_

Args:

* *console* (am.Console) -- reference to an am.Console object.

### Properties

None

### Methods

#### \_\_iter\_\_

```__iter__()```

Returns an iterator for the command dict.

Args: None

Returns: A reference to the iterator for the command dict.

#### \_\_getitem\_\_

```__getitem__(item)```

Returns detail information of a command.

For example, ```__getitem__('exportWaifu')``` will return:

```
[
    console.export_waifu,
    {
        '-n': ['name', 'str', 'Name of waifu'],
        '-p': ['path', 'str', 'Path to export file']
    },
    'Export a waifu',
    "exportWaifu -n 'waifu name' -p 'path_name'"
]
```

Args: 

* *item* (`str`) -- command name.

Returns: A dict similar to the example above.


## am.Console

Defined in [animius/Console.py](https://github.com/gundamMC/animius/blob/master/animius/Console.py).

Class am.Console contains all of the console commands.

### \_\_init\_\_

### Properties

### Methods

## am.Console.server

``` am.Console.server(console, port, local=True, password='', max_clients=10) ```

Defined in [animius/Console.py](https://github.com/gundamMC/animius/blob/master/animius/Console.py).

Start socket server on specific port.

Args:

* *console* (`am.Console`) -- reference to an am.Console object.

* *port* (`int`) -- specific port which the socket server listening on.

* *local* (`boolean`) -- whether or not the server runs on local address. (as known as '127.0.0.1' or 'localhost')

* *password* (`str`) -- password which requires when clients creating a connection with the socket server. (optional)

* *max_clients* (`int`) -- specific number of clients the server can communicate with. (optional)

Returns:

The reference to a thread object which socket server is running on.