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

[class Model](https://gundammc.github.io/animius/python/am#ammodel)

[class ModelConfig](https://gundammc.github.io/animius/python/am#ammodelconfig)

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

Class am.Console includes the method corresponding to each command, a queue which controls the workflow, and a command handler.

### \_\_init\_\_

Args:

* *init_directory* (```str```) -- the path to the directory in which animius saves resources. (Optional)

### Properties

### Methods

#### am.Console.server

``` am.Console.server(console, port, local=True, password='', max_clients=10) ```

Start socket server on specific port.

Args:

* *console* (`am.Console`) -- reference to an am.Console object.

* *port* (`int`) -- specific port which the socket server listening on.

* *local* (`boolean`) -- whether or not the server runs on local address. (as known as '127.0.0.1' or 'localhost')

* *password* (`str`) -- password which requires when clients creating a connection with the socket server. (optional)

* *max_clients* (`int`) -- specific number of clients the server can communicate with. (optional)

Returns:

The reference to a thread object which socket server is running on.

## am.Model

am.Model is an abstract class which is the template of other models.

Defined in [animius/Model.py](https://github.com/gundamMC/animius/blob/master/animius/Model.py).

### \_\_init\_\_

Args: None

### Properties

### Methods

#### DEFAULT_CONFIG

```am.Model.DEFAULT_CONFIG()```

Get defaul model config.

Args: None

Returns: A dict of model config.

#### DEFAULT_MODEL_STRUCTURE

```am.Model.DEFAULT_MODEL_STRUCTURE()```

Get defaul model structure of specific model.

Args: None

Returns: A dict of model structure.

#### DEFAULT_HYPERPARAMETERS

```am.Model.DEFAULT_HYPERPARAMETERS()```

Get defaul hyperparameters.

Args: None

Returns: A dict of hyperparameters.

#### build_graph

'build_graph' is a abstract method.

#### init_tensorflow

```init_tensorflow(graph=None, init_param=True, init_sess=True)```

Initialize TensorFlow.

Args:

* *graph* (```tf.Graph```) -- reference to a tf.Graph object. (Optional)

* *init_param** (```bool```) --  whether or not to initialize parameters. (Optional)

* *init_sess** (```bool```) --  whether or not to initialize tf.Session. (Optional)

Returns: None.

#### model_config

```model_config()```

Get Model Config of specific model.

Args: None

Returns: The reference to a am.ModelConfig object.

#### save

```save(directory=None, name='model', meta=True, graph=False)```

Save model to local file.

Args:

* *directory* (```str```) -- directory where you want to save file. (Optional)

* *name** (```str```) -- name of model file. (Optional)

* *meta** (```boolean```) --  whether or not to save meta file. (Optional)

* *graph** (```boolean```) --  whether or not to save graph. (Optional)

Returns: directory where model file saves.

#### load (Class method)

```load(cls, directory, name='model',  data=None)```

Load model from local file.

Args:

* *cls* (```str```) -- type of model, must be included in ['SpeakerVerification', 'Chatbot', 'IntentNER', 'CombinedChatbot'].

* *directory* (```str```) -- directory where you want to save model file.

* *name** (```str```) -- name of model file. (Optional)

* *data** (```str```) -- name of data to load. (Optional)

Returns: The reference to the am.Model object.

## am.ModelConfig

Defined in [animius/ModelConfig.py](https://github.com/gundamMC/animius/blob/master/animius/ModelConfig.py).

### \_\_init\_\_

Args:

* *cls* (```str```) -- type of model config, must be included in ['SpeakerVerification', 'Chatbot', 'IntentNER', 'CombinedChatbot'].

* *config* (```dict```) -- dict of config. (Optional)

* *hyperparameters* (```dict```) -- dict of hyperparameters. (Optional)

* *model_structure* (```dict```) -- dict of model structures. (Optional)


### Properties

config

hyperparameters

model_structure

### Methods

#### apply_defaults

```apply_defaults()```

Apply default model config.

For example:

```
config = am.ModelConfig(cls="Chatbot")
config.apply_defaults()
```

Args: None

Returns: None

#### save

```save(directory, name='model_config')```

Save model config to local file.

Args:

* *directory* (```str```) -- directory where you want to save config file.

* *name** (```str```) -- name of model config file. (Optional)

Returns: directory where config file saves.

#### load (Class method)

```load(cls, directory, name='model_config')```

Load model config from local file.

Args:

* *cls* (```str```) -- type of model config, must be included in ['SpeakerVerification', 'Chatbot', 'IntentNER', 'CombinedChatbot'].

* *directory* (```str```) -- directory where you want to save config file.

* *name** (```str```) -- name of model config file. (Optional)

Returns: The reference to the am.ModelConfig object.