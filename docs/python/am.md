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

[class SubtitleParser](https://gundammc.github.io/animius/python/am#amsubtitleparser)

[class Waifu](https://gundammc.github.io/animius/python/am#amwaifu)

[class WordEmbedding](https://gundammc.github.io/animius/python/am#amwordembedding)

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

#### init_hyperdash

```init_hyperdash(name)```

Initialize Hyperdash.

Args:

* *name* (```str```) -- name of hyperdash.

Returns: None

#### init_embedding

```init_embedding(word_embedding_placeholder)```

Initialize embedding.

Args:

* *word_embedding_placeholder* (```str```) -- name of hyperdash.

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

## am.SubtitleParser

Defined in [animius/ParseSubtitle.py](https://github.com/gundamMC/animius/blob/master/animius/ParseSubtitle.py).

### \_\_init\_\_

Args: None

### Methods

#### load

```load(subtitle_path)```

Load subtitle (SSA file) from local file.

Args:

* *subtitle_path** (```str```) -- path to subtitle file.

Returns: None

#### parse_audio_sentences

```parse_audio_sentences()```

Add audio sentences from subtitle file.

Args: None

Returns: None

#### slice_audio

```slice_audio(audio_path, save_path)```

Slice audio and save results into specific path.

Args:

* *audio_path** (```str```) -- path to audio file.

* *save_path** (```str```) -- path to save results.

#### detect_conversation

```detect_conversation(speaking, time_gap=5000)```

Args:

Returns:


## am.Waifu

Defined in [animius/Waifu.py](https://github.com/gundamMC/animius/blob/master/animius/Waifu.py).

### \_\_init\_\_

```am.Waifu(name, models=None, description='', image='')```

Args:

* *name** (```str```) -- name of waifu.

* *models** (```dict```) -- dict of models. (Optional)

* *description** (```str```) -- description of waifu. (Optional)

* *image** (```str```) -- image of waifu. (Optional)

### Methods

#### add_combined_prediction_model

```add_combined_prediction_model(directory, name)```

Add combined prediction model to specific waifu.

Args:

* *directory** (```str```) -- path to the combined prediction model.

* *name** (```str```) -- name of combined prediction model.

Returns: None

#### load_combined_prediction_model

```load_combined_prediction_model()```

Load combined prediction model to specific waifu.

Args: None

Returns: None

#### build_input

#### add_regex

```add_regex(regex_rule, isIntentNER, result)```

Add regex rule.

For example: 

```testWaifu.add_regex('how's the weather in (.+)', True, 'getWeather')```

```testWaifu.add_regex('good morning', False, 'Good morning!')```

Args:

* *regex_rule** (```str```) -- regex rule.

* *isIntentNER** (```boolean```) -- whether or not the regex rule will return Intent and NER.

* *result** (```str```) -- string which the regex rule will return.

Returns: None

#### predict

```predict(sentence)```

Predict results using regex rules and model.

Args:

* *sentence** (```str```) -- sentence which will be predicted by waifu.

Returns: 

For general messages, the format of returned value will be: ```{'message': string}```.

For commands, the format of returned value will be: ``` {'intent': intent, 'ner': [ner, ner_sentence]}```.

#### save

```save(directory, name='waifu')```

Save waifu to local files.

Args: 

* *directory** (```str```) -- path to save files.

* *name** (```str```) -- name of waifu to save.

Returns: Directory to saved file.

#### load (Class method)

```am.Waifu.load(directory, name='waifu')```

Load Waifu from local files.

Args:

* *directory** (```str```) -- path to save files.

* *name** (```str```) -- name of waifu to load.

Returns: a reference to am.Waifu object.

## am.WordEmbedding

Defined in [animius/WordEmbedding.py](https://github.com/gundamMC/animius/blob/master/animius/WordEmbedding.py).

### \_\_init\_\_

Args: None

### Methods

#### create_embedding

```create_embedding(glove_path, vocab_size=100000)```

Create word embeddings from pre-trained word embeddings file.

Since Animius natively does not support word embedding training, you have to download pre-trained word embeddings such as [GloVe](https://nlp.stanford.edu/projects/glove/).

Args:

* *glove_path** (```str```) -- path to pre-trained word embeddings.

* *vocab_size** (```int```) -- amount of vocabularies which will be contained in the word embedding. (Optional)

Returns: None

#### save

```save(directory, name='embedding')```

Save word embeddings to local file.

Args:

* *directory* (```str```) -- directory where you want to save word embeddings file.

* *name** (```str```) -- name of word embeddings file. (Optional)

Returns: directory where word embedding file saves.

#### load (Class method)

```am.WordEmbedding.load(directory, name='embedding')```

Load embedding from local file.

Args:

* *directory* (```str```) -- directory where you want to save word embeddings file.

* *name** (```str```) -- name of word embeddings file. (Optional)

Returns: The reference to the am.WordEmbeddings object.
