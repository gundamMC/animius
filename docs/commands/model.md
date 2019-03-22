# Model

The following section describe the commands related to Models.

Defined in animius\Console.py

### createModel

Create a model.

```
createModel -n 'model name' -t 'ModelType' -c 'model_config name' -d 'data name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model

* *-t, --type* (`str`) -- Type of model

* *-c, --model_config* (`str`) -- Name of model config to use

* *-d, --data* (`str`) -- Name of data to use

* *-i, --intent_ner_model** (`str`) -- (Optional) Name of IntentNER Model (Only required for creating CombinedChatbot Model)

Here's a list of model types.

Chatbot: am.Chatbot.ChatbotModel()

CombinedChatbot: animius.Chatbot.CombinedChatbotModel()

IntentNER: animius.IntentNER.IntentNERModel()

SpeakerVerification: animius.SpeakerVerification.SpeakerVerificationModel()

### deleteModel

Delete a model.

```
deleteModel -n 'model name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model to delete

### saveModel

Save a model.

The graph is saved in '\resource\model_name\model_name_graph.pb'

```
saveModel -n 'model name' -g True
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model to save
* *-g, --graph* (``bool``) -- Whether to save the graph

### loadModel

Load a model.

```
loadModel -n 'model name' -d 'data name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model to load

* *-d, --data* (`str`) -- Name of data to set to model

### exportModel

Export a model to zip file.

```
exportModel -n 'model name' -p 'some\path\to\export\'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model to export

* *-p, --path* (`str`) -- Path to export file

### importModel

Import a model from zip file.

```
importModel -n 'model name' -p 'some\path\to\export\model_name.zip'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model to export

* *-p, --path* (`str`) -- Path to import file


###getModels

Get a list of existing models.

```
getModels
```

No argument required.

This command returns a dictionary of which the keys are the name of models and the values are the details.

The details will be empty if the model is not loaded.

```
{
	"model_name": {
		"name": "model_name",
		"type": "<class 'model_class'>"
	}
}
```


### getModelDetails

Get the details of a model.

```
getModelDetails -n 'model name'
```

Keyword Arguments:

* *-n, --name* (``str``) -- Name of model

This command returns a dictionary of details of a model, which contains configs, hyperparameters, structures, saved name and saved directory of the model.

```
{
	'config': {
		'device': '/gpu:0',
		'class': 'IntentNER',
		'epoch': 0,
		'cost': None,
		'display_step': 1,
		'tensorboard': None,
		'hyperdash': None,
		'graph': 'resources\\models\\model_name\\model_name_graph.pb',
		'frozen_graph': 'resources\\models\\model_name\\frozen_model.pb'
	},
	'model_structure': {
		'max_sequence': 20,
		'n_hidden': 128,
		'gradient_clip': 5.0,
		'node': 'gru',
		'n_intent_output': 15,
		'n_ner_output': 8,
		'n_vector': 303,
		'word_count': 100000
	},
	'hyperparamter': {
		'learning_rate': 0.003,
		'batch_size': 1024,
		'optimizer': 'adam'
	},
	'saved_directory': 'resources\\models\\model_name',
	'saved_name': 'model_name'
}
```


### setData

Set model data.

```
setData -n 'model name' -d 'data name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model

* *-d, --data* (`str`) -- Name of data

### train

Train a model.

The training process will be held in another thread.

The training device is defined in the model config.

```
train -n 'model name' -e 10
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model to train

* *-e, --epoch* (`int`) -- Number of epochs to train for

### stopTraining

Cancel training a model. 

The model will stop once it finishes the current epoch.

```
stopTraining -n 'model name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model to stop


### predict

Make predictions with a model.

```
predict -n 'model name' -i 'name of input data' -s '\some\path.txt'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model

* *-i, --input_data* (`str`) -- Name of input data

* *-s, --save_path* (`str`) -- Path to save result (Optional)

### freezeGraph

Freeze Tensorflow graph and latest checkpoint to 'resource\model_name\frozen_model.pb'.

```
freezeGraph -n 'model name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model

### optimize

Optimize a frozen model (see FreezeGraph) for inference.

```
optimize -n 'model name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model