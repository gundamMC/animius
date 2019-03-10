# Model Config

###getModelConfigs

Get a list of existing model configs.

```
getModelConfigs
```

No argument required.

### createModelConfig

Create a model config with the provided values.

```
createModelConfig -n 'model config name' -t 'type' [-c '{"some_key": "some_value"}'] [-h '{}'] [-ms '{}']
```

Keyword Arguements:

* *-n, --name* (`str`) -- Name of model config

* *-t, --type* (`str`) -- Name of the model type

* *-c, --config* (`dict`) -- Dictionary of config values (Optional)

* *-h, --hyperparameters* (`dict`) -- Dictionary of hyperparameters values (Optional)

* *-s, --model_structure* (`dict`) -- Dictionary of model_structure values (Optional)

### editModelConfig

Update a model config with provided values.

Either providing full dict of configs or changing specific values is allowed.

```
editModelConfig -n 'model config name' [-c '{"some_key": "some_value"}'] [-h '{}'] [-s '{}']

editModelConfig -n 'model config name' [-bs 1024] [-lr 0.1] [-d '/gpu:0']
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model config to edit

* *-c, --config* (`dict`) -- Dictionary containing the updated config values (Optional)

* *-h, --hyperparameters* (`dict`) -- Dictionary containing the updated hyperparameters values (Optional)

* *-s, --model_structure* (`dict`) -- Dictionary containing the updated model structure values (Optional)

* *-d, --device* (`str`) -- Name of device to use (Optional)

* *-cls, --class* (`str`) -- Model class (Optional)

* *-e, --epoch* (`int`) -- Number of epoches (Optional)

* *-cost, --cost* (``) -- config.cost (Optional)

* *-ds, --display_step* (``) -- config.display_step (Optional)

* *-tb, --tensorboard* (``) -- config.tensorboard (Optional)

* *-lr, --learning_rate* (`float`) -- Learning rate (Optional)

* *-bs, --batch_size* (`int`) -- Batch size (Optional)

* *-op, --optimizer* (`str`) -- Name of optimizer (Optional)

* *-ms, --max_sequence* (`int`) -- model_structure.max_sequence (Optional)

* *-nh, --n_hidden* (`int`) -- model_structure.n_hidden (Optional)

* *-gc, --gradient_clip* (`float`) -- model_structure.gradient_clip (Optional)

* *-no, --node* (`str`) -- model_structure.node (Optional)

* *-nio, --n_intent_output* (`int`) -- model_structure.n_intent_output (Optional)

* *-nno, --n_ner_output* (`int`) -- model_structure.n_ner_output (Optional)

* *-l, --layer* (`int`) -- Number of layers (Optional)

* *-bw, --beam_width* (`int`) -- Beam width (Optional)

* *-fs1, --filter_size_1* (`int`) -- model_structure.filter_size_1 (Optional)

* *-fs2, --filter_size_2* (`int`) -- model_structure.filter_size_2 (Optional)

* *-nf1, --num_filter_1* (`int`) -- model_structure.num_filter_1 (Optional)

* *-nf2, --num_filter_2* (`int`) -- model_structure.num_filter_2 (Optional)

* *-ps1, --pool_size_1* (`int`) -- model_structure.pool_size_1 (Optional)

* *-pt, --pool_type* (`str`) -- model_structure.pool_type (Optional)

* *-fc1, --fully_connected_1* (`int`) -- model_structure.fully_connect_1 (Optional)

* *-iw, --input_window* (`int`) -- model_structure.input_window (Optional)

* *-ic, --input_cepstral* (`int`) -- model_structure.input_cepstral (Optional)

### deleteModelConfig

Delete a model config.

```
deleteModelConfig -n 'model config name'
```

Keyword Argument:

* *-n, --name* (`str`) -- Name of model config to delete

### saveModelConfig

Save a model config.

```
saveModelConfig -n 'model config name'
```

Keyword Argument:

* *-n, --name* (`str`) -- Name of model config to save

### loadModelConfig

Load a model config.

```
loadModelConfig -n 'model config name'
```

Keyword Argument:

* *-n, --name* (`str`) -- Name of model config to load

### exportModelConfig

Export a model config to zip file.

```
exportModelConfig -n 'model config name' -p 'some\path\to\export\'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model config to export

* *-p, --path* (`str`) -- Path to export file

### importModelConfig

Import a model config from zip file.

```
importModel -n 'model name' -p 'some\path\to\export\model_name.zip'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model config to export

* *-p, --path* (`str`) -- Path to import file

