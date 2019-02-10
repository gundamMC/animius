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

Update a model config with the provided values.

```
editModelConfig -n 'model config name' [-c '{"some_key": "some_value"}'] [-h '{}'] [-s '{}']
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model config to edit

* *-c, --config* (`dict`) -- Dictionary containing the updated config values (Optional)

* *-h, --hyperparameters* (`dict`) -- Dictionary containing the updated hyperparameters values (Optional)

* *-s, --model_structure* (`dict`) -- Dictionary containing the updated model structure values (Optional)

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

Load a model config from disk.

```
loadModelConfig -n 'model config name'
```

Keyword Argument:

* *-n, --name* (`str`) -- Name of model config to load