# Model

###getModels

Get a list of existing models.

```
getModels
```

No argument required.

### createModel

Create a model.

```
createModel -n 'model name' -t 'ModelType'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model

* *-t, --type* (`str`) -- Type of model

### deleteModel

Delete a model.

```
deleteModel -n 'model name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model to delete

### saveModel

Save a model.

```
saveModel -n 'model name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model to save

### loadModel

Load a model.

```
loadModel -n 'model name' -d 'data name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model to load

* *-d, --data* (`str`) -- Name of data to set to model

### getModelDetails

Return the details of a model.

```
getModelDetails -n 'model name'
```

Keyword Arguments:

* *-n, --name* (``str``) -- Name of model

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

```
train -n 'model name' -e 10
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of model to train

* *-e, --epoch* (`int`) -- Number of epochs to train for

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

Freeze Tensorflow graph and latest checkpoint to a file.

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