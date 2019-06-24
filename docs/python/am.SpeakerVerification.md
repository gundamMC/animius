# Module: am.SpeakerVerification

## Overview

### Classes

[class MFCC](https://gundammc.github.io/animius/python/am.SpeakerVerification/#amspeakerverificationmodelmfcc)

[class SpeakerVerificationModel](https://gundammc.github.io/animius/python/am.SpeakerVerification/#amspeakerVerificationspeakerVerificationModel)

## am.SpeakerVerification.MFCC

Defined in [animius/SpeakerVerification/MFCC.py](https://github.com/gundamMC/animius/blob/master/animius/SpeakerVerification/MFCC.py)

### Methods

#### get_MFCC

```get_MFCC(path, window=10, step=3, num_cepstral=39, flatten=False)```

Get MFCC data.

Args:

* *path* (`str`) -- path to wav files.

* *window* (`int`) -- amount of window. (Optional)

* *step* (`int`) -- amount of step. (Optional)

* *num_cepstral* (`int`) -- amount of cepstral. (Optional)

* *flatten* (`boolean`) -- whether or not to output flatten data. (Optional)

Returns:

A numpy array of MFCC data.

## am.SpeakerVerification.SpeakerVerificationModel

Defined in [animius/SpeakerVerification/SpeakerVerificationModel.py](https://github.com/gundamMC/animius/blob/master/animius/SpeakerVerification/SpeakerVerificationModel.py)

### \_\_init\_\_

Args: None

### Methods

#### DEFAULT_HYPERPARAMETERS (Static Method)

```am.SpeakerVerification.SpeakerVerificationModel.DEFAULT_HYPERPARAMETERS()```

Get default hyperparameters of SpeakerVerification model.

Args: None

Returns: 

```

{
    'learning_rate': 0.005,
    'batch_size': 2048,
    'optimizer': 'adam'
}
        
```

#### DEFAULT_MODEL_STRUCTURE (Static Method)

```am.SpeakerVerification.SpeakerVerificationModel.DEFAULT_MODEL_STRUCTURE()```

Get default model structure of SpeakerVerification model.

Args: None

Returns: 

```
{
    'filter_size_1': 3,
    'num_filter_1': 10,
    'pool_size_1': 2,
    'pool_type': 'max',
    'filter_size_2': 5,
    'num_filter_2': 15,
    'fully_connected_1': 128,
    'input_window': 10,
    'input_cepstral': 39
}
```

#### build_graph

```build_graph(model_config, data, graph=None)```

Build the graph for SpeakerVerification model.

Args:

* *model_config* (`am.ModelConfig`) -- reference to an am.ModelConfig object.

* *data* (`am.SpeakerVerificationData`) -- reference to an am.SpeakerVerificationData object.

* *graph* (`tf.Graph`) -- reference to a tf.Graph object.

Returns: None

#### train

```train(epochs=800, CancellationToken=None)```

Train model with specific epochs.

Args:

* *epochs* (`int`) -- amount of epochs which the model will train with. (Optional)

* *data* (`am.SpeakerVerificationData`) -- reference to an am.SpeakerVerificationData object.

* *CancellationToken* (`NoneType`) -- CancellationToken. (Optional)

Returns: None

#### load (Class method)

```load(directory, name='model', data=None)```

Load SpeakerVerification model from local files.

Args:

* *directory* (`str`) -- directory of files.

* *name* (`str`) -- name of SpeakerVerification model.

* *data* (`am.SpeakerVerificationData`) -- reference to an am.SpeakerVerificationData object.

Returns: 

The reference to an am.SpeakerVerification.SpeakerVerificationModel object.

#### predict

```predict(input_data, save_path=None, raw=False)```

Predict input data and save predicted values to local files.

Args:

* *input_data* (`str`) -- input data.

* *save_path* (`str`) -- path to save results. (Optional)

* *raw* (`boolean`) -- whether or not to return raw data. (Optional)

Returns:

Predicted data.

#### predict_folder

```predict_folder(input_data, folder_directory, save_path=None, raw=False)```

Read data in the folder and predict results.

Args:

* *input_data* (`str`) -- input data.

* *folder_directory* (`str`) -- path to save results. (Optional)

* *save_path* (`str`) -- path to save results. (Optional)

* *raw* (`boolean`) -- whether or not to return raw data. (Optional)

Returns:

Predicted data.