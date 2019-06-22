# Module: am.SpeakerVerification

## Overview

### Classes

class MFCC

[class SpeakerVerificationModel](https://gundammc.github.io/animius/python/am.SpeakerVerification/#amspeakerverificationmodel)

## am.SpeakerVerification.SpeakerVerificationModel

Defined in [animius/SpeakerVerification/SpeakerVerificationModel.py](https://github.com/gundamMC/animius/blob/master/animius/SpeakerVerification/SpeakerVerificationModel.py)

### \_\_init\_\_

Args: None

### Methods

#### DEFAULT_HYPERPARAMETERS (Static Method)

```am.SpeakerVerification.SpeakerVerificationModel.DEFAULT_HYPERPARAMETERS()```


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

#### train

```train(self, epochs=800, CancellationToken=None)```

#### load

```load(directory, name='model', data=None)```

#### predict

```predict(input_data, save_path=None, raw=False)```
