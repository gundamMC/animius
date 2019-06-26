# Module: am.Utils

Defined in [animius/Utils.py](https://github.com/gundamMC/animius/blob/master/animius/Utils.py).

## Overview

### Functions

[get_system_info(...)](https://gundammc.github.io/animius/python/am.Utils#amutilsget_system_info)

[shuffle(...)](https://gundammc.github.io/animius/python/am.Utils#amutilsshuffle)

[get_mini_batches(...)](https://gundammc.github.io/animius/python/am.Utils#amutilsgetminibatches)

[get_length(...)](https://gundammc.github.io/animius/python/am.Utils#amutilsgetlength)

sentence_to_index(...)

set_sequence_length(...)

[freeze_graph(...)](https://gundammc.github.io/animius/python/am.Utils#amutilsfreezegraph)

optimize(...)

## am.Utils.get_system_info

```am.Utils.get_system_info()```

Defined in [animius/Utils.py](https://github.com/gundamMC/animius/blob/master/animius/Utils.py).

Returns basic hardware and runtime information.

This function use Python libraries ```psutil``` and ```pynvml``` to obtain system information.

For example:

```
{'cpu_percent': 25.5, 'cpu_count': 6, 'mem_total': 16307, 'mem_available': 11110, 'mem_percent': 31.9,
'disk_total': 339338, 'disk_used': 237581, 'disk_percent': 70.0, 'boot_time': 1556635120.0,'gpu_driver_version': '430.39', 'gpu_device_list': [
{'gpu_name': 'GeForce GTX 1060 6GB', 'gpu_mem_total': 6144, 'gpu_mem_used': 449, 'gpu_mem_percent': 0}]}
```

GPU information won't be shown if your system does not contain any GPU.

Args: None 

Returns:

A dict contains basic hardware and system information.

## am.Utils.shuffle

```am.Utils.shuffle(data_lists)```

Defined in [animius/Utils.py](https://github.com/gundamMC/animius/blob/master/animius/Utils.py).

Randomly shuffle a numpy array.

Args:

* *data_lists* (`list`) -- a list of data.

Returns:

Shuffled list object.

## am.Utils.get_mini_batches

```am.Utils.get_mini_batches(data_lists, mini_batch_size)```

Defined in [animius/Utils.py](https://github.com/gundamMC/animius/blob/master/animius/Utils.py).

Get a batch of data.

Args:

* *data_lists* (`list`) -- a list of data.

* *mini_batch_size* (`int`) -- size of mini batch.

Returns:

A batch of data which represents as a list object.

## am.Utils.get_length

```am.Utils.get_length(sequence)```

Defined in [animius/Utils.py](https://github.com/gundamMC/animius/blob/master/animius/Utils.py).

Get length of a tensor.

Args:

* *sequence* (`tf.Tensor`) -- a tensor of data.

Returns:

Length of the input tensor.

## am.Utils.freeze_graph

```am.Utils.freeze_graph(model, output_node_names, model_dir=None, model_name=None)```

Defined in [animius/Utils.py](https://github.com/gundamMC/animius/blob/master/animius/Utils.py).

Freeze graph of specific model and save it to local files.

Pass model_dir and model_name if model is not loaded.

Args:

* *model* (`str`) -- name of model.

* *output_node_names* (`str`) -- name of output nodes.

* *model_dir* (`str`) -- path to save model. (Optional)

* *model_name* (`str`) -- name of model to save. (Optional)

Returns: 

Path to the output graph.
