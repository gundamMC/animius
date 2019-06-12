# Module: am.Utils

## Overview

### Functions

[get_system_info(...)](https://gundammc.github.io/animius/python/am.Utils#amutilsget_system_info)

shuffle(...)

get_mini_batches(...)

get_length(...)

sentence_to_index(...)

set_sequence_length(...)

freeze_graph(...)

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
