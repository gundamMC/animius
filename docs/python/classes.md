# Classes 

## am.Commands

Defined in [animius/Console.py](https://github.com/gundamMC/animius/blob/master/animius/Console.py).

Class am.Commands contains the whole details of console commands, such as their arguments or their descriptions.

### __init__

Args:

* *console* (am.Console) -- reference to an am.Console object.

### Properties

None

### Methods

#### __iter__

```__iter__()```
Returns an iterator for the command dict.

Args: None

Returns: A reference to the iterator for the command dict.

#### __getitem__

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
