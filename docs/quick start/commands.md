# Quick Start

This guide will go over the console module of Animius, `am.Console`, and how the conosle provides a set a commands to simplify the interaction process while automating a lot of the house-keeping mess.
*This guide assumes that you have already read the [Quick Start Overview](overview.md) and is familiar with Animius's models*
 
## Starting console

To start the console, simply type in `animius` in your command line. If you installed animius with docker, the console should start automatically when you run an image.

At first, the console will ask you to provide a directory to save data in.

If your input is empty, Console will choose the default path. We highly recommend changing this path.

```
Please enter the data directory to save data in:
Default (\...\animius\resources\)
```

Now, the console will wait for your input - a command.


## Set up a model config

In order to create a model and ultimately a waifu, we will have to create a model config first. In this example, we will be creating an intent-NER model (You can read more about Intent NER here). To begin, let's create an Intent NER model config called `myModelConfig`.

```
createModelConfig --name 'myModelConfig' --type 'IntentNER'
```

With `getModelConfigs`, which reveals a list of model configs, you can verify that `myModelConfig` has been created and loaded.
If you like, you can also get more insight into the model config values with `getModelConfigDetails`. (Note that *--name* and *-n* can be used interchangably.)

```
getModelConfigs

getModelConfigDetails -n 'myModelConfig'
```

We will be coming back to the model config after creating the data.

## Prepare the data

Data is essential when training models. For Intent NER, which takes in English sentences as input, the data object requires a word embedding to both parse data and to create a model. So, let us begin by creating a data named `myData`.

```
createData --name 'myData' --model_config 'myModelConfig'
```

The data equivalent of `getModelConfigs` and `getModelConfigDetails` are `getData` and `getDataDetails`. 

Next, download the Intent NER Data from our [resources page](https://animius.org/). Extract the zip file and place the folder somewhere safe. We can import the data by using:

```
intentNERDataAddParseDatafolder --name 'myData' --path 'some/path/to/data/folder'
```

Now, the data will be parsed and stored in `myData`. You can have a closer look with `getModelConfigDetails`.