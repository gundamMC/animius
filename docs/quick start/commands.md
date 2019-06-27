# Quick Start

This guide will go over the console module of Animius, `am.Console`, and how the console provides a set of commands to simplify the interaction process while automating a lot of the house-keeping mess.
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

In order to create a model and ultimately a waifu, we will have to create a model config first. 
In this example, we will be creating an intent-NER model (You can read more about Intent NER here). 
To begin, let's create an Intent NER model config called `myModelConfig`.

```
createModelConfig --name 'myModelConfig' --type 'IntentNER'
```

With `getModelConfigs`, which reveals a list of model configs, you can verify that `myModelConfig` has been created and loaded.
If you like, you can also get more insight into the model config values with `getModelConfigDetails`. (Note that *--name* and *-n* can be used interchangeably.)

```
getModelConfigs

getModelConfigDetails -n 'myModelConfig'
```

We will be coming back to the model config after creating the data.

## Prepare the data

Data is essential when training models. 
For Intent NER, which takes in English sentences as input, the data object requires a word embedding to both parse data and to create a model. 
So, let us begin by creating a data named `myData`.

```
createData --name 'myData' --model_config 'myModelConfig'
```

The data equivalent of `getModelConfigs` and `getModelConfigDetails` are `getData` and `getDataDetails`. 

### Setting up the word embedding

Next, download a word embedding (we recommend glove) and the Intent NER Data from our [resources page](https://animius.org/). 
Extract the zip file and place the folder somewhere safe. 

To enable the parsing of English text, we will have to use a word embedding. 
We can create an embedding object with `createEmbedding`:

```
createEmbedding --name 'myEmbedding' --path '/some/path/to/embedding.txt' --vocab_size 50000
```

The vocab size parameter is optional but recommended to prevent loading enormous embeddings that take up too much resource.

### Importing data

We can import the data by using:

```
intentNERDataAddParseDatafolder --name 'myData' --path 'some/path/to/data_folder/'
```

Now, the data will be parsed and stored in `myData`. 
You can have a closer look with `getModelConfigDetails`.

## Setup the Model

After creating model config and data, we can create the model now.

```
createModel -n 'myModel' -t 'IntentNER' --model_config 'myModelConfig' --data 'myData'
```

The data equivalent of `getModelConfigs` and `getModelConfigDetails` are `getModels` and `getModelDetails`. 

### Training

Now we need to train our model, which means making the model learn from the data we prepared. 
Let's test it out by training 10 epochs. 
An epoch is just a cycle during which the model trains over the entire training set.

```
train -n 'myModel' -e 10
```

Training will be done in the background by another thread, and you can cancel the training process by using `stopTrain -n 'myModel'`.

## The Console System

### Saving

The console provides an automatic clean saving system. To save any object, simply use the command `save{Type}`. 
For instance, to save a model config, use `saveModelConfig -n 'myModelConfig'`. 
To save data, `saveData`. And, to save a model, `saveModel`.

And, please remember to save the console also, or else your created objects will not be recognized the next time you start animius. 
To save the console, simply use:

```
save
```

### Loading

An item created in the console will be automatically loaded. However, when restarting a console, an item will not be loaded to save performance. Thus, before an object can be used, it must be loaded with the `load{Type}` command. This is similar to the save command. (e.g. `loadModelConfig`, `loadData`)


### Deleting

If you would like to delete an object from the console, simply use `delete{Type}`. 
This will remove the object from console but will not remove the actual file storage. 
That is, any save files will remain. See [file structure](../file_structure/overview.md)

## Creating your Waifu

Now, this tutorial will jump a bit from the IntentNER model to a CombinedChatbot model to give a broader sense of using console. 
We will assume that we have already created a CombinedChatbot model called 'myCombinedChatbot' and a word embedding named 'myEmbedding'.

Now, create your waifu with `createWaifu`.

```
createWaifu -n 'myWaifu' --combined_chatbot_model 'myCombinedChatbot' --embedding 'myEmbedding'
```

We can take a sneak peek with `getWaifuDetail -n 'myWaifu'`.

### Prediction

To make a prediction (also referred to as inference) using our waifu, simply use `waifuPredict`.

```
waifuPredict -n 'myWaifu' --sentence 'Hello world!'
```

## Other commands

We have covered the basics of using commands to interact with the console in this tutorial. 
There are, nevertheless, much more commands that you can use to customize your workflow and your virtual assistant.

To learn more about commands, visit the [commands section](../commands/overview.md).

