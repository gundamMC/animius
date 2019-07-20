# Python Library Quick Start

Animius makes it easy for developers to learn and use our python API to create their own virtual assistants.

This is a step-by-step guide which assumes that you have already read the [Quick Start Overview](https://gundammc.github.io/animius/quick%20start/overview/) and is familiar with Animius's models. 

However, if you are interested in other features and functions, you may refer to the whole [Python API documentation](https://gundammc.github.io/animius/python/am/).

## Install and get started

Animius can be installed as a Pypi package.

```pip install animius```

See detail [install tutorial](https://www.animius.org/install/) on our official website.

After installing all packages and dependencies, import animius module to begin.

```import animius as am```

## Set up a model config

In order to create a model and ultimately a waifu, we will have to create a model config first. 

In this example, we will be creating an Intent NER model. (You can read more about Intent NER here)

To begin, let's create an Intent NER model config called ```myModelConfig``` which will be initialized with default configs, hyperparameters, and model structures of Intent NER model.

```
myIntentModelConfig = am.ModelConfig(cls='IntentNER')

myIntentModelConfig.apply_defaults()
```

Moreover, it's really convenient to retrieve or change this model config.

```
print(myModelIntentConfig.model_structure)

myIntentModelConfig.hyperparameters['learning_rate'] = 0.01
```

We will be coming back to the model config after creating the data.

## Setting up the word embedding

Download a word embedding (we recommend glove) and the Intent NER Data from our [dataset](https://www.animius.org/datasets/) page. 
Extract zip files and place the folder in a convenient place.

To enable the parsing of English text, we will have to use a word embedding called ```myGloveEmbedding```.

```
myGloveEmbedding = am.WordEmbedding()

myGloveEmbedding.create_embedding('\\resources\\glove.twitter.27B.50d.txt', vocab_size=40000)
```

The vocab size parameter is optional but recommended to prevent loading enormous embeddings that take up too much resource.

## Prepare the data and import word embedding

Data is essential when training models. 
For Intent NER, which takes in English sentences as input, the data object requires a word embedding to both parse data and to create a model. 
So, let's create an Intent NER data object named ```myData```, import our glove embedding which created in the previous section, and then set intent folder.

```
myIntentData = am.IntentNERData()

myIntentData.add_embedding_class(myGloveEmbedding)

myIntentData.set_intent_folder('resources\\IntentNER Data')
```

## Setup IntentNER Model

After creating the model config and data, we can create Intent NER model called ```myModel```.

```
myIntentModel = am.IntentNER.IntentNERModel()
```

## Build graph and initialize TensorFlow

TensorFlow uses a dataflow graph to represent your computation in terms of the dependencies between individual operations. 
In order to train our model, let's build its graph and initialize TensorFlow sessions and datasets.

```
myIntentModel.build_graph(myModelConfig, myData)

myIntentModel.init_tensorflow()
```

### Train model

Now we need to train our model, which means making the model learn from the data we prepared. 
Let's test it out by training 10 epochs. 
An epoch is just a cycle during which the model trains over the entire training set.

```
myIntentModel.build_graph(intentNER_mc, intentNER_data)
myIntentModel.init_tensorflow()

myIntentModel.train(epochs=10)
```

### Predict sentence

As we mentioned before, Intent NER will analyze English sentence and extract its Intents and NERs.
After training our model, let's generate our input sentence and test its performance. 
Rather than creating a data, we may simply use string, list, or numpy array as argument.

```
myIntentModel.add_embedding_class(myGloveEmbedding)

myIntentModel.predict(input_data)
```

### Save model

Congrats! Our model performs as well as we imagined before. 
Now, it's time to save it into your disk so that you can retrieve it whenever you want.

```
myIntentModel.save(directory='Animius', name='myIntentNER')
```

## Setup SpeakerVerification Model

Now, as a crucial step to build our own virtual assistant, let's create a SpeakerVerification Model which could verify speakers of audio clips.

### Parse Subtitle

First, let's use ```am.ParseSubtitle``` to split an audio file into small pieces of audio clips according to its corresponding subtitle.

```
parser = am.ParseSubtitle.Parser()

parser.load('Animius Violet Evergarden\\01 jp.ass')

parser.slice_audio('Animius Violet Evergarden\\01.mp3', 'Animius Violet Evergarden\\01_slices')
```

### Create model

Simply following the steps of creating IntentNER model as we mentioned above.

```
mySpeakerVerificationModelConfig = am.ModelConfig(cls='SpeakerVerification')

mySpeakerVerificationData = am.SpeakerVerificationData()

mySpeakerVerificationData.add_text_file('Animius Violet Evergarden\\01 true.txt', is_speaker=True)

mySpeakerVerificationData.add_text_file('Animius Violet Evergarden\\01 false.txt', is_speaker=False)

mySpeakerVerificationModel = am.SpeakerVerification.SpeakerVerificationModel()
```

### Train and save model

```
mySpeakerVerificationModel.build_graph(mySpeakerVerificationModelConfig, mySpeakerVerificationData)

mySpeakerVerificationModel.init_tensorflow()

mySpeakerVerificationModel.train()

mySpeakerVerificationModel.save(r'SpeakerVerification')
```

## Setup Chatbot Model

Now, as an another crucial step to build our own virtual assistant, let's create a Chatbot Model.

### Create Data and add Twitter

```
myChatbotModelConfig = am.ModelConfig(cls='Chatbot')

myChatbotData = am.ChatData()


myGloveEmbedding = am.WordEmbedding()

myGloveEmbedding.create_embedding('Animius Violet Evergarden\\glove.twitter.27B.50d.txt', vocab_size=40000)

myChatbotData.add_embedding_class(myGloveEmbedding)

myChatbotData.add_twitter(r'Animius Violet Evergarden\chat.txt')
```

### Create model

Simply following the steps of creating IntentNER model as we mentioned above.

```
myChatbotModel = am.Chatbot.CombinedChatbotModel()
```

### Train and save model

```
myChatbotModel.build_graph(myChatbotModelConfig, myChatbotModelData)

myChatbotModel.init_tensorflow()

myChatbotModel.train(10)

myChatbotModel.save(directory='Chatbot')
```

## Create Combined Chatbot Model

Now, as an another crucial step to build our own virtual assistant, let's create a Combined Chatbot Model.

### Create Model and Data

```
myCombinedChatbotModelConfig = am.ModelConfig(cls='Chatbot')

myGloveEmbedding = am.WordEmbedding()

myGloveEmbedding.create_embedding('Animius Violet Evergarden\\glove.twitter.27B.50d.txt', vocab_size=40000)

myCombinedChatbotModelData = am.ChatData()

myCombinedChatbotModelData.add_embedding_class(myGloveEmbedding)
```

### Create model

Simply following the steps of creating IntentNER model as we mentioned above.

```
myCombinedChatbotModel = am.Chatbot.CombinedChatbotModel()
```

### Initialize and save model

```
myCombinedChatbotModel.build_graph(mc, data, intent_ner=('Animius', 'myIntentNER'))

myCombinedChatbotModel.init_tensorflow()

myCombinedChatbotModel.save(directory='myCombinedChatbotModel')
```

## Creating your Waifu

We will assume that we have already created a CombinedChatbot model called 'myCombinedChatbot' and a word embedding named 'myEmbedding'.

```
myWaifu = am.Waifu('myWaifu', models=None, description='', image='')

myWaifu.add_combined_prediction_model('resource\\models', 'myCombinedChatbot')

myWaifu.build_input(myEmbedding)
```

### Prediction

To make a prediction (also referred to as inference) using our waifu, simply use the following code.

```
myWaifu.predict('Hey bro. What's up.')
```

## Other methods and classes

We've covered the basics of using Animius Python API to create our own virtual assistants. 
However, if you are interested in other features and awesome functions, you may refer to the whole [Python API guide](https://gundammc.github.io/animius/python/am/).