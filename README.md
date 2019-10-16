<div align="center">
  <img src="https://user-images.githubusercontent.com/10915258/54398575-da5b9b80-4677-11e9-8b12-dfa06e62819d.png"><br><br>
</div>

-----------------

Animius is an open source software library for creating deep-learning-powered virtual assistants.
It provides an intuitive workflow that extracts data from existing media (such as anime
and TV shows) and trains on them to provide a personalized AI. The flexible architecture enables you
to add custom functionality to your virtual assistant.

Animius also ships with a high-level API `animius.Console` that allows users without programming 
experience to use Animius.

## Installation

Install the current release from PyPi:
```
pip install animius
```

Then, install Tensorflow (recommended version 1.14. *Animius does not support TF 2.0*). We recommend using the GPU package (`tensorlfow-gpu`)
if you are going to train your own virtual assistant. Read more on Tensorflow installation [here](https://www.tensorflow.org/install).

*See [Installing Animius](https://www.animius.org/install) for detailed instructions and Docker installation guide.*

## Getting Started

For more information, check out our [quick start guides](https://gundammc.github.io/animius/quick%20start/overview/).

Here's a quick overview of how to create an intent-ner model with Animius:

```python
# create a new model config for intent ner and set default values
myIntentModelConfig = am.ModelConfig(cls='IntentNER')
myIntentModelConfig.apply_defaults()

# we can set & view the values of the config before creating the model 
print(myModelIntentConfig.model_structure)
myIntentModelConfig.hyperparameters['learning_rate'] = 0.01

# load a pretrained word embedding from disk
myGloveEmbedding = am.WordEmbedding()
myGloveEmbedding.create_embedding('\\resources\\glove.twitter.27B.50d.txt', vocab_size=40000)

# create a data object for your training set
myIntentData = am.IntentNERData()
myIntentData.add_embedding_class(myGloveEmbedding)
myIntentData.set_intent_folder('resources\\IntentNER Data')

# create a model
myIntentModel = am.IntentNER.IntentNERModel()

# build a TF graph based on the config and data
myIntentModel.build_graph(myModelConfig, myData)
# start tensorflow
myIntentModel.init_tensorflow()

# train!
myIntentModel.train(epochs=10)
```

After training the intent-ner and chatbot models, we can create a waifu - a virtual assistant. The code is a bit long, so I will skip a lot of it. Feel free to check out our demo and our quick start guide for more information.

Anyways, you can simply interact with a working waifu like this:

```python
myWaifu.predict('Hey!')

myWaifu.predict('What is the time in New York right now?')

myWaifu.predict('Play me some music')
```

## For more information
- [Animius Website](https://animius.org/)
- [Animius Documentation](https://gundammc.github.io/animius/)

Also, feel free to check out our other projects surrounding Animius:
- [Animius Link](https://github.com/gundamMC/animius-link) - A middleware for extensive assistance functions and socketio support
- [Animius Anywhere](https://github.com/gundamMC/animius-anywhere) - A Vue Native mobile GUI client for Animius.
- [Animius Live](https://github.com/gundamMC/animius-live) - A WIP Electron desktop client. 

## License
[Apache License 2.0](LICENSE)
