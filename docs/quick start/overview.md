# Get Started with Animius

Animius is an open-source deep-learning library for creating AI virtual assistants. Animius offers both a user-friendly console for begginers and a Python API for developers to add their own functionalities into their AIs. See the sections below to get started.

## Dissecting your Waifu

### Waifu

Each AI is called a "waifu" in Animius. A waifu functions by incorporating various deep learning models together and transforming their predictions. Said DL models are called "models" (*easy?*). Basically, waifus are powered by models.

### Models

There are, of course, a variety of models in Animius, and they are separated by their usage. For instance, a model can be a speaker verification model (since it verifies the speaker). These models, nevertheless, are only the blueprints for you to build upon. You, as the user, provides a model config and data when creating a model.

#### Introduction to Machine Learning

Models are *Machine Learning* models (Deep Learning is just a branch of Machine Learning). Essentially, they are mathematical algorithms that train on existing data and will make predictions based on such data. For instance, you can train a chatbot model on speech data from Jon Snow in Game of Thrones. Then, when you ask the model to make a response to your sentence, let's say "How's it going," the model will respond accordingly. *Winter is coming.*

To learn more about machine learning fundamentals and concepts, consider taking the [Stanford University Machine Learninig MOOC by Andrew Ng](https://www.coursera.org/learn/machine-learning) and his [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning).

### Model config

A model config has three parts: config, model structure, and hyperparameters. The config section includes basic information that applies to all models, such as the device it is running on, the type of model, the epoch (See training), and tensorboard location.
The model structure section, meanwhile, defines the mathematical structure of the deep learning algorithms and thus varies across different types of models. For instance, you can change the number of nodes and layers a model will have, thus effectively increasing or decreasing the performance and resource usage of a model. Nevertheless, we do not recommend changing the default values unless you know what you are doing.
Lastly, the hyperparameters define the training aspect of the model, including values such as learning rate, batch size, and optimizer. Like the model structures, we do not recommend changing the default values.

### Data

A data object is simply a data parser. That is, it reads data stored as files on your drives and stores the values. For some models, a word embedding is also required before parsing files and passing the data into a model.

### Word embedding

A word embedding is exactly what it sounds like. It translates word tokens into numerical vectors.

### Conclusion

Thus, to wrap up, the basic structure of your AI would be: *a waifu including various models. Each model contains a model config and a data object For some models, the data object will contain a word embedding.*

## Creating your Waifu

To create your own waifu, it is simply. Recall the structure detailed above. All you have to do is start from the bottom and move your way up. In other words, *gather data, prepare model configs, create models using these data and model configs, train the models, and compile them into a waifu.*

Now, there are two ways you can create your own AI: Python API and Console. The [Python API](python.md) is recommended for the experienced developers who wish to add their own functionalities to their AIs, and [the console](commands.md) is recommended for beginners who like a more user-friendly experienced without writing code (you will be using commands instead).