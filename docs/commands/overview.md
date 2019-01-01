# Commands Overview

Commands are used to interact with the console as well as through the network socket. A command has two parts: the command and the arguments. 
In the console, a command could look something like this:
```
createModel --name='myModel' --model='SpeakerVerification'
```
The command must start with the command, indi

The equivalent of this in the network socket would be:
``` JSON
{
  "command": "createModel",
  "arguments": {
    "name": "myModel",
    "model": "SpeakerVerification"
  }
}
```

The various commands and their arguments can be found in this section. Network-socket-related commands can be found under the [Network section](https://gundammc.github.io/Animius/network/overview/).
##Waifu
###Create Waifu
Create a waifu.
```
createWaifu --name='myWaifu' --model='myModel'
```
Keyword Arguments:

* *name* (``str``) -- Name of waifu

* *model* (``str``) -- Name of model to use

###Delete Waifu
Delete a waifu.
```
deleteWaifu --name='myWaifu'
```
Keyword Arguments:

* *name* (``str``) -- Name of waifu to delete

###Save Waifu
Save a waifu.
```
saveWaifu --name='myWaifu'
```
Keyword Arguments:

* *name* (``str``) -- Name of waifu to save

###Load Waifu
Load a waifu.
```
loadWaifu --name='myWaifu'
```
Keyword Arguments:

* *name* (``str``) -- Name of waifu to load

##Model
###Create Model
Create a model.
```
createModel --name='myModel' --type='ChatbotModel'
```
Keyword Arguments:

* *name* (``str``) -- Name of model

* *type* (``str``) -- Type of model

###Delete Model
Delete a model.
```
deleteModel --name='myModel'
```
Keyword Arguments:

* *name* (``str``) -- Name of model to delete

###Save Model
Save a model.
```
saveModel --name='myModel'
```
Keyword Arguments:

* *name* (``str``) -- Name of model to save

###Load Model
Load a model.
```
loadModel --name='myModel' --data='myData'
```
Keyword Arguments:

* *name* (``str``) -- Name of model to load

* *data* (``str``) -- Name of data to load

###Set Data
Set model data.
```
setData --name='myModel' --data='myData'
```
Keyword Arguments:

* *name* (``str``) -- Name of model to set

* *data* (``str``) -- Name of data to set

###Train
Train a model.
```
train --name='myModel' --epoch=20
```
Keyword Arguments:

* *name* (``str``) -- Name of model to set

* *epoch* (``int``) -- Number of epoch

###Predict
Predict a model.
```
predict --name='myModel' --input_data='' --save_path=''
```
Keyword Arguments:

* *name* (``str``) -- Name of model to predict

* *input_data* (``str``) -- Name of input data
        
* *save_path* (``str``) -- Path to save result (Optional)

##ModelConfig
###Create Model Config

Create a model config with the provided values.
```
createModelConfig --name='myModelConfig' --cls=''
```
Keyword Arguements:

* *name* (``str``) -- Name of model config

* *cls* (``str``) -- Name of the model class

* *config* (``dict``) -- Dictionary of config values (Optional)

* *hyperparameters* (``dict``) -- Dictionary of hyperparameters values (Optional)

* *model_structure* (``model_structure``) -- Dictionary of model_structure values (Optional)

###Edit Model Config

Update a model config with the provided values.
```
createModelConfig --name='myModelConfig'
```
Keyword Arguments:

* *name* (``str``) -- Name of model config to edit

* *config* (``dict``) -- Dictionary containing the updated config values

* *hyperparameters* (``dict``) -- Dictionary containing the updated hyperparameters values

* *model_structure* (``model_structure``) -- Dictionary containing the updated model_structure values

###Delete Model Config

Delete a model config.
```
deleteModelConfig --name='myModelConfig'
```
Keyword Argument:

* *name* (``str``) -- Name of model config to delete

###Save Model Config

Save a model config.
```
saveModelConfig --name='myModelConfig'
```
Keyword Argument:

* *name*(``str``) -- Name of model config to save

###Load Model Config

Load a model config.
```
loadModelConfig --name='myModelConfig'
```
Keyword Argument:

* *name*(``str``) -- Name of model config to load

##Data
###Create Data

Create a data with empty values.
```
createData --name='myData' --type='ChatbotData' --model_config='myModelConfig'
```
Keyword Arguments:

* *name* (``str``) -- Name of data

* *type* (``str``) -- Type of data (based on the model)

* *model_config* (``str``) -- Name of model config

###Add Embedding To Data
Add twitter dataset to a chatbot data.
```
dataAddEmbedding --name='myData' --name_embedding='myEmbedding'
```
Keyword Arguments:

* *name* (``str``) -- Name of data to add on

* *name_embedding* (``str``) -- Name of the embedding to add to data

###Reset Data
Reset a data, clearing all stored data values.
```
dataReset --name='myData'
```
Keyword Arguments:

* *name* (``str``) -- Name of data to reset

###Delete Data
Delete a data.
```
deleteData --name='myData'
```
Keyword Arguments:

* *name* (``str``) -- Name of data to delete

###Save Data
Save a data.
```
saveData --name='myData'
```
Keyword Arguments:

* *name* (``str``) -- Name of data to save

###Load Data
Load a data.
```
loadData --name='myData'
```
Keyword Arguments:

* *name* (``str``) -- Name of data to load

##Chatbot Data
###Add Twitter To Chatbot Data
Add twitter dataset to a chatbot data.
```
chatbotDataAddTwitter --name='myData' --path=''
```
Keyword Arguments:

* *name* (``str``) -- Name of data to add on

* *path* (``str``) -- Path to twitter file

###Add Cornell To Chatbot Data
Add Cornell dataset to a chatbot data.
```
chatbotDataAddTwitter --name='myData' --movie_conversations_path='' --movie_lines_path=''
```
Keyword Arguments:

* *name* (``str``) -- Name of data to add on

* *movie_conversations_path* (``str``) -- Path to movie_conversations.txt in the Cornell dataset

* *movie_lines_path* (``str``) -- Path to movie_lines.txt in the Cornell dataset

###Add Parse Sentences To Chatbot Data
Parse raw sentences and add them to a chatbot data.
```
chatbotDataAddParseSentences --name='myData' --x='' --y=''
```
Keyword Arguments:

* *name* (``str``) -- Name of data to add on

* *x* (``list<str>``) -- List of strings, each representing a sentence input

* *y* (``list<str>``) -- List of strings, each representing a sentence output

###Add Parse File To Chatbot Data
Parse raw sentences from text files and add them to a chatbot data.
```
chatbotDataAddParseFile --name='myData' --x_path='' --y_path=''
```
Keyword Arguments:

* *name* (``str``) -- Name of data to add on

* *x_path* (``str``) -- Path to a UTF-8 file containing a raw sentence input on each line

* *y_path* (``str``) -- Path to a UTF-8 file containing a raw sentence output on each line

###Add Parse Input To Chatbot Data
Parse a raw sentence as input and add it to a chatbot data.
```
chatbotDataAddParseInput --name='myData' --x='hey how are you'
```
Keyword Arguments:

*  *name* (``str``) -- Name of data to add on

* *x* (``str``) -- Raw sentence input

###Set Parse Input To Chatbot Data
Parse a raw sentence as input and set it as a chatbot data.
```
chatbotDataSetParseInput --name='myData' --x='hey how are you'
```
Keyword Arguments:

*  *name* (``str``) -- Name of data to set

* *x* (``str``) -- Raw sentence input

##IntentNER Data
###Add Parse Input To IntentNER Data
Parse a raw sentence as input and add it to an intent NER data.
```
intentNERDataAddParseInput --name='myData' --x='hey how are you'
```
Keyword Arguments:

*  *name* (``str``) -- Name of data to add on

* *x* (``str``) -- Raw sentence input

###Set Parse Input To IntentNER Data
Parse a raw sentence as input and set it as an intent NER data.
```
intentNERDataSetParseInput --name='myData' --x='hey how are you'
```
Keyword Arguments:

*  *name* (``str``) -- Name of data to set

* *x* (``str``) -- Raw sentence input

###Add Parse Data Folder To IntentNER Data
Parse files from a folder and add them to a chatbot data.
```
intentNERDataAddParseDatafolder --name='myData' --folder_directory=''
```
Keyword Arguments:

* *name* (``str``) -- Name of data to add on

* *folder_directory* (``str``) -- Path to a folder contains input files

##SpeakerVerification Data
###Add Data Paths To SpeakerVerification Data
Parse and add raw audio files to a speaker verification data.
```
speakerVerificationDataAddDataPaths --name='myData' --paths='' --y=True
```
Keyword Arguments:

* *name* (``str``) -- Name of data to add on

* *paths* (``list<str>``) -- List of string paths to raw audio files

* *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction. (Optional)

###Add Data File To SpeakerVerification Data
Read paths to raw audio files and add them to a speaker verification data.
```
speakerVerificationDataAddDataFile --name='myData' --paths='' --y=True
```
Keyword Arguments:

* *name* (``str``) -- Name of data to add on

* *path* (``str``) -- Path to file containing a path of a raw audio file on each line

* *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction.(Optional)

##Embedding
###Create Embedding
Create a word embedding.
```
createEmbedding --name='myEmbedding' --path=''
```
Keyword Arguments:

* *name* (``str``) -- Name of embedding

* *path* (``str``) -- Path to embedding file

* *vocab_size* (``int``) -- Maximum number of tokens to read from embedding file (Optional)

###Delete Embedding
Delete a word embedding.
```
deleteEmbedding --name='myEmbedding'
```
Keyword Arguments:

* *name* (``str``) -- Name of embedding to delete

###Save Embedding
Save an embedding.
```
saveEmbedding --name='myEmbedding'
```
Keyword Arguments:

* *name* (``str``) -- Name of embedding to save

###Load Embedding
Load an embedding.
```
loadEmbedding --name='myEmbedding'
```
Keyword Arguments:

* *name* (``str``) -- Name of embedding to load

##Utils
###Start Server
Start server.
```
startServer --port=25565 --local=True --pwd='p@ssword' --max_clients=10
```
Keyword Arguments:

* *port* (``int``) -- Port of server

* *local* (``bool``) -- Decide if the server is running locally

* *pwd* (``str``) -- Password of server

* *max_clients* (``int``) -- Maximum number of clients

###Freeze Graph
Freeze checkpoints to a file.
```
freezeGraph --model_dir='' --output_node_names='' --stored_model_config=''
```
Keyword Arguments:

* *model_dir* (``str``) -- Path to your model

* *output_node_names* (``str``) -- Name of output nodes

* *stored_model_config* (``str``) -- Name of model config to use

###Optimize
Optimizing for inference.
```
optimize --model_dir='' --input_node_names='' --output_node_names=''
```
Keyword Arguments:

* *model_dir* (``str``) -- Path to your model
* *input_node_names* (``str``) -- Name of input nodes
* *output_node_names* (``str``) -- Name of output nodes