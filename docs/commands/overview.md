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

##Model
###Create Model Config

Create a model config with the provided values.

```
createModelConfig --name='myModelConfig' --cls='myModelClass'
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
deleteModelConfig --name='myModel'
```
Keyword Argument:

* *name* (``str``) -- Name of model config to delete

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
chatbotDataAddParseInput --name='myData' --x=''
```
Keyword Arguments:

*  *name* (``str``) -- Name of data to add on

* *x* (``str``) -- Raw sentence input

###Set Parse Input To Chatbot Data
Parse a raw sentence as input and set it as a chatbot data.
```
chatbotDataSetParseInput --name='myData' --x=''
```
Keyword Arguments:

*  *name* (``str``) -- Name of data to set

* *x* (``str``) -- Raw sentence input

##IntentNER Data
###Add Parse Input To IntentNER Data
Parse a raw sentence as input and add it to an intent NER data.
```
intentNERDataAddParseInput --name='myData' --x=''
```
Keyword Arguments:

*  *name* (``str``) -- Name of data to add on

* *x* (``str``) -- Raw sentence input

###Set Parse Input To IntentNER Data
Parse a raw sentence as input and set it as an intent NER data.
```
intentNERDataSetParseInput --name='myData' --x=''
```
Keyword Arguments:

*  *name* (``str``) -- Name of data to set

* *x* (``str``) -- Raw sentence input

###intentNER_data_add_parse_data_folder
###intentNER_data_add_parse_input_file

##SpeakerVerification Data
###Add Data Paths To SpeakerVerification Data
Parse and add raw audio files to a speaker verification data.
```
speakerVerificationDataAddDataPaths --name='myData' --paths='' --y=''
```
Keyword Arguments:

* *name* (``str``) -- Name of data to add on

* *paths* (``list<str>``) -- List of string paths to raw audio files

* *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction. (Optional)

###Add Data File To SpeakerVerification Data
Read paths to raw audio files and add them to a speaker verification data.
```
speakerVerificationDataAddDataFile --name='myData' --paths='' --y=''
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

##Delete Embedding
Delete a word embedding.
```
deleteEmbedding --name='myEmbedding'
```
Keyword Arguments:

* *name* (``str``) -- Name of embedding to delete