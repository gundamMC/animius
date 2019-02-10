# Data

## Overview

###getData

Get a list of existing data.

```
getData
```

No argument required.

### createData

Create a data with empty values.

```
createData -n 'data name' -t 'ModelType' -c 'model config name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data

* *-t, --type* (`str`) -- Type of data (based on the model)

* *-c, --model_config* (`str`) -- Name of model config

### dataAddEmbedding

Add word embedding to data

```
dataAddEmbedding -n 'data name' -e 'embedding name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-e, --embedding* (`str`) -- Name of embedding

### resetData

Reset a data, clearing all stored data values.

```
dataReset -n 'data name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to reset

### deleteData

Delete a data.

```
deleteData -n 'data name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to delete

### saveData

Save a data.

```
saveData -n 'data name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to save

### loadData

Load a data.

```
loadData -n 'data name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to load

### getDataDetails

Return the details of a data.

```
getDataDetails -n 'data name'
```

Keyword Arguments:

* *-n, --name* (``str``) -- Name of data


## Chatbot Data

### chatbotDataAddTwitter

Add twitter dataset to a chatbot data.

```
chatbotDataAddTwitter -n 'data name' -p '\some\path\twitter.txt'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-p, --path* (`str`) -- Path to twitter file

### chatbotDataAddCornell

Add Cornell dataset to a chatbot data.

```
chatbotDataAddCornell -n 'data name' -mcp '\some\cornell\movie_conversations.txt' -mlp '\some\cornell\movie_lines.txt'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-mcp, --movie_conversations_path* (`str`) -- Path to movie_conversations.txt in the Cornell dataset

* *-mlp, --movie_lines_path* (`str`) -- Path to movie_lines.txt in the Cornell dataset

### chatbotDataAddParseSentences

Parse raw sentences and add them to a chatbot data.

```
chatbotDataAddParseSentences -n 'data name' -x '["some input"]' -y '["some output"]'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-x, --x* (`list<str>`) -- List of strings, each representing a sentence input

* *-y, --y* (`list<str>`) -- List of strings, each representing a sentence output

### chatbotDataAddParseFile

Parse raw sentences from text files and add them to a chatbot data.

```
chatbotDataAddParseFile -n 'data name' -x '\some\path\x.txt' -y '\some\path\y.txt'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-x, --x_path* (`str`) -- Path to a UTF-8 file containing a raw sentence input on each line

* *-y, --y_path* (`str`) -- Path to a UTF-8 file containing a raw sentence output on each line

### chatbotDataAddParseInput

Parse a raw sentence as input and add it to a chatbot data.

```
chatbotDataAddParseInput -n 'data name' -x 'hey how are you'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-x, --x* (`str`) -- Raw sentence input

### chatbotDataSetParseInput

Parse a raw sentence as input and set it as a chatbot data.

```
chatbotDataSetParseInput -n 'data name' -x 'hey how are you'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to set

* *-x, --x* (`str`) -- Raw sentence input

## IntentNER Data

### intentNERDataAddParseDatafolder

Parse files from a folder and add them to a chatbot data.

```
intentNERDataAddParseDatafolder -n 'data name' -p '\some\path\to\intents'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-p, --path* (`str`) -- Path to a folder contains input files

### intentNERDataAddParseInput

Parse a raw sentence as input and add it to an intent NER data.

```
intentNERDataAddParseInput -n 'data name' -x 'hey how are you'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-x, --x* (`str`) -- Raw sentence input

### intentNERDataSetParseInput

Parse a raw sentence as input and set it as an intent NER data.

```
intentNERDataSetParseInput -n 'data name' -x 'hey how are you'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to set

* *-x, --x* (`str`) -- Raw sentence input

## SpeakerVerification Data

### speakerVerificationDataAddDataPaths

Parse and add raw audio files to a speaker verification data.

```
speakerVerificationDataAddDataPaths -n 'data name' -p '["\some\path\01.wav"]' [-y True]
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-p, -path* (`list<str>`) -- List of string paths to raw audio files

* *-y, --y* (`bool`) -- The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)

### speakerVerificationDataAddDataFile

Read paths to raw audio files and add them to a speaker verification data.

```
speakerVerificationDataAddDataFile -n 'data name' -p '\some\path\audios.txt' -y True
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-p, --path* (`str`) -- Path to file containing a path of a raw audio file on each line

* *-y, --y* (`bool`) -- The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)
