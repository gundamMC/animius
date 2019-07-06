# Data

## Overview

### createData

Create a data with empty values.

Available data types: 'Chatbot', 'IntentNER', 'SpeakerVerification'.

```
createData -n 'data name' -t 'DataType'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data

* *-t, --type* (`str`) -- Type of data (based on the model)

### dataAddEmbedding

Add word embedding to data

```
dataAddEmbedding -n 'data name' -e 'embedding name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-e, --embedding* (`str`) -- Name of embedding

### dataReset

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

### exportData

Export a data to zip file.

```
exportData -n 'data name' -p 'some\path\to\export\'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to export

* *-p, --path* (`str`) -- Path to export file

### importData

Import a data from zip file.

```
importModel -n 'data name' -p 'some\path\to\export\data_name.zip'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to import

* *-p, --path* (`str`) -- Path to import file

###getData

Get a list of existing data.

```
getData
```

No argument required.

This command returns a dictionary of which the keys are the name of data and the values are the details.

The details will be empty if the data is not loaded.

```
{
	"data_name": {
		"name": "data_name",
		"type": "<class 'data _class'>"
	}
}
```

### getDataDetails

Return the details of a data.

```
getDataDetails -n 'data name'
```

Keyword Arguments:

* *-n, --name* (``str``) -- Name of data

```
{
	'model_config_saved_directory': 'resources\\model_configs',
	'model_config_saved_name': 'model_config_name',
	'model_config_name': 'model_config_name',
	'embedding_saved_directory': 'resources\\embeddings\\embedding_name',
	'embedding_saved_name': 'embedding_name',
	'embedding_name': 'embedding_name',
	'cls': 'ChatbotData',
	'values': ['arr_0', 'embedding']
}
```

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

### chatbotDataAddFiles

Add text files to a chatbot data.

```
chatbotDataAddFile -n 'data name' -x '\some\path\x.txt' -y '\some\path\y.txt'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-x, --x_path* (`str`) -- Path to a UTF-8 file containing a raw sentence input on each line

* *-y, --y_path* (`str`) -- Path to a UTF-8 file containing a raw sentence output on each line

### chatbotDataAddInput

Add input to a chatbot data.

```
chatbotDataAddInput -n 'data name' -x 'hey how are you'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-x, --x* (`str`) -- Raw sentence input

### chatbotDataSetInput

Set input as a chatbot data.

```
chatbotDataSetInput -n 'data name' -x 'hey how are you'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to set

* *-x, --x* (`str`) -- Raw sentence input

## IntentNER Data

### intentNERDataSetIntentfolder

Set folder for IntentNER Data.

```
intentNERDataSetIntentFolder -n 'data name' -p '\\some\\path\\to\\intents'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-p, --path* (`str`) -- Path to a folder contains input files

### intentNERDataAddInput

Add a raw sentence as an intent NER data.

```
intentNERDataAddParseInput -n 'data name' -x 'hey how are you'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-x, --x* (`str`) -- Raw sentence input

### intentNERDataSetInput

Set a raw sentence as an intent NER data.

```
intentNERDataSetParseInput -n 'data name' -x 'hey how are you'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to set

* *-x, --x* (`str`) -- Raw sentence input

## SpeakerVerification Data

### speakerVerificationDataAddDataFolder

Add folder to a speaker verification data.

```
speakerVerificationDataAddDataFolder -n 'data name' -p 'path' [-y True]
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-p, -path* (`str`) -- Path of folder to set

* *-y, --y* (`bool`) -- The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)

### speakerVerificationDataSetDataFolder

set folder to a speaker verification data.

```
speakerVerificationDataSetDataFolder -n 'data name' -p 'path' [-y True]
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to set

* *-p, -path* (`str`) -- Path of folder to set

* *-y, --y* (`bool`) -- The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)


### speakerVerificationDataAddWavFile

Add wav file to a speaker verification data.

```
speakerVerificationDataAddWavFile -n 'data name' -p '["\some\path\01.wav"]' [-y True]
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-p, --path* (`str`) -- Path to wav file

* *-y, --y* (`bool`) -- The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)

### speakerVerificationDataSetWavFile

Set wav file to a speaker verification data.

```
speakerVerificationDataSetWavFile -n 'data name' -p '["\some\path\01.wav"]' [-y True]
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to set

* *-p, --path* (`str`) -- Path to wav file

* *-y, --y* (`bool`) -- The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)

### speakerVerificationDataAddTextFile

Add text file to a speaker verification data.

```
speakerVerificationDataAddTextFile -n 'data name' -p '["\some\path\01.txt"]' [-y True]
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to add on

* *-p, --path* (`str`) -- Path to text file

* *-y, --y* (`bool`) -- The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)

### speakerVerificationDataSetTextFile

Set text file to a speaker verification data.

```
speakerVerificationDataSetTextFile -n 'data name' -p '["\some\path\01.txt"]' [-y True]
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of data to set

* *-p, --path* (`str`) -- Path to text file

* *-y, --y* (`bool`) -- The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)
