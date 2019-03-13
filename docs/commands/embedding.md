# Embedding

###getEmbeddings

Get a list of existing word embeddings.

```
getEmbeddings
```

No argument required.

### createEmbedding

Create a word embedding.

```
createEmbedding -n 'embedding name' -p '\some\path\embedding.txt' [-v 100000]
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of embedding

* *-p, --path* (`str`) -- Path to embedding file

* *-v, --vocab_size* (`int`) -- Maximum number of tokens to read from embedding file (Optional)

### deleteEmbedding

Delete a word embedding.

```
deleteEmbedding -n 'embedding name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of embedding to delete

### saveEmbedding

Save an embedding.

```
saveEmbedding -n 'embedding name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of embedding to save

### loadEmbedding

Load an embedding.

```
loadEmbedding -n 'embedding name'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of embedding to load

## exportEmbedding

Export an embedding to zip file.

```
exportEmbedding -n 'embedding name' -p 'some\path\to\export\'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of embedding to export

* *-p, --path* (`str`) -- Path to export file

### importEmbedding

Import an embedding from zip file.

```
importModel -n 'embedding name' -p 'some\path\to\export\embedding_name.zip'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of embedding to import

* *-p, --path* (`str`) -- Path to import file

###getEmbeddings

Get a list of existing embedding.

```
getEmbeddings
```

No argument required.

This command returns a dictionary of which the keys are the name of embeddings and the values are the details.

The details will be empty if the embedding is not loaded.

```
{
	"embedding_name": {
		"name": "embedding_name"
	}
}
```

### getEmbeddingDetails

Return the details of an embedding.

```
getEmbeddingDetails -n 'embedding name'
```

Keyword Arguments:

* *-n, --name* (``str``) -- Name of embedding

```
{	
    'name': 'embedding_name',
	'saved_directory': 'resources\\embeddings\\embedding_name',
	'saved_name': 'embedding_name'
}
```
