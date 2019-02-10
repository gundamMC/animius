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
