# Waifu

###getWaifu

Get a list of existing waifu.

```
getWaifu
```

No argument required.

### createWaifu

Create a waifu.

```
createWaifu -n 'waifu name' -c 'name of model' -e 'name of embedding'
```

Keyword Arguments:

* *-n, --name* (`str`) -- Name of waifu

* *-c, --combined_chatbot_model* (`str`) -- Name or directory of combined chatbot model to use

* *-e, --embedding* (`str`) -- Name of word embedding to use

### deleteWaifu

Delete a waifu.

```
deleteWaifu -n 'waifu name'
```

Keyword Arguments:

* *-n, --name* (``str``) -- Name of waifu to delete

### saveWaifu

Save a waifu.

```
saveWaifu -n 'waifu name'
```

Keyword Arguments:

* *-n, --name* (``str``) -- Name of waifu to save

### loadWaifu

Load a waifu.

```
loadWaifu -n 'waifu name'
```

Keyword Arguments:

* *-n, --name* (``str``) -- Name of waifu to load

### getWaifuDetail

Get the detail information of a waifu.

```
getWaifuDetail -n 'waifu name'
```

Keyword Arguments:

* *-n, --name* (``str``) -- Name of waifu

### waifuPredict

Make prediction using waifu.

```
getWaifuDetail -n 'waifu name'
```

Keyword Arguments:

* *-n, --name* (``str``) -- Name of waifu
