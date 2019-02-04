# Network Encryption

**IMPORTANT** | Animius no longer uses *ANY* encrytion. It sends plain strings/bytes instead. The following is *outdated*

------

In short, Animius uses AES-CBC + base64 encryption.

## Initial Connection

Upon connection, the server will send a unencrypted response containing the key and iv for AES-CBC. It should look something similar to:

``` JSON
{
  "id": "01:01",
  "status": 0,
  "message": "",
  "data": {
    "key": "base64 encoded string",
    "iv": "base64 encoded string"
  }
}
```

The client should decode the base64 strings to obtain the original bytes.

## Client sending

When sending a JSON-formatted request (see the overview page for details) to the server, it must be converted bytes through UTF-8. Then, encrypt the message with AES-CBC with the recevied keys and IVs. Convert the encrypted bytes to a base64 string. Finally, encode the string into bytes with UTF-8 and send it.

## Client receiving

Upon receiving the bytes, the client should decode it with UTF-8, obtaining a base64-encoded string. Then, decode the string with base64. Decrypt the decoded bytes using AES-CBC with the key and iv provided upon connection. Decode the bytes with UTF-8. The result should be a JSON string.
