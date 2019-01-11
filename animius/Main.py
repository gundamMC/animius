import animius as am
import readline


def completer(user_input, state):
    options = [i for i in commands if i.startswith(user_input)]
    if state < len(options):
        return options[state]
    else:
        return None


readline.parse_and_bind("tab: complete")
readline.set_completer(completer)

console = am.Console()

commands = {
    # command : console_method, short-to-long arguments dict, help message
    'waifus': [console.get_waifus(), {}, ''],
    'models': [console.get_models, {}, ''],
    'model_configs': [console.get_model_configs, {}, ''],
    'data': [console.get_data, {}, ''],
    'embeddings': [console.get_embeddings, {}, ''],

    'createWaifu': [console.create_waifu, {'n': 'name', 'cm': 'combined_chatbot_model'},
                    {'createWaifu': "-n 'waifu name' -c 'name of model' -e 'name of embedding",
                     '-n,--name (str)': '-- Name of waifu',
                     '-c, --combined_chatbot_model (str) ': '-- Name or directory of combined chatbot model to use',
                     '-e, --embedding (str)': '-- Name of word embedding to use'}],

    'deleteWaifu': [console.delete_waifu, {'n': 'name'},
                    {'deleteWaifu': "-n 'waifu name'", '=-n, --name (str)': '-- Name of waifu to delete'}],

    'saveWaifu': [console.save_waifu, {'n': 'name'},
                  {'saveWaifu': "-n 'waifu name'", '- n, --name(str)': ' - - Name of waifu to save'}],

    'loadWaifu': [console.load_waifu, {'n': 'name'},
                  {'loadWaifu': "-n 'waifu name'", '-n, --name (str)': '-- Name of waifu to load'}],

    'getWaifuDetail': [console.get_waifu_detail, {'n': 'name'}, ''],

    'createModel': [console.create_model, {'n': 'name', 't': 'type'},
                    {'createModel': "-n 'model name' -t 'ModelType'", '-n, --name (str)': 'Name of model',
                     '-t, --type (str)': '-- Type of model'}],

    'deleteModel': [console.delete_model, {'n': 'name'},
                    {'deleteModel': "-n 'model name'", '-n, --name (str)': ' -- Name of model to delete'}],

    'saveModel': [console.save_model, {'n': 'name'},
                  {'saveModel': "-n 'model name'", '-n, --name (str)': '-- Name of model to save'}],

    'loadModel': [console.load_model, {'n': 'name', 'd': 'data'},
                  {'loadModel': "-n 'model name' -d 'data name'", '-n, --name (str)': '-- Name of model to load',
                   '-d, --data (str)': '-- Name of data to set to model'}],

    'getModelDetails': [console.get_model_details, {'n': 'name'}, ''],

    'setData': [console.set_data, {'n': 'name', 'd': 'data'},
                {'setData': "-n 'model name' -d 'data name'", '-n, --name (str)': '-- Name of model',
                 '-d, --data (str)': '-- Name of data'}],

    'train': [console.train, {'n': 'name', 'e': 'epoch'},
              {'train': "-n 'model name' -e 10", '-n, --name (str)': '-- Name of model to train',
               '-e, --epoch (int)': '-- Number of epochs to train for'}],

    'predict': [console.predict, {'n': 'name', 's': 'save_path', 'i': 'input_data'},
                {'predict': "-n 'model name' -i 'name of input data' -s '\some\path.txt'",
                 '-n, --name (str)': '-- Name of model', '-i, --input_data (str)': '-- Name of input data',
                 '-s, --save_path (str)': '-- Path to save result (Optional)'}],

    'createModelConfig': [console.create_model_config, {'n': 'name', 'c': 'cls'},
                          {
                              'createModelConfig': "-n 'model config name' -t 'type' [-c '{'some_key': 'some_value'}'] [-h '{}'] [-ms '{}']",
                              "-n, --name (str)": '-- Name of model config',
                              "-t, --type (str)": '-- Name of the model type',
                              '-c, --config (dict)': ' -- Dictionary of config values (Optional)',
                              '-h, --hyperparameters (dict)': '-- Dictionary of hyperparameters values (Optional)',
                              '-s, --model_structure (dict)': '-- Dictionary of model_structure values (Optional)'}],

    'editModelConfig': [console.edit_model_config,
                        {'n': 'name', 'c': 'config', 'h': 'hyperparameters', 'ms': 'model_structure'}, {
                            'editModelConfig': ''''-n 'model config name' [-c '{"some_key": "some_value"}'] [-h '{}'] [-s '{}']''',
                            '-n, --name (str)': ' -- Name of model config to edit',
                            '-c, --config (dict)': '-- Dictionary containing the updated config values (Optional)',
                            '-h, --hyperparameters (dict) ': '-- Dictionary containing the updated hyperparameters values (Optional)',
                            '-s, --model_structure (dict)': ' -- Dictionary containing the updated model structure values (Optional)'}],

    'deleteModelConfig': [console.delete_model_config, {'n': 'name'}, {'deleteModelConfig': "-n 'model config name",
                                                                       '-n, --name (str)': '-- Name of model config to delete'}],

    'saveModelConfig': [console.save_model_config, {'n': 'name'}, {'saveModelConfig': "-n 'model config name'",
                                                                   '-n, --name (str)': '-- Name of model config to save'}],

    'loadModelConfig': [console.load_model_config, {'n': 'name'}, {'loadModelConfig': "-n 'model config name'",
                                                                   '-n, --name (str)': '-- Name of model config to load'}],

    'createData': [console.create_data, {'n': 'name', 't': 'type', 'mc': 'model_config'}, {}],
    'dataAddEmbedding': [console.data_add_embedding, {'n': 'name', 'ne': 'name_embedding'},
                         {'createData': "-n 'data name' -t 'ModelType' -c 'model config name'",
                          '-n, --name (str)': '-- Name of data',
                          '-t, --type (str)': '-- Type of data (based on the model)',
                          '-c, --model_config (str)': '-- Name of model config'}],

    'dataReset': [console.data_reset, {'n': 'name'},
                  {'dataReset': "-n 'data name'", '-n, --name (str)': '-- Name of data to reset'}],

    'deleteData': [console.delete_data, {'n': 'name'},
                   {'deleteData': "-n 'data name'", '-n, --name (str)': '-- Name of data to delete'}],

    'saveData': [console.save_data, {'n': 'name'},
                 {'saveData': "-n 'data name'", '-n, --name (str)': '-- Name of data to save'}],

    'loadData': [console.load_data, {'n': 'name'},
                 {'loadData': "-n 'data name'", '-n, --name (str)': '-- Name of data to load'}],

    'getDataDetails': [console.get_data_details, {'n': 'name'}, ''],

    'chatbotDataAddTwitter': [console.chatbot_data_add_twitter, {'n': 'name', 'p': 'path'},
                              {'chatbotDataAddTwitter': "-n 'data name' -p '\some\path\twitter.txt'",
                               '-n, --name (str)': '-- Name of data to add on',
                               '-p, --path (str)': '-- Path to twitter file'}],

    'chatbotDataAddCornell': [console.chatbot_data_add_cornell,
                              {'n': 'name', 'mcp': 'movie_conversations_path', 'mlp': 'movie_lines_path'}, {
                                  'chatbotDataAddCornell': "-n 'data name' -mcp '\some\cornell\movie_conversations.txt' -mlp '\some\cornell\movie_lines.txt'",
                                  '-n, --name (str)': '-- Name of data to add on',
                                  '-mcp, --movie_conversations_path (str)': '-- Path to movie_conversations.txt in the Cornell dataset',
                                  '-mlp, --movie_lines_path (str)': '-- Path to movie_lines.txt in the Cornell dataset'}],

    'chatbotDataAddParseSentences': [console.chatbot_data_add_parse_sentences, {'n': 'name', 'x': 'x', 'y': 'y'},
                                     {'chatbotDataAddParseSentences': ''''-n 'data name' -x '["some input"]' -y '["some
                                      output"]''''', '-n, --name (str)': '-- Name of data to add on',
                                      '-x, --x (list<str>)': '-- List of strings, each representing a sentence input',
                                      '-y, --y (list<str>)': '-- List of strings, each representing a sentence output'}],

    'chatbotDataAddParseFile': [console.chatbot_data_add_parse_file, {'n': 'name', 'x': 'x', 'y': 'y'}, {
        'chatbotDataAddParseFile': "-n 'data name' -x '\some\path\\x.txt' -y '\some\path\y.txt'",
        '-n, --name (str)': '-- Name of data to add on',
        '-x, --x_path (str)': '-- Path to a UTF-8 file containing a raw sentence input on each line',
        '-y, --y_path (str) ': '-- Path to a UTF-8 file containing a raw sentence output on each line'}],

    'chatbotDataAddParseInput': [console.chatbot_data_add_parse_input, {'n': 'name', 'x': 'x'},
                                 {'chatbotDataAddParseInput': "-n 'data name' -x 'hey how are you'",
                                  '-n, --name (str)': '-- Name of data to add on',
                                  '-x, --x (str)': '-- Raw sentence input'}],

    'chatbotDataSetParseInput': [console.chatbot_data_set_parse_input, {'n': 'name', 'x': 'x'},
                                 {'chatbotDataSetParseInput': "-n 'data name' -x 'hey how are you'",
                                  '-n, --name (str)': '-- Name of data to set',
                                  '-x, --x (str)': ' -- Raw sentence input'}],

    'intentNERDataAddParseInput': [console.intentNER_data_add_parse_input, {'n': 'name', 'x': 'x'},
                                   {'intentNERDataAddParseInput': "-n 'data name' -x 'hey how are you'",
                                    '-n, --name (str)': '-- Name of data to add on',
                                    '-x, --x (str)': '-- Raw sentence input'}],

    'intentNERDataSetParseInput': [console.intentNER_data_set_parse_input, {'n': 'name', 'x': 'x'},
                                   {'intentNERDataSetParseInput': "-n 'data name' -x 'hey how are you'",
                                    '-n, --name (str)': '-- Name of data to set',
                                    '-x, --x (str)': '-- Raw sentence input'}],

    'intentNERDataAddParseDatafolder': [console.intentNER_data_add_parse_data_folder,
                                        {'n': 'name', 'fd': 'folder_directory'},
                                        {
                                            'intentNERDataAddParseDatafolder': "-n 'data name' -p '\some\path\\to\intents'",
                                            '-n, --name (str)': ' -- Name of data to add on',
                                            '-p, --path (str)': ' -- Path to a folder contains input files'}],

    'speakerVerificationDataAddDataPaths': [console.speakerVerification_data_add_data_paths,
                                            {'n': 'names', 'p': 'paths', 'y': 'y'},
                                            {
                                                'speakerVerificationDataAddDataPaths': '''-n 'data name' -p '["\some\path\\01.wav"]' [-y True]''',
                                                '-n, --name (str)': '-- Name of data to add on',
                                                '-p, -path (list<str>)': '-- List of string paths to raw audio files',
                                                '-y, --y (bool)': '-- The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)'}],

    'speakerVerificationDataAddDataFile': [console.speakerVerification_data_add_data_file,
                                           {'n': 'names', 'p': 'paths', 'y': 'y'}, {
                                               'speakerVerificationDataAddDataFile': '''-n 'data name' -p '\some\path\audios.txt' -y True''',
                                               '-n, --name (str)': '-- Name of data to add on',
                                               '-p, --path (str)': '-- Path to file containing a path of a raw audio file on each line',
                                               '-y, --y (bool)': '-- The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)'}],

    'createEmbedding': [console.create_embedding, {'n': 'name', 'p': 'path'},
                        {'createEmbedding': "-n 'embedding name' -p '\some\path\embedding.txt' [-v 100000]",
                         '-n, --name (str)': ' -- Name of embedding', '-p, --path (str)': '-- Path to embedding file',
                         '-v, --vocab_size (int)': '-- Maximum number of tokens to read from embedding file (Optional)'}],

    'deleteEmbedding': [console.delete_embedding, {'n': 'name'}, {'deleteEmbedding': "-n 'embedding name'",
                                                                  '-n, --name (str)': '-- Name of embedding to delete'}],

    'saveEmbedding': [console.save_embedding, {'n': 'name'},
                      {'saveEmbedding': "-n 'embedding name'", '-n, --name (str)': '-- Name of embedding to save'}],

    'loadEmbedding': [console.load_embedding, {'n': 'name'},
                      {'loadEmbedding': "-n 'embedding name'", '-n, --name (str)': '-- Name of embedding to load'}],

    'startServer': [console.start_server, {'p': 'port', 'pwd': 'pwd', 'mc': 'max_clients', 'l': 'local'},
                    {'startServer': "-p 23333 [-l True] [-pwd 'p@ssword'] [-c 10]",
                     '-p, --port (int)': '-- Port to listen on',
                     '-l, --local (bool)': '-- If the server is running locally (server will listen on 127.0.0.1 if this is true or not set) (Optional)',
                     '-pwd, --password (str)': '-- Password of server (Optional)',
                     '-c, --max_clients (int)': '-- Maximum number of clients (Optional)'}],
    'stopServer': [console.stop_server, {}, {'stopServer': ''}],
    'freezeGraph': [console.freeze_graph, {'md': 'model_dir', 'o': 'output_node_names',
                                           's': 'stored_model_config'},
                    {'freezeGraph': "-n 'model name'", '-n, --name (str)': ' -- Name of model'}],
    'optimize': [console.optimize, {'md': 'model_dir', 'o': 'output_node_names',
                                    'i': 'input_node_names'},
                 {'optimize': "-n 'model name'", '-n, --name (str)': '-- Name of waifu to load'}],
    'save': [console.save, {}, {'save': ''}],
    's': [console.save, {}, {'save': ''}],
}

print("Animius. Type 'help' or '?' to list commands.")

while True:
    user_input = input('Input: ')

    if user_input.lower() == 'exit':
        break
    elif user_input.lower() == 'help' or user_input == '?':
        continue
    elif user_input.lower() == 'about' or user_input.lower() == 'version':
        continue
    elif user_input is None or not user_input.strip():  # empty string gives false
        continue
    else:
        command, args = am.Console.ParseArgs(user_input)

        print('command:', command)
        print('args', args)

        if command is None:
            break
        elif command in commands:
            if '--help' in args:
                for i in commands[command][2]:
                    print(i, commands[command][2][i])
            else:
                # valid command and valid args

                # change arguments into kwargs for passing into console
                kwargs = {}
                for arg in args:
                    if arg[:2] == '--':  # long
                        kwargs[arg[2:]] = args[arg]
                    elif arg[:1] == '-':  # short
                        if arg[1:] not in commands[command][1]:
                            print("Invalid short argument {0}, skipping it".format(arg))
                            continue

                        long_arg = commands[command][1][arg[1:]]
                        kwargs[long_arg] = args[arg]

                print(kwargs)
                print(commands[command][0])
                print('==================================================')

                try:
                    result = commands[command][0].__call__(**kwargs)
                    if result is not None:
                        print(result)
                except Exception as exc:
                    print('{0}: {1}'.format(type(exc).__name__, exc))
        else:
            print('Invalid command')
