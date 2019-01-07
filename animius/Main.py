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

    'createWaifu': [console.create_waifu, {'n': 'name', 'cm': 'combined_chatbot_model'}, ''],
    'deleteWaifu': [console.delete_waifu, {'n': 'name'}, ''],
    'saveWaifu': [console.save_waifu, {'n': 'name'}, ''],
    'loadWaifu': [console.load_waifu, {'n': 'name'}, ''],

    'createModel': [console.create_model, {'n': 'name', 't': 'type'}, ''],
    'deleteModel': [console.delete_model, {'n': 'name'}, ''],
    'saveModel': [console.save_model, {'n': 'name'}, ''],
    'loadModel': [console.load_model, {'n': 'name', 'd': 'data'}, ''],

    'setData': [console.set_data, {'n': 'name', 'd': 'data'}, ''],
    'train': [console.train, {'n': 'name', 'e': 'epoch'}, ''],
    'predict': [console.predict, {'n': 'name', 's': 'save_path', 'i': 'input_data'}, ''],

    'createModelConfig': [console.create_model_config, {'n': 'name', 'c': 'cls'}, ''],
    'editModelConfig': [console.edit_model_config,
                        {'n': 'name', 'c': 'config', 'h': 'hyperparameters', 'ms': 'model_structure'}, ''],
    'deleteModelConfig': [console.delete_model_config, {'n': 'name'}, ''],
    'saveModelConfig': [console.save_model_config, {'n': 'name'}, ''],
    'loadModelConfig': [console.load_model_config, {'n': 'name'}, ''],

    'createData': [console.create_data, {'n': 'name', 't': 'type', 'mc': 'model_config'}, ''],
    'dataAddEmbedding': [console.data_add_embedding, {'n': 'name', 'ne': 'name_embedding'}, ''],
    'dataReset': [console.data_reset, {'n': 'name'}, ''],
    'deleteData': [console.delete_data, {'n': 'name'}, ''],
    'saveData': [console.save_data, {'n': 'name'}, ''],
    'loadData': [console.load_data, {'n': 'name'}, ''],

    'chatbotDataAddTwitter': [console.chatbot_data_add_twitter, {'n': 'name', 'p': 'path'}, ''],
    'chatbotDataAddCornell': [console.chatbot_data_add_cornell,
                              {'n': 'name', 'mcp': 'movie_conversations_path', 'mlp': 'movie_lines_path'}, ''],
    'chatbotDataAddParseSentences': [console.chatbot_data_add_parse_sentences, {'n': 'name', 'x': 'x', 'y': 'y'}, ''],
    'chatbotDataAddParseFile': [console.chatbot_data_add_parse_file, {'n': 'name', 'x': 'x', 'y': 'y'}, ''],
    'chatbotDataAddParseInput': [console.chatbot_data_add_parse_input, {'n': 'name', 'x': 'x'}, ''],
    'chatbotDataSetParseInput': [console.chatbot_data_set_parse_input, {'n': 'name', 'x': 'x'}, ''],

    'intentNERDataAddParseInput': [console.intentNER_data_add_parse_input, {'n': 'name', 'x': 'x'}, ''],
    'intentNERDataSetParseInput': [console.intentNER_data_set_parse_input, {'n': 'name', 'x': 'x'}, ''],
    'intentNERDataAddParseDatafolder': [console.intentNER_data_add_parse_data_folder,
                                        {'n': 'name', 'fd': 'folder_directory'}, ''],

    'speakerVerificationDataAddDataPaths': [console.speakerVerification_data_add_data_paths,
                                            {'n': 'names', 'p': 'paths', 'y': 'y'}, ''],
    'speakerVerificationDataAddDataFile': [console.speakerVerification_data_add_data_file,
                                           {'n': 'names', 'p': 'paths', 'y': 'y'}, ''],

    'createEmbedding': [console.create_embedding, {'n': 'name', 'p': 'path'}, ''],
    'deleteEmbedding': [console.delete_embedding, {'n': 'name'}, ''],
    'saveEmbedding': [console.save_embedding, {'n': 'name'}, ''],
    'loadEmbedding': [console.load_embedding, {'n': 'name'}, ''],

    'startServer': [console.start_server, {'p': 'port', 'pwd': 'pwd', 'mc': 'max_clients', 'l': 'local'}, ''],
    'stopServer': [console.stop_server, {}, ''],
    'freezeGraph': [console.freeze_graph, {'md': 'model_dir', 'o': 'output_node_names',
                                           's': 'stored_model_config'}, ''],
    'optimize': [console.optimize, {'md': 'model_dir', 'o': 'output_node_names',
                                    'i': 'input_node_names'}, ''],
    'save': [console.save, {}, ''],
    's': [console.save, {}, ''],
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
                print(commands[command][2])
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
