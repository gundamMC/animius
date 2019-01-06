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
    'createWaifu': [console.create_waifu, {'n': 'name', 'cm': 'combined_chatbot_model'}, ''],
    'deleteWaifu': [console.delete_waifu, ''],
    'saveWaifu': [console.save_waifu, ''],
    'loadWaifu': [console.load_waifu, ''],

    'createModel': [console.create_model, ''],
    'deleteModel': [console.delete_model, ''],
    'saveModel': [console.save_model, ''],
    'loadModel': [console.load_model, ''],

    'setData': [console.set_data, ''],
    'train': [console.train, ''],
    'predict': [console.predict, ''],

    'createModelConfig': [console.create_model_config, ''],
    'deleteModelConfig': [console.delete_model_config, ''],
    'saveModelConfig': [console.save_model_config, ''],
    'loadModelConfig': [console.load_model_config, ''],

    'createData': [console.create_data, ''],
    'dataAddEmbedding': [console.data_add_embedding, ''],
    'dataReset': [console.data_reset, ''],
    'deleteData': [console.delete_data, ''],
    'saveData': [console.save_data, ''],
    'loadData': [console.load_data, ''],

    'chatbotDataAddTwitter': [console.chatbot_data_add_twitter, ''],
    'chatbotDataAddCornell': [console.chatbot_data_add_cornell, ''],
    'chatbotDataAddParseSentences': [console.chatbot_data_add_parse_sentences, ''],
    'chatbotDataAddParseFile': [console.chatbot_data_add_parse_file, ''],
    'chatbotDataAddParseInput': [console.chatbot_data_add_parse_input, ''],
    'chatbotDataSetParseInput': [console.chatbot_data_set_parse_input, ''],

    'intentNERDataAddParseInput': [console.intentNER_data_add_parse_input, ''],
    'intentNERDataSetParseInput': [console.intentNER_data_set_parse_input, ''],
    'intentNERDataAddParseDatafolder': [console.intentNER_data_add_parse_data_folder, ''],

    'speakerVerificationDataAddDataPaths': [console.speakerVerification_data_add_data_paths, ''],
    'speakerVerificationDataAddDataFile': [console.speakerVerification_data_add_data_file, ''],

    'createEmbedding': [console.create_embedding, ''],
    'deleteEmbedding': [console.delete_embedding, ''],
    'saveEmbedding': [console.save_embedding, ''],
    'loadEmbedding': [console.load_embedding, ''],

    'startServer': [console.start_server, ''],
    'stopServer': [console.stop_server, ''],
    'freezeGraph': [console.freeze_graph, ''],
    'optimize': [console.optimize, ''],
    'save': [console.save, ''],
    's': [console.save, ''],
}

print("Animius. Type 'help' or '?' to list commands.")

while True:
    user_input = input('Input: ')

    if user_input.lower() == 'exit':
        break
    elif user_input.lower() == 'help' or '?':
        pass
    elif user_input.lower() == 'about' or 'version':
        pass

    command, args = am.Console.ParseArgs(user_input)

    print(command)
    print(args)

    if command is None:
        continue
    elif command in commands:
        if '--help' in args:
            print(commands[command][1])
        else:
            # valid command and valid args

            # change arguments into kwargs for passing into console
            kwargs = {}
            for arg in args:
                if arg[0:2] == '--':  # long
                    kwargs[arg[3:]] = args[arg]
                elif arg[0:1] == '-':  # short
                    if arg[2:] not in command[command][1]:
                        print("Invalid short argument {0}, skipping it".format(arg))
                        continue

                    long_arg = command[command][1][arg[2:]]
                    kwargs[long_arg] = args[arg]

            commands[command][0](args)
    else:
        print('Invalid command')
