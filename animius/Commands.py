class Commands:

    def __getitem__(self, item):
        return self.comm_dict[item]

    def __iter__(self):
        return iter(self.comm_dict)

    def __init__(self, console):
        self.comm_dict = {
            # command : console_method, arguments dict [long arg, type, message], help message, example
            # (example not provided if there are no arguments)

            # region export commands
            'exportWaifu': [console.export_waifu(),
                            {
                                '-n': ['name', 'str', 'Name of waifu'],
                                '-p': ['path', 'str', 'Path to export file']
                            },
                            'Export a waifu',
                            "exportWaifu -n 'waifu name' -p 'path_name'"
                            ],

            'exportModel': [console.export_model_config,
                            {
                                '-n': ['name', 'str', 'Name of model'],
                                '-p': ['path', 'str', 'Path to export file']
                            },
                            'Export a model',
                            "exportModel -n 'model name' -p 'path_name'"
                            ],

            'exportModelConfig': [console.export_model_config,
                                  {
                                      '-n': ['name', 'str', 'Name of model config'],
                                      '-p': ['path', 'str', 'Path to export file']
                                  },
                                  'Export a model config',
                                  "exportModelConfig -n 'model config name' -p 'path_name'"
                                  ],

            'exportData': [],

            'exportEmbedding': [],

            # region import commands
            'importWaifu': [],

            'importModel': [],

            'importModelConfig': [],

            'importData': [],

            'importEmbedding': [],

            # region list commands
            'getWaifu': [console.get_waifu, {},
                         'Get a list of existing waifu.'
                         ],
            'getModels': [console.get_models, {},
                          'Get a list of existing models.'
                          ],
            'getModelConfigs': [console.get_model_configs, {},
                                'Get a list of existing model configs.'
                                ],

            'getData': [console.get_data, {},
                        'Get a list of existing data.'
                        ],

            'getEmbeddings': [console.get_embeddings, {},
                              'Get a list of existing word embeddings.'
                              ],
            # endregion

            # region waifu commands
            'createWaifu': [console.create_waifu,
                            {
                                '-n': ['name', 'str', 'Name of waifu'],
                                '-c': ['combined_chatbot_model', 'str',
                                       'Name or directory of combined chatbot model to use'],
                                '-e': ['embedding', 'str', 'Name of word embedding to use'],
                                '-d': ['description', 'str', 'Description of waifu (Optional)']
                            },
                            'Create a waifu.',
                            'createWaifu -n \'waifu name\' -c \'name of model\' -e \'name of embedding\''
                            ],

            'deleteWaifu': [console.delete_waifu,
                            {
                                '-n': ['name', 'str', 'Name of waifu to delete']
                            },
                            'Delete  a waifu.',
                            'deleteWaifu -n \'waifu name\''
                            ],

            'saveWaifu': [console.save_waifu,
                          {
                              '-n': ['name', 'str', 'Name of waifu to save']
                          },
                          'Save a waifu.',
                          'saveWaifu -n \'waifu name\''
                          ],

            'loadWaifu': [console.load_waifu,
                          {
                              '-n': ['name', 'str', 'Name of waifu to load']
                          },
                          'Load a waifu.',
                          'loadWaifu -n \'waifu name\''
                          ],

            'getWaifuDetail': [console.get_waifu_detail,
                               {
                                   '-n': ['name', 'str', 'Name of waifu']
                               },
                               'Get the detail information of a waifu.',
                               'getWaifuDetail -n \'waifu name\''
                               ],

            'waifuPredict': [console.waifu_predict,
                             {
                                 '-n': ['name', 'str', 'Name of waifu'],
                                 '-s': ['sentence', 'str', 'Sentence input']
                             },
                             'Make prediction using waifu',
                             'waifuPredict -n \'waifu name\' -s \'Hello!\''
                             ],
            # endregion

            # region model commands
            'createModel': [console.create_model,
                            {
                                '-n': ['name', 'str', 'Name of model'],
                                '-t': ['type', 'str', 'Type of model'],
                                '-c': ['model_config', 'str', 'Name of model config to use'],
                                '-d': ['data', 'str', 'Name of data to use']
                            },
                            'Create a Model',
                            "createModel: -n 'model name' -t 'ModelType‘ -c 'model_config name' -d 'data name'"
                            ],

            'deleteModel': [console.delete_model,
                            {
                                '-n': ['name', 'str', 'Name of model to delete']
                            },
                            'Delete a Model',
                            "deleteModel -n 'model name'"
                            ],

            'saveModel': [console.save_model,
                          {
                              '-n': ['name', 'str', 'Name of model to save']
                          },
                          'Save a model',
                          "saveModel -n 'model name'"
                          ],

            'loadModel': [console.load_model,
                          {
                              '-n': ['name', 'str', 'Name of model to load'],
                              '-d': ['data', 'str', 'Name of data to set to model']
                          },
                          'Load a model',
                          "loadModel -n 'model name' -d 'data name'"
                          ],

            'getModelDetails': [console.get_model_details,
                                {
                                    '-n': ['--name', 'str', 'Name of model'],
                                },
                                'Return the details of a model',
                                "getModelDetails -n 'model name'"
                                ],

            'setData': [console.set_data,
                        {
                            '-n': ['name', 'str', 'Name of model'],
                            '-d': ['data', 'str', 'Name of data']
                        },
                        'Set model data',
                        "setData -n 'model name' -d 'data name'"
                        ],

            'train': [console.train,
                      {
                          '-n': ['name', 'str', 'Name of model to train'],
                          '-e': ['epoch', 'int', 'Number of epochs to train for']
                      },
                      'Train a model',
                      "train -n 'model name' -e 10"
                      ],

            'predict': [console.predict,
                        {
                            '-n': ['name', 'str', 'Name of model'],
                            '-i': ['input_data', 'str', 'Name of input data'],
                            '-s': ['save_path', 'str', 'Path to save result (Optional)']
                        },
                        'Make predictions with a model',
                        "predict -n 'model name' -i 'name of input data' -s '\\some\\path.txt'"
                        ],

            # endregion
            'getModelConfigDetails': [console.get_model_config_details,
                                      {
                                          '-n': ['name', 'str', 'Name of Model Config']
                                      },
                                      'Return the details of a model config',
                                      "getModelConfigDetails -n 'model config name'"],

            'createModelConfig': [console.create_model_config,
                                  {
                                      '-n': ['name', 'str', 'Name of model config'],
                                      '-t': ['type', 'str', 'Name of the model type'],
                                      '-c': ['config', 'dict', 'Dictionary of config values (Optional)'],
                                      '-h': ['hyperparameters', 'dict',
                                             'Dictionary of hyperparameters values (Optional)'],
                                      '-s': ['model_structure', 'dict',
                                             'Dictionary of model_structure values (Optional)']
                                  },
                                  'Create a model config with the provided values',
                                  "createModelConfig -n 'model config name' -t 'type' [-c '{'some_key': 'some_value'}'] [-h '{}'] [-ms '{}']"
                                  ],

            'editModelConfig': [console.edit_model_config,
                                {
                                    '-n': ['name', 'str', 'Name of model config to edit'],
                                    '-c': ['config', 'dict',
                                           'Dictionary containing the updated config values (Optional)'],
                                    '-h': ['hyperparameters', 'dict',
                                           'Dictionary containing the updated hyperparameters values (Optional)'],
                                    '-s': ['model_structure', 'dict',
                                           'Dictionary containing the updated model structure values (Optional)'],

                                    '-d': ['device', 'str', 'cibfug.device'],
                                    '-cls': ['class', 'str', 'confiig.class'],
                                    '-e': ['epoch', 'int', 'config.epoch'],
                                    '-cost': ['cost', '', 'config.cost'],
                                    '-ds': ['display_step', '', 'config.display_step'],
                                    '-tb': ['tensorboard', '', 'config.tensorboard'],
                                    '-hd': ['hyperdash', '', 'config.hyperdash'],

                                    '-lr': ['learning_rate', 'float', 'hyperparamters.learning_rate'],
                                    '-bs': ['batch_size', 'int', 'hyperparamters.batch_size'],
                                    '-op': ['optimizer', 'str', 'hyperparamters.optimizer'],

                                    '-ms': ['max_sequence', 'int', 'model_structure.max_sequence'],
                                    '-nh': ['n_hidden', 'int', 'model_structure.n_hidden'],
                                    '-gc': ['gradient_clip', 'float', 'model_structure.gradient_clip'],
                                    '-no': ['node', 'str', 'model_structure.node'],
                                    '-nio': ['n_intent_output', 'int', 'model_structure.n_intent_output'],
                                    '-nno': ['n_ner_output', 'int', 'model.n_ner_output']
                                },
                                'Update a model config with the provided values',
                                'editModelConfig -n \'model config name\' [-c \'{"some_key": "some_value"}\'] [-h \'{}\'] [-s \'{}\']'
                                ],

            'deleteModelConfig': [console.delete_model_config,
                                  {
                                      '-n': ['name', 'str', 'Name of model config to delete']
                                  },
                                  'Delete a model config',
                                  "deleteModelConfig -n 'model config name"
                                  ],

            'saveModelConfig': [console.save_model_config,
                                {
                                    '-n': ['name', 'str', 'Name of model config to save']
                                },
                                'Save a model config',
                                "saveModelConfig -n 'model config name'"
                                ],

            'loadModelConfig': [console.load_model_config,
                                {
                                    '-n': ['name', 'str', 'Name of model config to load']
                                },
                                'Load a model config',
                                "loadModelConfig -n 'model config name'"
                                ],

            'createData': [console.create_data,
                           {
                               '-n': ['name', 'str', 'Name of data'],
                               '-t': ['type', 'str', 'Type of data (based on the model)'],
                               '-c': ['model_config', 'str', 'Name of model config']
                           },
                           'Create a data with empty values',
                           "createData -n 'data name' -t 'ModelType' -c 'model config name"
                           ],

            'dataAddEmbedding': [console.data_add_embedding,
                                 {
                                     '-n': ['name', 'str', 'Name of data'],
                                     '-e': ['embedding', 'str', 'Name of embedding']
                                 },
                                 'Add word embedding to data',
                                 "dataAddEmbedding -n 'data name' -e 'embedding name'"
                                 ],

            'dataReset': [console.data_reset,
                          {
                              '-n': ['name', 'str', 'Name of data to reset']
                          },
                          'Reset a data, clearing all stored data values',
                          "dataReset -n 'data name'"
                          ],

            'deleteData': [console.delete_data,
                           {
                               '-n': ['name', 'str', 'Name of data to delete']
                           },
                           'Delete a data',
                           "deleteData -n 'data name'"
                           ],

            'saveData': [console.save_data,
                         {
                             '-n': ['name', 'str', 'Name of data to save']
                         },
                         'Save a data',
                         "saveData -n 'data name'"
                         ],

            'loadData': [console.load_data,
                         {
                             '-n': ['name', 'str', 'Name of data to load']
                         },
                         'Load a data',
                         "loadData -n 'data name'"
                         ]
            ,

            'getDataDetails': [console.get_data_details,
                               {
                                   '-n': ['--name', 'str', 'Name of data']
                               },
                               'Return the details of a data',
                               "getDataDetails -n 'data name'"],

            'chatbotDataAddTwitter': [console.chatbot_data_add_twitter,
                                      {
                                          '-n': ['name', 'str', 'Name of data to add on'],
                                          '-p': ['path', 'str', 'Path to twitter file']
                                      },
                                      'Add twitter dataset to a chatbot data',
                                      "chatbotDataAddTwitter -n 'data name' -p '\\some\\path\twitter.txt'"
                                      ],

            'chatbotDataAddCornell': [console.chatbot_data_add_cornell,
                                      {
                                          '-n': ['name', 'str', 'Name of data to add on'],
                                          '-mcp': ['movie_conversations_path', 'str',
                                                   'Path to movie_conversations.txt in the Cornell dataset'],
                                          '-mlp': ['movie_lines_path', 'str',
                                                   'Path to movie_lines.txt in the Cornell dataset']
                                      },
                                      'Add Cornell dataset to a chatbot data',
                                      "chatbotDataAddCornell -n 'data name' -mcp '\\some\\cornell\\movie_conversations.txt' -mlp '\\some\\cornell\\movie_lines.txt'"
                                      ],

            'chatbotDataAddParseSentences': [console.chatbot_data_add_parse_sentences,
                                             {
                                                 '-n': ['name', 'str', 'Name of data to add on'],
                                                 '-x': ['x', 'list<str>',
                                                        'List of strings, each representing a sentence input'],
                                                 '-y': ['y', 'list<str>',
                                                        'List of strings, each representing a sentence output']
                                             },
                                             'Add Cornell dataset to a chatbot data',
                                             "chatbotDataAddParseSentences -n 'data name' -x '['some input']' -y '['some output']'"
                                             ],

            'chatbotDataAddParseFile': [console.chatbot_data_add_parse_file,
                                        {
                                            '-n': ['name', 'str', 'Name of data to add on'],
                                            '-x': ['x_path', 'str',
                                                   'Path to a UTF-8 file containing a raw sentence input on each line'],
                                            '-y': ['y_path', 'str',
                                                   'Path to a UTF-8 file containing a raw sentence output on each line']
                                        },
                                        'Parse raw sentences from text files and add them to a chatbot data',
                                        "chatbotDataAddParseFile -n 'data name' -x '\\some\\path\\x.txt' -y '\\some\\path\\y.txt'"
                                        ],

            'chatbotDataAddParseInput': [console.chatbot_data_add_parse_input,
                                         {
                                             '-n': ['name', 'str', 'Name of data to add on'],
                                             '-x': ['x', 'str', 'Raw sentence input']
                                         },
                                         'Parse a raw sentence as input and add it to a chatbot data',
                                         "chatbotDataAddParseInput -n 'data name' -x 'hey how are you'"
                                         ],

            'chatbotDataSetParseInput': [console.chatbot_data_set_parse_input,
                                         {
                                             '-n': ['name', 'str', 'Name of data to set'],
                                             '-x': ['x', 'str', 'Raw sentence input']
                                         },
                                         'Parse a raw sentence as input and set it as a chatbot data',
                                         "chatbotDataSetParseInput -n 'data name' -x 'hey how are you'"
                                         ],

            'intentNERDataAddParseInput': [console.intentNER_data_add_parse_input,
                                           {
                                               '-n': ['name', 'str', 'Name of data to add on'],
                                               '-x': ['x', 'str', 'Raw sentence input']
                                           },
                                           'Parse a raw sentence as input and add it to an intent NER data',
                                           "intentNERDataAddParseInput -n 'data name' -x 'hey how are you'"
                                           ],

            'intentNERDataSetParseInput': [console.intentNER_data_set_parse_input,
                                           {
                                               '-n': ['name', 'str', 'Name of data to set'],
                                               '-x': ['x', 'str', 'Raw sentence input']
                                           },
                                           'Parse a raw sentence as input and set it as an intent NER data',
                                           "intentNERDataSetParseInput -n 'data name' -x 'hey how are you'"
                                           ],

            'intentNERDataAddParseDatafolder': [console.intentNER_data_add_parse_data_folder,
                                                {
                                                    '-n': ['name', 'str', 'Name of data to add on'],
                                                    '-p': ['path', 'str', 'Path to a folder contains input files']
                                                },
                                                'Parse files from a folder and add them to a chatbot data',
                                                "intentNERDataAddParseDatafolder -n 'data name' -p '\\some\\path\\to\\intents'"
                                                ],

            'speakerVerificationDataAddDataPaths': [console.speakerVerification_data_add_data_paths,
                                                    {
                                                        '-n': ['name', 'str', 'Name of data to add on'],
                                                        '-p': ['-path', 'list<str>',
                                                               'List of string paths to raw audio files'],
                                                        '-y': ['y', 'bool',
                                                               'The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)']
                                                    },
                                                    'Parse and add raw audio files to a speaker verification data',
                                                    'speakerVerificationDataAddDataPaths -n \'data name\' -p \'["\\some\\path\\01.wav"]\' [-y True]'
                                                    ],

            'speakerVerificationDataAddDataFile': [console.speakerVerification_data_add_data_file,
                                                   {
                                                       '-n': ['name', 'str', 'Name of data to add on'],
                                                       '-p': ['path', 'str',
                                                              'Path to file containing a path of a raw audio file on each line'],
                                                       '-y': ['y', 'bool',
                                                              'The label (True for is speaker and vice versa) of the audio files. Include for training, leave out for prediction. (Optional)']
                                                   },
                                                   'Read paths to raw audio files and add them to a speaker verification data',
                                                   "speakerVerificationDataAddDataFile -n 'data name' -p '\\some\\path\x07udios.txt' -y True"
                                                   ],

            'getEmbeddingDetails': [console.get_embedding_details,
                                    {
                                        '-n': ['name', 'str', 'Name of Embedding']
                                    },
                                    'Return the details of an embedding',
                                    "getEmbeddingDetails -n 'embedding name'"],

            'createEmbedding': [console.create_embedding,
                                {
                                    '-n': ['name', 'str', 'Name of embedding'],
                                    '-p': ['path', 'str', 'Path to embedding file'],
                                    '-v': ['vocab_size', 'int',
                                           'Maximum number of tokens to read from embedding file (Optional)']
                                },
                                'Create a word embedding',
                                "createEmbedding -n 'embedding name' -p '\\some\\path\\embedding.txt' [-v 100000]"
                                ],

            'deleteEmbedding': [console.delete_embedding,
                                {
                                    '-n': ['name', 'str', 'Name of embedding to delete']
                                },
                                'Delete a word embedding', "deleteEmbedding -n 'embedding name'"
                                ],

            'saveEmbedding': [console.save_embedding,
                              {
                                  '-n': ['name', 'str', 'Name of embedding to save']
                              },
                              'Save an embedding', "saveEmbedding -n 'embedding name'"
                              ],

            'loadEmbedding': [console.load_embedding,
                              {
                                  '-n': ['name', 'str', 'Name of embedding to load']
                              },
                              'Load an embedding', "loadEmbedding -n 'embedding name'"
                              ],

            'startServer': [console.start_server,
                            {
                                '-p': ['port', 'int', 'Port to listen on'], '-l': ['local', 'bool',
                                                                                   'If the server is running locally (server will listen on 127.0.0.1 if this is true or not set) (Optional)'],
                                '-pwd': ['password', 'str', 'Password of server (Optional)'],
                                '-c': ['max_clients', 'int', 'Maximum number of clients (Optional)']
                            },
                            'Start a socket server and listen for clients. The server runs on a separate thread so the console will still function',
                            "startServer -p 23333 [-l True] [-pwd 'p@ssword'] [-c 10]"
                            ],

            'stopServer': [console.stop_server,
                           {
                           },
                           'Stop current socket server and close all connections',
                           'stopServer'
                           ],

            'freezeGraph': [console.freeze_graph,
                            {
                                '-n': ['name', 'str', 'Name of model']
                            },
                            'Freeze Tensorflow graph and latest checkpoint to a file',
                            "freezeGraph -n 'model name'"
                            ],

            'optimize': [console.optimize,
                         {
                             '-n': ['name', 'str', 'Name of waifu to load']
                         },
                         'Optimize a frozen model (see FreezeGraph) for inference.',
                         "optimize -n 'model name'"
                         ],

            'sliceAudio': [console.slice_audio,
                           {
                               '-sp': ['subtitle_path', 'str', 'Path to subtitle file'],
                               '-ap': ['audio_path', 'str', 'Path to audio file'],
                               '-s': ['save_path', 'str', 'Path to save audio']
                           },
                           'Loading subtitle and slicing audio.',
                           "sliceAudio -sp 'subtitle.ass' -ap 'test.mp3' -s 'some\\path\\save\\'"
                           ],

            'save': [console.save,
                     {

                     },
                     'Save the console. (This does not save individual items such as models and waifus.)',
                     'save '
                     ],

            's': [console.save,
                  {

                  },
                  'Save the console. (This does not save individual items such as models and waifus.)',
                  'save '
                  ]
        }
