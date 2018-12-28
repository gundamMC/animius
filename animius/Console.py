import animius as am


class ArgumentError(Exception):
    pass


class Console:

    def __init__(self):
        self.models = {}
        self.waifu = {}
        self.model_configs = {}
        self.data = {}
        self.embeddings = {}

    @staticmethod
    def check_arguments(args, hard_requirements=None, soft_requirements=None):
        # Check if the user-provided arguments meet the requirements of the method/command
        # hard_requirement throws ArgumentError if not fulfilled
        # soft_requirement gives a value of None
        if hard_requirements is not None:
            for req in hard_requirements:
                if req not in args:
                    raise ArgumentError("{0} is required".format(req))

        if soft_requirements is not None:
            for req in soft_requirements:
                if req not in args:
                    args['req'] = None

    def create_model_config(self, **kwargs):
        """
        Create a model config with the provided values

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model config
        * *cls* (``str``) -- Name of the model class
        * *config* (``dict``) -- Dictionary of config values
        * *hyperparameters* (``dict``) -- Dictionary of hyperparameters values
        * *model_structure* (``model_structure``) -- Dictionary of model_structure values
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'cls'],
                                soft_requirements=['config', 'hyperparameters', 'model_structure'])

        self.model_configs[kwargs['name']] = am.ModelConfig(kwargs['cls'],
                                                            kwargs['config'],
                                                            kwargs['hyperparameters'],
                                                            kwargs['model_structure'])

    def edit_model_config(self, **kwargs):
        """
        Update a model config with the provided values

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model config to edit
        * *config* (``dict``) -- Dictionary containing the updated config values
        * *hyperparameters* (``dict``) -- Dictionary containing the updated hyperparameters values
        * *model_structure* (``model_structure``) -- Dictionary containing the updated model_structure values
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] in self.model_configs:
            def update_dict(target, update_values):
                for key in update_values:
                    target[key] = update_values[key]

            update_dict(self.model_configs[kwargs['name']].config, kwargs['config'])
            update_dict(self.model_configs[kwargs['name']].hyperparameters, kwargs['hyperparameters'])
            update_dict(self.model_configs[kwargs['name']].model_structure, kwargs['model_structure'])

        else:
            raise KeyError("Model config \"{0}\" not found.".format(kwargs['name']))

    def delete_model_config(self, **kwargs):
        """
        Delete a model config

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model config to delete
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] in self.model_configs:
            self.model_configs.pop(kwargs['name'])

        else:
            raise KeyError("Model config \"{0}\" not found.".format(kwargs['name']))

    def create_data(self, **kwargs):
        """
        Create a data with empty values

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data
        * *type* (``str``) -- Type of data (based on the model)
        * *model_config* (``str``) -- Name of model config
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'type', 'model_config'])

        if kwargs['name'] in self.model_configs:
            self.data[kwargs['name']] = am.ChatbotData(kwargs['model_Config'])

        else:
            raise KeyError("Model config \"{0}\" not found.".format(kwargs['name']))

    def data_add_embedding(self, **kwargs):
        """
        Add twitter dataset to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *name_embedding* (``str``) -- Name of the embedding to add to data
        """
        pass

    def data_reset(self, **kwargs):
        """
        Reset a data, clearing all stored data values.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to reset
        """
        pass

    def chatbot_data_add_twitter(self, **kwargs):
        """
        Add twitter dataset to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *path* (``str``) -- Path to twitter file
        """
        pass

    def chatbot_data_add_cornell(self, **kwargs):
        """
        Add Cornell dataset to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *movie_conversations_path* (``str``) -- Path to movie_conversations.txt in the Cornell dataset
        * *movie_lines_path* (``str``) -- Path to movie_lines.txt in the Cornell dataset
        """
        pass

    def chatbot_data_add_parse_sentences(self, **kwargs):
        """
        Parse raw sentences and add them to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *x* (``list<str>``) -- List of strings, each representing a sentence input
        * *y* (``list<str>``) -- List of strings, each representing a sentence output
        """
        pass

    def chatbot_data_add_parse_file(self, **kwargs):
        """
        Parse raw sentences from text files and add them to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *x_path* (``str``) -- Path to a UTF-8 file containing a raw sentence input on each line
        * *y_path* (``str``) -- Path to a UTF-8 file containing a raw sentence output on each line
        """
        pass

    def chatbot_data_add_parse_input(self, **kwargs):
        """
        Parse a raw sentence as input and add it to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *x* (``str``) -- Raw sentence input
        """
        pass

    def chatbot_data_set_parse_input(self, **kwargs):
        """
        Parse a raw sentence as input and set it as a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to set
        * *x* (``str``) -- Raw sentence input
        """
        pass

    def intentNER_data_add_parse_data_folder(self, **kwargs):
        """
        Parse raw sentences from text files and add them to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *x_path* (``list<str>``) -- Path to a UTF-8 file containing a raw sentence input on each line
        * *y_path* (``list<str>``) -- Path to a UTF-8 file containing a raw sentence output on each line
        """
        pass

    def intentNER_data_add_parse_input_file(self, **kwargs):
        """
        Parse raw sentences from text files and add them to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *x_path* (``list<str>``) -- Path to a UTF-8 file containing a raw sentence input on each line
        * *y_path* (``list<str>``) -- Path to a UTF-8 file containing a raw sentence output on each line
        """
        pass

    def intentNER_data_add_parse_input(self, **kwargs):
        """
        Parse a raw sentence as input and add it to an intent NER data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *x* (``str``) -- Raw sentence input
        """
        pass

    def intentNER_data_set_parse_input(self, **kwargs):
        """
        Parse a raw sentence as input and set it as an intent NER data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to set
        * *x* (``str``) -- Raw sentence input
        """
        pass

    def speakerVerification_data_add_data_paths(self, **kwargs):
        """
        Parse and add raw audio files to a speaker verification data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *paths* (``list<str>``) -- List of string paths to raw audio files
        * *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction.
        """
        pass

    def speakerVerification_data_add_data_file(self, **kwargs):
        """
        Read paths to raw audio files and add them to a speaker verification data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *path* (``str``) -- Path to file containing a path of a raw audio file on each line
        * *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction.
        """
        pass

    def delete_data(self, **kwargs):
        """
        Delete a data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to delete
        """
        pass

    def create_embedding(self, **kwargs):
        """
        Create a word embedding

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of embedding
        * *path* (``str``) -- Path to embedding file
        * *vocab_size* (``int``) -- Maximum number of tokens to read from embedding file
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'],
                                soft_requirements=['vocab_size'])

        embedding = am.WordEmbedding()
        embedding.create_embedding(kwargs['path'],kwargs['vocab_size'])
        self.embeddings[kwargs['name']] = embedding

    def delete_embedding(self, **kwargs):
        """
        Delete a word embedding

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of embedding to delete
        """
        pass

    def handle_network(self, request):

        command = request.command.lower().replace(' ', '_')
        method_to_call = getattr(self, command)

        try:
            result = method_to_call(request.arguments)
            if result is None:
                result = {}
            return request.id, 0, 'success', result
        except ArgumentError as exc:
            return request.id, 1, exc, {}
        except Exception as exc:
            return request.id, 2, exc, {}
