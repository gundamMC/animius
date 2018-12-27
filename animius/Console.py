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
        embedding.create_embedding(kwargs['name'],kwargs['path'],kwargs['vocab_size'])
        self.embeddings[kwargs['name']]=embedding

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
