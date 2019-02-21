import animius as am
import os
import json
from ast import literal_eval
from shlex import split as arg_split
from concurrent.futures import ThreadPoolExecutor
from functools import partial


class ArgumentError(Exception):
    pass


class NameAlreadyExistError(Exception):
    pass


class NameNotFoundError(Exception):
    pass


class NotLoadedError(Exception):
    pass


class _ConsoleItem:
    def __init__(self, item=None, save_directory=None, save_name=None):
        self.item = item

        self.loaded = item is not None

        self.saved_directory = save_directory
        self.saved_name = save_name

    def save(self):
        self.item.save(self.saved_directory, self.saved_name)


class CancellationToken:
    def __init__(self):
        self.is_cancalled = False

    def cancel(self):
        self.is_cancalled = True


class Console:

    def __init__(self, init_directory=None):

        self.commands = None

        animius_dir = os.path.dirname(os.path.realpath(__file__))
        self.config_dir = os.path.join(animius_dir, 'user-config.json')

        sub_dirs = {'waifu', 'models', 'model_configs', 'data', 'embeddings'}

        self.models = {}
        self.waifu = {}
        self.model_configs = {}
        self.data = {}
        self.embeddings = {}

        # used for SocketServer interactions
        self.socket_server = None

        # No config / first time initializing
        if not os.path.exists(self.config_dir):

            if init_directory is None:
                print("Please enter the data directory to save data in:")
                print("Default ({0})".format(os.path.join(animius_dir, 'resources')))
                init_directory = input()

            if not init_directory.strip():
                init_directory = os.path.join(animius_dir, 'resources')
                if not os.path.exists(init_directory):
                    os.mkdir(init_directory)

            self.directories = {}

            # create sub directories
            for sub_dir in sub_dirs:
                sub_dir_path = os.path.join(init_directory, sub_dir)
                if not os.path.exists(sub_dir_path):
                    os.mkdir(sub_dir_path)
                self.directories[sub_dir] = sub_dir_path

        else:  # load config

            with open(self.config_dir, 'r') as f:
                stored = json.load(f)

            self.directories = stored['directories']

            # read all saved items
            for sub_dir in sub_dirs:
                with open(os.path.join(self.directories[sub_dir], sub_dir + '.json'), 'r') as f:
                    stored = json.load(f)
                    for item in stored['items']:
                        console_item = _ConsoleItem()
                        console_item.saved_directory = stored['items'][item]['saved_directory']
                        console_item.saved_name = stored['items'][item]['saved_name']
                        # get the self. dictionary from sub_dir name
                        getattr(self, sub_dir)[item] = console_item

        self.training_pool = ThreadPoolExecutor(max_workers=3)  # thread pool for training threads
        self.training_models = dict()

    @staticmethod
    def ParseArgs(user_input):
        user_input = arg_split(user_input)
        command = user_input[0]
        if '--help' in user_input or '-h' in user_input:
            return command, {'--help': ''}
        else:
            values = []

            for value in user_input[2::2]:
                try:
                    values.append(literal_eval(value))
                except (ValueError, SyntaxError):
                    # Errors would occur b/c strings are not quoted from arg_split
                    values.append(value)

            args = dict(zip(user_input[1::2], values))
        # leave key parsing to Main

        return command, args

    def save(self, **kwargs):
        """
        Save console

        :param kwargs: Useless.
        """
        with open(self.config_dir, 'w') as f:
            json.dump({'directories': self.directories}, f, indent=4)

        # save all items
        for sub_dir in {'waifu', 'models', 'model_configs', 'data', 'embeddings'}:
            tmp_dict = {}
            for item_name, console_item in getattr(self, sub_dir).items():
                tmp_dict[item_name] = {'saved_directory': console_item.saved_directory,
                                       'saved_name': console_item.saved_name}
            with open(os.path.join(self.directories[sub_dir], sub_dir + '.json'), 'w') as f:
                json.dump({'items': tmp_dict}, f, indent=4)

    def get_waifu(self, **kwargs):
        results = {}
        for key in self.waifu:
            if self.waifu[key].loaded:
                results[key] = {'name': self.waifu[key].item.config['name'],
                                'desciprtion': self.waifu[key].item.config['description']}
            else:
                results[key] = {}
        return results

    def get_models(self, **kwargs):
        return list(self.models.keys())

    def get_model_configs(self, **kwargs):
        return list(self.model_configs.keys())

    def get_data(self, **kwargs):
        return list(self.data.keys())

    def get_embeddings(self, **kwargs):
        return list(self.embeddings.keys())

    def get_waifu_detail(self, **kwargs):
        """
        Return the details of a waifu

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.waifu:
            raise NameNotFoundError("Waifu {0} not found".format(kwargs['name']))

        tmp = dict(self.waifu[kwargs['name']].item.config)

        tmp['saved_directory'] = self.waifu[kwargs['name']].saved_directory
        tmp['saved_name'] = self.waifu[kwargs['name']].saved_name

        return tmp

    def get_model_details(self, **kwargs):
        """
        Return the details of a model

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model {0} not found".format(kwargs['name']))

        tmp = {'config': self.models[kwargs['name']].item.config,
               'model_structure': self.models[kwargs['name']].item.model_structure,
               'hyperparamter': self.models[kwargs['name']].hyperparameters.config,
               'saved_directory': self.models[kwargs['name']].saved_directory,
               'saved_name': self.models[kwargs['name']].saved_name}

        return tmp

    def get_data_details(self, **kwargs):
        """
        Return the details of a data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.data:
            raise NameNotFoundError("Data {0} not found".format(kwargs['name']))

        # get model config saved dir, saved name, and name in console (if there is one)
        model_config = self.data[kwargs['name']].item.model_config

        tmp = {'model_config_saved_directory': model_config.saved_directory,
               'model_config_saved_name': model_config.saved_name}

        for name, mc_item in self.model_configs.items():
            if mc_item.saved_directory == model_config.saved_directory:
                tmp['model_config_name'] = name

        if 'model_config_name' not in tmp:
            tmp['model_config_name'] = 'no matching model config found in console'

        # Embedding - same as model config
        embedding = self.data[kwargs['name']].item.values['embedding']

        tmp['embedding_saved_directory'] = embedding.saved_directory
        tmp['embedding_saved_name'] = embedding.saved_name

        for name, emb in self.embeddings.items():
            if emb.saved_directory == emb.saved_directory:
                tmp['embedding_name'] = name

        if 'embedding_name' not in tmp:
            tmp['embedding_name'] = 'no matching embedding found in console'

        tmp['cls'] = type(self.data[kwargs['name']].item).__name__

        tmp['values'] = list(self.data[kwargs['name']].item.values.keys())

        return tmp

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
                    args[req] = None

    def create_waifu(self, **kwargs):
        """
        Use existing model to create a waifu

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu
        * *combined_chatbot_model* (``str``) -- Name or directory of combined chatbot model to use
        * *embedding* (``str``) -- Name of embedding
        * *description* (``str``) -- Description of waifu. Optional
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'combined_chatbot_model', 'embedding'])

        if kwargs['name'] in self.waifu:
            raise NameAlreadyExistError("The name {0} is already used by another waifu".format(kwargs['name']))

        if kwargs['embedding'] not in self.embeddings:
            raise NameNotFoundError("Embedding {0} no found".format(kwargs['embedding']))

        if not os.path.isdir(kwargs['combined_chatbot_model']):
            if kwargs['combined_chatbot_model'] in self.models:
                kwargs['combined_chatbot_model'] = self.models[kwargs['combined_chatbot_model']].saved_directory
            else:
                raise NameNotFoundError("Model {0} not found".format(kwargs['model']))

        desc = '' if 'description' not in kwargs else kwargs['description']

        waifu = am.Waifu(kwargs['name'], {'CombinedPrediction': kwargs['combined_chatbot_model']}, description=desc)
        waifu.load_combined_prediction_model()
        waifu.build_input(self.embeddings[kwargs['embedding']].item)

        console_item = _ConsoleItem(waifu, os.path.join(self.directories['waifu'], kwargs['name']), kwargs['name'])
        # saving it first to set up its saving location
        console_item.save()

        self.waifu[kwargs['name']] = console_item

    def delete_waifu(self, **kwargs):
        """
        Delete a waifu

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu to delete
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] in self.waifu:
            self.waifu.pop(kwargs['name'])
        else:
            raise NameNotFoundError("Waifu \"{0}\" not found.".format(kwargs['name']))

    def save_waifu(self, **kwargs):
        """
        Save a waifu

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu to save
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.waifu:
            raise NameNotFoundError("Waifu \"{0}\" not found".format(kwargs['name']))

        self.waifu[kwargs['name']].save()

    def load_waifu(self, **kwargs):
        """
        Load a waifu

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu to load
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.waifu:
            raise NameNotFoundError("Waifu \"{0}\" not found".format(kwargs['name']))

        waifu = am.Waifu.load(
            self.waifu[kwargs['name']].saved_directory,
            self.waifu[kwargs['name']].saved_name)

        self.waifu[kwargs['name']].item = waifu
        self.waifu[kwargs['name']].loaded = True

    def waifu_predict(self, **kwargs):
        """
        Predict waifu output

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu to use
        * *sentence* (``str``) -- Sentence input
        """

        Console.check_arguments(kwargs, hard_requirements=['name', 'sentence'])

        if kwargs['name'] not in self.waifu:
            raise NameNotFoundError("Waifu \"{0}\" not found".format(kwargs['waifu']))

        return self.waifu[kwargs['name']].item.predict(kwargs['sentence'])

    def create_model(self, **kwargs):
        """
        Create a model and built its graph

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model
        * *type* (``str``) -- Type of model
        * *model_config* (``str``) -- Name of model config to use
        * *data* (``str``) -- Name of data
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'type', 'model_config', 'data'])

        if kwargs['name'] in self.models:
            raise NameAlreadyExistError("The name {0} is already used by another model".format(kwargs['name']))

        if kwargs['type'] == 'Chatbot':
            model = am.Chatbot.ChatbotModel()
        elif kwargs['type'] == 'IntentNER':
            model = am.IntentNER.IntentNERModel()
        elif kwargs['type'] == 'CombinedChatbot':
            if kwargs['model_config'] not in self.model_configs:
                raise NameNotFoundError("Model Config {0} not found".format(kwargs['model_config']))
            if kwargs['data'] not in self.data:
                raise NameNotFoundError("Data {0} not found".format(kwargs['data']))
            model = am.Chatbot.CombinedChatbotModel(self.model_configs[kwargs['model_config']].item,
                                                    self.data[kwargs['data']].item)
        elif kwargs['type'] == 'SpeakVerification':
            model = am.SpeakerVerification.SpeakerVerificationModel()
        else:
            raise KeyError("Model type \"{0}\" not found.".format(kwargs['type']))

        # CombinedChatbotModel builds graph in its constructor
        if kwargs['type'] != 'CombinedChatbotModel':
            model.build_graph(self.model_configs[kwargs['model_config']].item, self.data[kwargs['data']].item)

        console_item = _ConsoleItem(model, os.path.join(self.directories['model'], kwargs['name']), kwargs['name'])
        # saving it first to set up its saving location
        console_item.save()

        self.models[kwargs['name']] = console_item

    def delete_model(self, **kwargs):
        """
        Delete a model

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model to delete
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] in self.models:
            self.models.pop(kwargs['name'])
        else:
            raise NameNotFoundError("Model \"{0}\" not found.".format(kwargs['name']))

    def save_model(self, **kwargs):
        """
        Save a model

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model to save
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model \"{0}\" not found".format(kwargs['name']))

        self.models[kwargs['name']].save()

    def load_model(self, **kwargs):
        """
        Load a model

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model to load
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model \"{0}\" not found".format(kwargs['name']))

        model = am.Model.load(
            self.models[kwargs['name']].saved_directory,
            self.models[kwargs['name']].saved_name)

        self.models[kwargs['name']].item = model
        self.models[kwargs['name']].loaded = True

    def set_data(self, **kwargs):
        """
        Set model data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model to set
        * *data* (``str``) -- Name of data to set
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'data'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model \"{0}\" not found".format(kwargs['name']))
        if kwargs['data'] not in self.data:
            raise NameNotFoundError("Data \"{0}\" not found".format(kwargs['data']))

        self.models[kwargs['name']].item.set_data(self.data[kwargs['data']].item)

    @staticmethod
    def train_complete_callback(print_string, model_name, training_dict):
        """
        Called when a model finishes training. See train() for details
        """
        print(print_string)
        training_dict.pop(model_name)

    def train(self, **kwargs):
        """
        Train a model

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model to set
        * *epoch* (``int``) -- Number of epoch
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'],
                                soft_requirements=['epoch'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model \"{0}\" not found".format(kwargs['name']))

        cancelToken = CancellationToken()

        self.training_models[kwargs['name']] = cancelToken

        future = self.training_pool.submit(self.models[kwargs['name']].item.train,
                                           epochs=kwargs['epoch'],
                                           CancellationToken=cancelToken)

        callback_string = 'Model {0} has finished training for {1} epochs!'.format(kwargs['name'], kwargs['epoch'])

        future.add_done_callback(partial(Console.train_complete_callback,
                                         callback_string, kwargs['name'], self.training_models))

        print('Started training model {0}'.format(kwargs['name']))

    def stop_training(self, **kwargs):
        """
        Cancel training a model. (The model will stop once it finishes the current epoch)

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model to stop
        """

        Console.check_arguments(kwargs, hard_requirements=['name'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model \"{0}\" not found".format(kwargs['name']))

        if kwargs['name'] not in self.training_models:
            raise NameNotFoundError("Model \"{0}\" is currently not training".format(kwargs['name']))

        self.training_models[kwargs['name']].cancel()

    def predict(self, **kwargs):
        """
        Predict model

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model to predict
        * *input_data* (``str``) -- Name of input data
        * *save_path* (``str``) -- Path to save result
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'input_data'],
                                soft_requirements=['save_path'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model \"{0}\" not found".format(kwargs['name']))
        if kwargs['input_data'] not in self.data:
            raise NameNotFoundError("Data \"{0}\" not found".format(kwargs['input_data']))

        return self.models[kwargs['name']].item.predict(kwargs['input_data'], save_path=kwargs['save_path'])

    def freeze_graph(self, **kwargs):
        """
        Freeze model and save a frozen graph to file

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model to freeze
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model \"{0}\" not found".format(kwargs['name']))

        with open(os.path.join(self.models[kwargs['name']].saved_directory,
                               self.models[kwargs['name']].saved_name + '.json'
                               ), 'r') as f:
            stored = json.load(f)
            class_name = stored['config']['class']

        if class_name == 'Chatbot':
            output_node_names = 'decode_1/output_infer'
        elif class_name == 'CombinedChatbot':
            output_node_names = 'decode_1/output_infer, intent/output_intent, intent/output_ner'
        elif class_name == 'IntentNER':
            output_node_names = 'output_intent, output_ner'
        elif class_name == 'SpeakerVerification':
            output_node_names = 'output_predict'
        else:
            raise ValueError("Class name not found")

        am.Utils.freeze_graph(self.models[kwargs['name']].saved_directory, output_node_names, stored)

    def optimize(self, **kwargs):
        """
        Optimize a frozen model for inference

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model to optimize
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model \"{0}\" not found".format(kwargs['name']))

        with open(os.path.join(self.models[kwargs['name']].saved_directory,
                               self.models[kwargs['name']].saved_name + '.json'
                               ), 'r') as f:
            stored = json.load(f)
            class_name = stored['config']['class']

        if class_name == 'Chatbot':
            input_node_names = ['input_x', 'input_x_length']
            output_node_names = ['decode_1/output_infer']
        elif class_name == 'CombinedChatbot':
            input_node_names = ['input_x', 'input_x_length']
            output_node_names = ['decode_1/output_infer', 'intent/output_intent', 'intent/output_ner']
        elif class_name == 'IntentNER':
            input_node_names = ['input_x', 'input_x_length']
            output_node_names = ['output_intent', 'output_entities']
        elif class_name == 'SpeakerVerification':
            input_node_names = ['input_x']
            output_node_names = ['output_predict']
        else:
            raise ValueError("Class name not found")

        am.Utils.optimize(self.models[kwargs['name']].saved_directory, input_node_names, output_node_names)

    def create_model_config(self, **kwargs):
        """
        Create a model config with the provided values

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model config
        * *type* (``str``) -- Name of the model type
        * *config* (``dict``) -- Dictionary of config values
        * *hyperparameters* (``dict``) -- Dictionary of hyperparameters values
        * *model_structure* (``model_structure``) -- Dictionary of model_structure values
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'type'],
                                soft_requirements=['config', 'hyperparameters', 'model_structure'])

        if kwargs['name'] in self.model_configs:
            raise NameAlreadyExistError("The name {0} is already used by another model config".format(kwargs['name']))

        if kwargs['type'] not in ['SpeakVerification', 'Chatbot', 'IntentNER', 'CombinedChatbot']:
            raise KeyError("Model type \"{0}\" not found.".format(kwargs['type']))

        model_config = am.ModelConfig(cls=kwargs['type'],
                                      config=kwargs['config'],
                                      hyperparameters=kwargs['hyperparameters'],
                                      model_structure=kwargs['model_structure'])

        console_item = _ConsoleItem(model_config, self.directories['model_configs'], kwargs['name'])
        # saving it first to set up its saving location
        console_item.save()

        self.model_configs[kwargs['name']] = console_item

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
                                hard_requirements=['name'],
                                soft_requirements=['config, hyperparameters, model_structure'])

        if kwargs['name'] in self.model_configs:
            def update_dict(target, update_values):
                for key in update_values:
                    target[key] = update_values[key]

            update_dict(self.model_configs[kwargs['name']].item.config, kwargs['config'])
            update_dict(self.model_configs[kwargs['name']].item.hyperparameters, kwargs['hyperparameters'])
            update_dict(self.model_configs[kwargs['name']].item.model_structure, kwargs['model_structure'])

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

    def save_model_config(self, **kwargs):
        """
        Save a model config

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model config to save
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.model_configs:
            raise NameNotFoundError("Model config \"{0}\" not found".format(kwargs['name']))

        self.model_configs[kwargs['name']].save()

    def load_model_config(self, **kwargs):
        """
        load a model config

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model config to load
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.model_configs:
            raise NameNotFoundError("Model config \"{0}\" not found".format(kwargs['name']))

        model_config = am.ModelConfig.load(
            self.model_configs[kwargs['name']].saved_directory,
            self.model_configs[kwargs['name']].saved_name)

        self.model_configs[kwargs['name']].item = model_config
        self.model_configs[kwargs['name']].loaded = True

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

        if kwargs['name'] in self.data:
            raise NameAlreadyExistError("The name {0} is already used by another data".format(kwargs['name']))

        if kwargs['model_config'] in self.model_configs:
            if kwargs['type'] == 'Chatbot' or kwargs['type'] == 'CombinedChatbot':
                data = am.ChatbotData(self.model_configs[kwargs['model_config']].item)
            elif kwargs['type'] == 'IntentNERD':
                data = am.IntentNERData(self.model_configs[kwargs['model_config']].item)
            elif kwargs['type'] == 'SpeakerVerification':
                data = am.SpeakerVerificationData(self.model_configs[kwargs['model_config']].item)
            else:
                raise KeyError("Data type \"{0}\" not found.".format(kwargs['type']))
        else:
            raise KeyError("Model config \"{0}\" not found.".format(kwargs['model_config']))

        # saving it first to set up its saving location
        console_item = _ConsoleItem(data, os.path.join(self.directories['data'], kwargs['name']), kwargs['name'])
        console_item.save()

        self.data[kwargs['name']] = console_item

    def save_data(self, **kwargs):
        """
        Save a data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to save
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.data:
            raise NameNotFoundError("Data \"{0}\" not found".format(kwargs['name']))

        self.data[kwargs['name']].save()

    def load_data(self, **kwargs):
        """
        Load a data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to load
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.data:
            raise NameNotFoundError("Data \"{0}\" not found".format(kwargs['name']))

        data = am.Data.load(
            self.data[kwargs['name']].saved_directory,
            self.data[kwargs['name']].saved_name)

        self.data[kwargs['name']].item = data
        self.data[kwargs['name']].loaded = True

    def data_add_embedding(self, **kwargs):
        """
        Add twitter dataset to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *name_embedding* (``str``) -- Name of the embedding to add to data
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'name_embedding'])

        if kwargs['name'] in self.data:
            if kwargs['name_embedding'] in self.embeddings:
                self.data[kwargs['name']].item.add_embedding_class(self.embeddings[kwargs['name_embedding']].item)
            else:
                raise KeyError("Embedding \"{0}\" not found.".format(kwargs['name_embedding']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def data_reset(self, **kwargs):
        """
        Reset a data, clearing all stored data values.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to reset
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] in self.data:
            self.data[kwargs['name']].item.reset()
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def chatbot_data_add_twitter(self, **kwargs):
        """
        Add twitter dataset to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *path* (``str``) -- Path to twitter file
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.ChatbotData):
                self.data[kwargs['name']].item.add_twitter(kwargs['path'])
            else:
                raise KeyError("Data \"{0}\" is not a ChatbotData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def chatbot_data_add_cornell(self, **kwargs):
        """
        Add Cornell dataset to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *movie_conversations_path* (``str``) -- Path to movie_conversations.txt in the Cornell dataset
        * *movie_lines_path* (``str``) -- Path to movie_lines.txt in the Cornell dataset
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'movie_conversations_path', 'movie_lines_path'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.ChatbotData):
                self.data[kwargs['name']].item.add_cornell(kwargs['movie_conversations_path'],
                                                           kwargs['movie_lines_path'])
            else:
                raise KeyError("Data \"{0}\" is not a ChatbotData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def chatbot_data_add_parse_sentences(self, **kwargs):
        """
        Parse raw sentences and add them to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *x* (``list<str>``) -- List of strings, each representing a sentence input
        * *y* (``list<str>``) -- List of strings, each representing a sentence output
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'x', 'y'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.ChatbotData):
                self.data[kwargs['name']].item.add_parse_sentences(kwargs['x'], kwargs['y'])
            else:
                raise KeyError("Data \"{0}\" is not a ChatbotData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def chatbot_data_add_parse_file(self, **kwargs):
        """
        Parse raw sentences from text files and add them to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *x_path* (``str``) -- Path to a UTF-8 file containing a raw sentence input on each line
        * *y_path* (``str``) -- Path to a UTF-8 file containing a raw sentence output on each line
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'x_path', 'y_path'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.ChatbotData):
                self.data[kwargs['name']].item.add_parse_file(kwargs['x_path'], kwargs['y_path'])
            else:
                raise KeyError("Data \"{0}\" is not a ChatbotData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def chatbot_data_add_parse_input(self, **kwargs):
        """
        Parse a raw sentence as input and add it to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *x* (``str``) -- Raw sentence input
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'x'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.ChatbotData):
                self.data[kwargs['name']].item.add_parse_input(kwargs['x'])
            else:
                raise KeyError("Data \"{0}\" is not a ChatbotData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def chatbot_data_set_parse_input(self, **kwargs):
        """
        Parse a raw sentence as input and set it as a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to set
        * *x* (``str``) -- Raw sentence input
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'x'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.ChatbotData):
                self.data[kwargs['name']].item.set_parse_input(kwargs['x'])
            else:
                raise KeyError("Data \"{0}\" is not a ChatbotData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def intentNER_data_add_parse_data_folder(self, **kwargs):
        """
        Parse files from a folder and add them to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *folder_directory* (``str``) -- Path to a folder contains input files
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'folder_directory'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.IntentNERData):
                self.data[kwargs['name']].item.add_parse_data_folder(kwargs['folder_directory'])
            else:
                raise KeyError("Data \"{0}\" is not a IntentNERData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def intentNER_data_add_parse_input(self, **kwargs):
        """
        Parse a raw sentence as input and add it to an intent NER data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *x* (``str``) -- Raw sentence input
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'x'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.IntentNERData):
                self.data[kwargs['name']].item.add_parse_input(kwargs['x'])
            else:
                raise KeyError("Data \"{0}\" is not a IntentNERData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def intentNER_data_set_parse_input(self, **kwargs):
        """
        Parse a raw sentence as input and set it as an intent NER data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to set
        * *x* (``str``) -- Raw sentence input
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'x'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.IntentNERData):
                self.data[kwargs['name']].item.set_parse_input(kwargs['x'])
            else:
                raise KeyError("Data \"{0}\" is not a IntentNERData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def speakerVerification_data_add_data_paths(self, **kwargs):
        """
        Parse and add raw audio files to a speaker verification data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *paths* (``list<str>``) -- List of string paths to raw audio files
        * *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction.
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'paths'],
                                soft_requirements=['y'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.SpeakerVerificationData):
                self.data[kwargs['name']].item.add_parse_data_paths(kwargs['paths'], kwargs['y'])
            else:
                raise KeyError("Data \"{0}\" is not a SpeakerVerificationData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def speakerVerification_data_add_data_file(self, **kwargs):
        """
        Read paths to raw audio files and add them to a speaker verification data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *path* (``str``) -- Path to file containing a path of a raw audio file on each line
        * *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction.
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'paths'],
                                soft_requirements=['y'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.SpeakerVerificationData):
                self.data[kwargs['name']].item.add_parse_data_file(kwargs['paths'], kwargs['y'])
            else:
                raise KeyError("Data \"{0}\" is not a SpeakerVerificationData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def delete_data(self, **kwargs):
        """
        Delete a data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to delete
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] in self.data:
            self.data.pop(kwargs['name'])

        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

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
        if kwargs['vocab_size'] is not None:
            embedding.create_embedding(kwargs['path'], kwargs['vocab_size'])
        else:
            embedding.create_embedding(kwargs['path'])

        # saving it first to set up its saving location
        console_item = _ConsoleItem(embedding,
                                    os.path.join(self.directories['embeddings'], kwargs['name']),
                                    kwargs['name'])
        console_item.save()

        self.embeddings[kwargs['name']] = console_item

    def save_embedding(self, **kwargs):
        """
        Save an embedding

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of embedding to save
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.embeddings:
            raise NameNotFoundError("Embedding \"{0}\" not found".format(kwargs['name']))

        self.embeddings[kwargs['name']].save()

    def load_embedding(self, **kwargs):
        """
        load an embedding

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of embedding to load
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.embeddings:
            raise NameNotFoundError("Embedding \"{0}\" not found".format(kwargs['name']))

        embedding = am.WordEmbedding.load(
            self.embeddings[kwargs['name']].saved_directory,
            self.embeddings[kwargs['name']].saved_name)

        self.embeddings[kwargs['name']].item = embedding
        self.embeddings[kwargs['name']].loaded = True

    def delete_embedding(self, **kwargs):
        """
        Delete a word embedding

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of embedding to delete
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] in self.embeddings:
            self.embeddings.pop(kwargs['name'])

        else:
            raise KeyError("Embedding \"{0}\" not found.".format(kwargs['name']))

    def start_server(self, **kwargs):
        """
        Start server

        :param kwargs:

        :Keyword Arguments:
        * *port* (``int``) -- Port of server
        * *local* (``bool``) -- Decide if the server is running locally
        * *pwd* (``str``) -- Password of server
        * *max_clients* (``int``) -- Maximum number of clients
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['port'],
                                soft_requirements=['local', 'pwd', 'max_clients'])

        if self.socket_server is not None:
            raise ValueError("A server is already running on this console.")

        if kwargs['max_clients'] is None:
            kwargs['max_clients'] = 10

        if kwargs['pwd'] is None:
            kwargs['pwd'] = ''

        if kwargs['local'] is None:
            kwargs['local'] = True

        self.socket_server = \
            am.start_server(self, kwargs['port'], kwargs['local'], kwargs['pwd'], kwargs['max_clients'])

    def stop_server(self, **kwargs):
        """
        Stop server

        :param kwargs: Useless.
        """

        if self.socket_server is None:
            raise ValueError("No server is currently running.")

        self.socket_server.stop()

        self.socket_server = None

    def init_commands(self):
        self.commands = am.Commands(self)

    def handle_network(self, request):

        # initialize commands first
        if self.commands is None:
            self.init_commands()

        # command must be pre-defined
        if request.command not in self.commands:
            return

        method_to_call = self.commands[request.command][0]

        try:
            result = method_to_call.__call__(**request.arguments)
            if result is None:
                result = {}
            return request.id, 0, 'success', result
        except ArgumentError as exc:
            return request.id, 1, exc, {}
        except Exception as exc:  # all other errors
            return request.id, 2, exc, {}

    def handle_command(self, user_input):

        # initialize commands first
        if self.commands is None:
            self.init_commands()

        if user_input.lower() == 'help' or user_input == '?':
            pass
        elif user_input.lower() == 'about' or user_input.lower() == 'version':
            pass
        elif user_input is None or not user_input.strip():  # empty string gives false
            return
        else:
            command, args = am.Console.ParseArgs(user_input)

            print('command:', command)
            print('args', args)

            if command is None:
                return
            elif command in self.commands:
                if '--help' in args:
                    print(self.commands[command][2])
                    print('usage: ' + self.commands[command][3])
                    print('  arguments:')
                    for short in self.commands[command][1]:
                        print('    {0}, --{1} ({2}) \t {3}'.format(
                            short,
                            self.commands[command][1][short][0],
                            self.commands[command][1][short][1],
                            self.commands[command][1][short][2]
                        ).expandtabs(30)
                              )
                else:
                    # valid command and valid args

                    # change arguments into kwargs for passing into console
                    kwargs = {}
                    for arg in args:
                        if arg[:2] == '--':  # long
                            kwargs[arg[2:]] = args[arg]
                        elif arg[:1] == '-':  # short
                            if arg not in self.commands[command][1]:
                                print("Invalid short argument {0}, skipping it".format(arg))
                                continue

                            long_arg = self.commands[command][1][arg][0]
                            kwargs[long_arg] = args[arg]

                    print(kwargs)
                    print(self.commands[command][0])
                    print('==================================================')

                    try:
                        result = self.commands[command][0].__call__(**kwargs)
                        if result is not None:
                            print(result)
                    except Exception as exc:
                        print('{0}: {1}'.format(type(exc).__name__, exc))
                        raise exc
            else:
                print('Invalid command')
