import base64
import json
import os
import queue
import threading
import zipfile
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from shlex import split as arg_split

import animius as am


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
        self.queue = None
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

    def export_waifu(self, **kwargs):
        """
        Export a waifu

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu
        * *path*  (``str``) -- Path to export file
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if kwargs['name'] not in self.waifu:
            raise NameNotFoundError("Waifu {0} not found".format(kwargs['name']))
        elif self.waifu[kwargs['name']] is None:
            raise NotLoadedError("Waifu {0} not loaded".format(kwargs['name']))
        elif not os.path.exists(kwargs['path']):
            os.makedirs(kwargs['path'], exist_ok=True)

        waifu_directory = self.waifu[kwargs['name']].saved_directory
        waifu_name = self.waifu[kwargs['name']].saved_name

        model_name = self.waifu[kwargs['name']].item.config['models']['CombinedChatbotName']
        model_directory = self.waifu[kwargs['name']].item.config['models']['CombinedChatbotDirectory']

        zip_path = os.path.join(kwargs['path'], waifu_name + '.zip')
        zf = zipfile.ZipFile(zip_path, mode='w')

        if os.path.exists(waifu_directory):
            files = os.listdir(waifu_directory)
            for file in files:
                path = os.path.join(waifu_directory, file)
                zf.write(path, '\\waifu\\' + file, compress_type=zipfile.ZIP_DEFLATED)
        else:
            raise FileNotFoundError()

        if os.path.exists(model_directory) and model_name in self.models:
            self.models[kwargs['name']].saved_directory = os.path.join(model_directory, "temp")
            self.models[kwargs['name']].save()

            files = os.listdir(model_directory)
            for file in files:
                path = os.path.join(model_directory, "temp", file)
                zf.write(path, file, compress_type=zipfile.ZIP_DEFLATED)
                os.remove(path)

            os.rmdir(os.path.join(model_directory + "temp"))
            self.models[kwargs['name']].saved_directory = model_directory

    def export_model(self, **kwargs):
        """
        Export a model

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model
        * *path*  (``str``) -- Path to export file
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model {0} not found".format(kwargs['name']))
        elif self.models[kwargs['name']] is None:
            raise NotLoadedError("Model {0} not loaded".format(kwargs['name']))
        elif not os.path.exists(kwargs['path']):
            os.makedirs(kwargs['path'], exist_ok=True)

        model_name = self.models[kwargs['name']].saved_name
        model_directory = self.models[kwargs['name']].saved_directory
        self.models[kwargs['name']].saved_directory = os.path.join(model_directory, "temp")
        self.models[kwargs['name']].save()

        if os.path.exists(model_directory):
            zip_path = os.path.join(kwargs['path'], model_name + '.zip')
            zf = zipfile.ZipFile(zip_path, mode='w')
            files = os.listdir(os.path.join(model_directory, "temp"))
            for file in files:
                path = os.path.join(model_directory, "temp", file)
                zf.write(path, file, compress_type=zipfile.ZIP_DEFLATED)
                os.remove(path)
        else:
            raise FileNotFoundError()

        os.rmdir(os.path.join(model_directory, "temp"))
        self.models[kwargs['name']].saved_directory = model_directory

    def export_model_config(self, **kwargs):
        """
        Export a model config

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model config
        * *path*  (``str``) -- Path to export file
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if kwargs['name'] not in self.model_configs:
            raise NameNotFoundError("ModelConfig {0} not found".format(kwargs['name']))
        elif self.model_configs[kwargs['name']] is None:
            raise NotLoadedError("Model Config {0} not loaded".format(kwargs['name']))
        elif not os.path.exists(kwargs['path']):
            os.makedirs(kwargs['path'], exist_ok=True)

        model_config_directory = self.model_configs[kwargs['name']].saved_directory
        model_config_name = self.model_configs[kwargs['name']].saved_name
        file_path = os.path.join(model_config_directory, model_config_name + '.json')

        self.model_configs[kwargs['name']].saved_directory = os.path.join(model_config_directory, "temp")
        self.model_configs[kwargs['name']].save()

        if os.path.exists(model_config_directory):
            zip_path = os.path.join(kwargs['path'], model_config_name + '.zip')
            zf = zipfile.ZipFile(zip_path, mode='w')
            files = os.listdir(os.path.join(model_config_directory, "temp"))
            for file in files:
                path = os.path.join(model_config_directory, "temp", file)
                zf.write(path, file, compress_type=zipfile.ZIP_DEFLATED)
                os.remove(path)
        else:
            raise FileNotFoundError()

        os.rmdir(os.path.join(model_config_directory, "temp"))
        self.models[kwargs['name']].saved_directory = model_config_directory

    def export_data(self, **kwargs):
        """
        Export a data

        :param kwargs:
i
        :Keyword Arguments:
        * *name* (``str``) -- Name of data
        * *path*  (``str``) -- Path to export file
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if kwargs['name'] not in self.data:
            raise NameNotFoundError("Data {0} not found".format(kwargs['name']))
        elif self.data[kwargs['name']] is None:
            raise NotLoadedError("Data {0} not loaded".format(kwargs['name']))
        elif not os.path.exists(kwargs['path']):
            os.makedirs(kwargs['path'], exist_ok=True)

        data_directory = self.data[kwargs['name']].saved_directory
        data_name = self.data[kwargs['name']].saved_name
        file_path = os.path.join(data_directory, data_name + '.json')

        self.data[kwargs['name']].saved_directory = os.path.join(data_directory, "temp")
        self.data[kwargs['name']].save()

        if os.path.exists(data_directory):
            zip_path = os.path.join(kwargs['path'], data_name + '.zip')
            zf = zipfile.ZipFile(zip_path, mode='w')
            files = os.listdir(os.path.join(data_directory, "temp"))
            for file in files:
                path = os.path.join(data_directory, "temp", file)
                zf.write(path, file, compress_type=zipfile.ZIP_DEFLATED)
                os.remove(path)
        else:
            raise FileNotFoundError()

        os.rmdir(os.path.join(data_directory, "temp"))
        self.data[kwargs['name']].saved_directory = data_directory

    def export_embedding(self, **kwargs):
        """
        Export a data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data
        * *path*  (``str``) -- Path to export file
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if kwargs['name'] not in self.embeddings:
            raise NameNotFoundError("Embedding {0} not found".format(kwargs['name']))
        elif self.embeddings[kwargs['name']] is None:
            raise NotLoadedError("Embedding {0} not loaded".format(kwargs['name']))
        elif not os.path.exists(kwargs['path']):
            os.makedirs(kwargs['path'], exist_ok=True)

        embedding_directory = self.embeddings[kwargs['name']].saved_directory
        embedding_name = self.embeddings[kwargs['name']].saved_name
        file_path = os.path.join(embedding_directory, embedding_name + '.json')

        self.embeddings[kwargs['name']].saved_directory = os.path.join(embedding_directory, "temp")
        self.embeddings[kwargs['name']].save()

        if os.path.exists(embedding_directory):
            zip_path = os.path.join(kwargs['path'], embedding_name + '.zip')
            zf = zipfile.ZipFile(zip_path, mode='w')
            files = os.listdir(os.path.join(embedding_directory, "temp"))
            for file in files:
                path = os.path.join(embedding_directory, "temp", file)
                zf.write(path, file, compress_type=zipfile.ZIP_DEFLATED)
                os.remove(path)
        else:
            raise FileNotFoundError()

        os.rmdir(os.path.join(embedding_directory, "temp"))
        self.embeddings[kwargs['name']].saved_directory = embedding_directory

    def import_waifu(self, **kwargs):
        """
        Import a waifu

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu
        * *path*  (``str``) -- Path to import file
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if os.path.isfile(kwargs['path']):
            zf = zipfile.ZipFile(kwargs['path'], 'r')
            file_name = kwargs['name']  # os.path.split(kwargs['path'])[1]
            lists = zf.namelist()

            waifu_dir = os.path.join(self.directories['waifu'], file_name)
            if not os.path.exists(waifu_dir):
                os.mkdir(waifu_dir)

            for file in lists:
                if 'waifu/' in file:
                    zf.extract(file, waifu_dir)

            waifu = _ConsoleItem(None, waifu_dir, file_name)
            self.waifu[file_name] = waifu
            self.waifu[file_name].loaded = False

            model_dir = os.path.join(self.directories['models'], file_name)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            for file in lists:
                if 'model/' in file:
                    zf.extract(file, model_dir)

            model = _ConsoleItem(None, model_dir, file_name)
            self.models[file_name] = model
            self.models[file_name].loaded = False

        else:
            raise FileNotFoundError()

    def import_model(self, **kwargs):

        """
        Import a model

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model
        * *path*  (``str``) -- Path to import file
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if os.path.isfile(kwargs['path']):
            zf = zipfile.ZipFile(kwargs['path'], 'r')
            file_name = kwargs['name']  # os.path.split(kwargs['path'])[1]

            model_dir = os.path.join(self.directories['models'], file_name)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            zf.extractall(model_dir)
            model = _ConsoleItem(None, model_dir, file_name)
            self.models[file_name] = model
            self.models[file_name].loaded = False

        else:
            raise FileNotFoundError()

    def import_model_config(self, **kwargs):
        """
        Import a model config

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model config
        * *path*  (``str``) -- Path to import file
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if os.path.isfile(kwargs['path']):
            zf = zipfile.ZipFile(kwargs['path'], 'r')
            file_name = kwargs['name']  # os.path.split(kwargs['path'])[1]

            model_config_dir = os.path.join(self.directories['model_configs'], file_name)
            if not os.path.exists(model_config_dir):
                os.mkdir(model_config_dir)

            zf.extractall(model_config_dir)
            model_config = _ConsoleItem(None, model_config_dir, file_name)
            self.model_configs[file_name] = model_config
            self.model_configs[file_name].loaded = False

        else:
            raise FileNotFoundError()

    def import_data(self, **kwargs):
        """
         Import a data

         :param kwargs:

         :Keyword Arguments:
         * *name* (``str``) -- Name of data
         * *path*  (``str``) -- Path to import file
         """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if os.path.isfile(kwargs['path']):
            zf = zipfile.ZipFile(kwargs['path'], 'r')
            file_name = kwargs['name']  # os.path.split(kwargs['path'])[1]

            data_dir = os.path.join(self.directories['data'], file_name)
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)

            zf.extractall(data_dir)
            data = _ConsoleItem(None, data_dir, file_name)
            self.data[file_name] = data
            self.data[file_name].loaded = False

        else:
            raise FileNotFoundError()

    def import_embedding(self, **kwargs):
        """
         Import an embedding

         :param kwargs:

         :Keyword Arguments:
         * *name* (``str``) -- Name of embedding
         * *path*  (``str``) -- Path to import file
         """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if os.path.isfile(kwargs['path']):
            zf = zipfile.ZipFile(kwargs['path'], 'r')
            file_name = kwargs['name']  # os.path.split(kwargs['path'])[1]

            embedding_dir = os.path.join(self.directories['embeddings'], file_name)
            if not os.path.exists(embedding_dir):
                os.mkdir(embedding_dir)

            zf.extractall(embedding_dir)
            embedding = _ConsoleItem(None, embedding_dir, file_name)
            self.embeddings[file_name] = embedding
            self.embeddings[file_name].loaded = False

        else:
            raise FileNotFoundError()

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
        results = {}

        for key in self.models:
            if self.models[key].loaded:
                results[key] = {'name': key,
                                'type': type(self.models[key].item)}
            else:
                results[key] = {}
        return results

    def get_model_configs(self, **kwargs):
        results = {}

        for key in self.model_configs:
            if self.model_configs[key].loaded:
                results[key] = {'name': key}
            else:
                results[key] = {}
        return results

    def get_data(self, **kwargs):
        results = {}

        for key in self.data:
            if self.data[key].loaded:
                results[key] = {'name': key,
                                'type': type(self.data[key].item)}
            else:
                results[key] = {}
        return results

    def get_embeddings(self, **kwargs):
        results = {}

        for key in self.embeddings:
            if self.embeddings[key].loaded:
                results[key] = {'name': key}
            else:
                results[key] = {}
        return results

    def get_waifu_detail(self, **kwargs):
        """
        Return the details of a waifu

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu
        * *base64* (``boolean``) -- Convert image to base64 (Optional)
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.waifu:
            raise NameNotFoundError("Waifu {0} not found".format(kwargs['name']))

        tmp = dict(self.waifu[kwargs['name']].item.config)

        if 'base64' in kwargs and kwargs['base64'] and os.path.isfile(tmp['image']):
            with open(tmp['image'], "rb") as imageFile:
                tmp['image'] = base64.b64encode(imageFile.read())
                tmp['image'].decode()

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
               'hyperparamter': self.models[kwargs['name']].item.hyperparameters,
               'saved_directory': self.models[kwargs['name']].item.saved_directory,
               'saved_name': self.models[kwargs['name']].item.saved_name}

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

    def get_model_config_details(self, **kwargs):
        """
        Return the details of a model config

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model config
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.model_configs:
            raise NameNotFoundError("Model Config {0} not found".format(kwargs['name']))

        tmp = {'config': self.model_configs[kwargs['name']].item.config,
               'model_structure': self.model_configs[kwargs['name']].item.model_structure,
               'hyperparamters': self.model_configs[kwargs['name']].item.hyperparameters,
               'saved_directory': self.model_configs[kwargs['name']].saved_directory,
               'saved_name': self.model_configs[kwargs['name']].saved_name}

        return tmp

    def get_embedding_details(self, **kwargs):
        """
        Return the details of an embedding

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of embedding
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.embeddings:
            raise NameNotFoundError("Embedding {0} not found".format(kwargs['name']))

        tmp = {'saved_directory': self.model_configs[kwargs['name']].saved_directory,
               'saved_name': self.model_configs[kwargs['name']].saved_name}

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

    def slice_audio(self, **kwargs):
        """
        Loading subtitle and slicing audio

        :param kwargs:

        :Keyword Arguments:
        * *subtitle_path* (``str``)  -- Path to subtitle file
        * *audio_path* (``str``) -- Path to audio file
        * *save_path* (``str``) -- Path to save audio
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['subtitle_path', 'audio_path', 'save_path'])

        if not os.path.isfile(kwargs['subtitle_path']):
            raise FileNotFoundError('Subtitle File Does Not Exist')
        elif not os.path.isfile(kwargs['audio_path']):
            raise FileNotFoundError('Audio File Does Not Exist')
        elif not os.path.exists(kwargs['save_path']):
            raise NotADirectoryError('Save Path Not Found')

        parser = am.SubtitleParser()

        parser.load(kwargs['subtitle_path'])
        parser.slice_audio(kwargs['audio_path'], kwargs['save_path'])

    def create_waifu(self, **kwargs):
        """
        Use existing model to create a waifu

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu
        * *combined_chatbot_model* (``str``) -- Name or directory of combined chatbot model to use
        * *embedding* (``str``) -- Name of embedding
        * *description* (``str``) -- Description of waifu. Optional
        * *image* (``str``) -- Image of waifu. (path or base64 string) Optional
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'combined_chatbot_model', 'embedding'])

        if kwargs['name'] in self.waifu:
            raise NameAlreadyExistError("The name {0} is already used by another waifu".format(kwargs['name']))

        if kwargs['embedding'] not in self.embeddings:
            raise NameNotFoundError("Embedding {0} no found".format(kwargs['embedding']))

        if not os.path.isdir(kwargs['combined_chatbot_model']):
            if kwargs['combined_chatbot_model'] in self.models:
                model_directory = self.models[kwargs['combined_chatbot_model']].saved_directory
                model_name = self.models[kwargs['combined_chatbot_model']].saved_name
            else:
                raise NameNotFoundError("Model {0} not found".format(kwargs['model']))
        else:
            # try parsing it as a path
            # not really supposed to be dosing this when using console
            norm_path = os.path.normpath(kwargs['combined_chatbot_model'])
            model_directory = os.path.dirname(norm_path)
            model_name = os.path.splitext(os.path.basename(norm_path))[0]

        desc = '' if 'description' not in kwargs else kwargs['description']

        # [file_type, base64 string] or 'file_path'
        image = '' if 'image' not in kwargs else kwargs['image']

        waifu = am.Waifu(kwargs['name'], description=desc, image=image)
        waifu.add_combined_chatbot_model(model_directory, model_name)
        waifu.add_embedding(self.embeddings[kwargs['embedding']].item)

        console_item = _ConsoleItem(waifu, os.path.join(self.directories['waifu'], kwargs['name']), kwargs['name'])
        # saving it first to set up its saving location
        console_item.save()

        self.waifu[kwargs['name']] = console_item

    def edit_waifu(self, **kwargs):
        """
        Edit details of waifu

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu
        * *new_name* (``str``) -- Change the name of waifu Optional
        * *image* (``str``) -- Set image of waifu (path or base64 string) Optional
        * *description* (``str``) -- Change Description of waifu. Optional
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'combined_chatbot_model', 'embedding'])

        if kwargs['name'] in self.waifu:
            raise NameNotFoundError("Waifu {0} not found".format(kwargs['name']))

        waifu = self.waifu[kwargs['name']]

        if 'image' in kwargs['waifu']:
            waifu.config['image'] = kwargs['image']

        if 'description' in kwargs['waifu']:
            waifu.config['description'] = kwargs['description']

        if 'new_name' in kwargs['waifu']:
            waifu.config['name'] = kwargs['new_name']
            self.waifu.pop(kwargs['name'])
            self.waifu['new_name'] = waifu
            waifu.save()

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

    def add_regex(self, **kwargs):
        """
        Add regex rule to a waifu

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of waifu
        * *regex* (``str``) -- Regex rule
        * *intent* (``str``) -- Intent (Optional)
        * *chat* (``str``) -- Chat message(Optional)
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'regex'])
        if kwargs['name'] not in self.waifu:
            raise NameNotFoundError("Waifu \"{0}\" not found".format(kwargs['name']))

        if 'chat' in kwargs:
            self.waifu[kwargs['name']].item.config['regex_rule'][kwargs['regex']] = [False, kwargs['chat']]

        elif 'intent' in kwargs:
            self.waifu[kwargs['name']].item.config['regex_rule'][kwargs['regex']] = [True, kwargs['intent']]

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
        * *intent_ner_model* (``str``) -- Name of IntentNER Model (Only required for creating CombinedChatbot Model)
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

            model = am.Chatbot.CombinedChatbotModel()
            model_config = self.model_configs[kwargs['model_config']].item

            graph_path = os.path.join(self.models[kwargs['intent_ner_model']].saved_directory + "frozen_model.pb")
            if os.path.isfile(graph_path):
                model_config.config['intent_ner_path'] = graph_path
                model.build_graph(model_config, self.data[kwargs['data']].item)

        elif kwargs['type'] == 'SpeakerVerification':
            model = am.SpeakerVerification.SpeakerVerificationModel()

        else:
            raise KeyError("Model type \"{0}\" not found.".format(kwargs['type']))

        # CombinedChatbotModel builds graph in its constructor
        if kwargs['type'] != 'CombinedChatbotModel':
            model.build_graph(self.model_configs[kwargs['model_config']].item, self.data[kwargs['data']].item)

        model.init_tensorflow()
        console_item = _ConsoleItem(model, os.path.join(self.directories['models'], kwargs['name']), kwargs['name'])
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
        * *graph* (``bool``) -- Save graph
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model \"{0}\" not found".format(kwargs['name']))
        if 'graph' in kwargs and kwargs['graph'] is True:
            self.models[kwargs['name']].item.save(graph=True)
        else:
            self.models[kwargs['name']].item.save(graph=False)

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
    def train_complete_callback(print_string, model_name, training_dict, future_object):
        """
        Called when a model finishes training. See train() for details
        """
        print(print_string)
        future_object.result()
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
                                           cancellation_token=cancelToken)

        callback_string = "Model {0} has finished training for {1} epochs!".format(kwargs['name'], kwargs['epoch'])

        func = partial(Console.train_complete_callback,
                       callback_string, kwargs['name'], self.training_models)

        future.add_done_callback(func)

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
        * *name* (``str``) -- Name of model to predict.
        * *input_data* (``str``) -- Name of input data. (Optional)
        * *input* (``str``) -- String to input. (Optional)
        * *save_path* (``str``) -- Path to save result. (Optional)
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name'],
                                soft_requirements=['save_path'])

        if kwargs['name'] not in self.models:
            raise NameNotFoundError("Model \"{0}\" not found".format(kwargs['name']))

        if 'input_data' in kwargs:
            if kwargs['input_data'] not in self.data:
                raise NameNotFoundError("Data \"{0}\" not found".format(kwargs['input_data']))
            else:
                if 'save_path' in kwargs:
                    result = self.models[kwargs['name']].item.predict(kwargs['input_data'],
                                                                      save_path=kwargs['save_path'])
                else:
                    result = self.models[kwargs['name']].item.predict(kwargs['input_data'])

        elif 'input' in kwargs:
            if 'save_path' in kwargs:
                result = self.models[kwargs['name']].item.predict(kwargs['input'], save_path=kwargs['save_path'])
            else:
                result = self.models[kwargs['name']].item.predict(kwargs['input'])

        else:
            result = ''

        return result

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

        am.Utils.freeze_graph(self.models[kwargs['name']].item, output_node_names, stored)

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

        if kwargs['type'] not in ['SpeakerVerification', 'Chatbot', 'IntentNER', 'CombinedChatbot']:
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
        Update a model config with provided values

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of model config to edit
        * *config* (``dict``) -- Dictionary containing the updated config values
        * *hyperparameters* (``dict``) -- Dictionary containing the updated hyperparameters values
        * *model_structure* (``model_structure``) -- Dictionary containing the updated model_structure values
        """

        Console.check_arguments(kwargs, hard_requirements=['name'])

        configs = ['device', 'class', 'epoch', 'cost', 'display_step', 'tensorboard', 'hyperdash']
        hyperparameters = ['learning_rate', 'batch_size', 'optimizer']
        intent_model_structures = ['n_ner_output', 'n_intent_output', 'node', 'gradient_clip', 'n_hidden',
                                   'max_sequence']
        chatbot_model_structures = ['max_sequence', 'n_hidden', 'gradient_clip', 'node', 'layer', 'beam_width']
        speaker_model_structures = ['filter_size_1', 'num_filter_1', 'pool_size_1', 'pool_type', 'filter_size_2',
                                    'num_filter_2', 'fully_connected_1', 'input_window', 'input_cepstral']

        if kwargs['name'] in self.model_configs:
            def update_dict(target, update_values):
                for key in update_values:
                    target[key] = update_values[key]

            for key, value in kwargs.items():

                if isinstance(value, dict):
                    if key == 'config':
                        update_dict(self.model_configs[kwargs['name']].item.config, kwargs['config'])
                    if key == 'hyperparameters':
                        update_dict(self.model_configs[kwargs['name']].item.hyperparameters, kwargs['hyperparamters'])
                    if key == 'model_structure':
                        update_dict(self.model_configs[kwargs['name']].item.model_structure, kwargs['model_structure'])

                else:
                    if key in configs:
                        self.model_configs[kwargs['name']].item.config[key] = value
                    if key in hyperparameters:
                        self.model_configs[kwargs['name']].item.hyperparameters[key] = value
                    if key in intent_model_structures or key in chatbot_model_structures or key in speaker_model_structures:
                        self.model_configs[kwargs['name']].item.model_structure[key] = value

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
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'type'])

        if kwargs['name'] in self.data:
            raise NameAlreadyExistError("The name {0} is already used by another data".format(kwargs['name']))

        if kwargs['type'] == 'Chatbot' or kwargs['type'] == 'CombinedChatbot':
            data = am.ChatData()
        elif kwargs['type'] == 'IntentNER':
            data = am.IntentNERData()
        elif kwargs['type'] == 'SpeakerVerification':
            data = am.SpeakerVerificationData()
        else:
            raise KeyError("Data type \"{0}\" not found.".format(kwargs['type']))

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
        * *embedding* (``str``) -- Name of the embedding to add to data
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'embedding'])

        if kwargs['name'] in self.data:
            if kwargs['embedding'] in self.embeddings:
                self.data[kwargs['name']].item.add_embedding_class(self.embeddings[kwargs['embedding']].item)
            else:
                raise KeyError("Embedding \"{0}\" not found.".format(kwargs['embedding']))
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
            if isinstance(self.data[kwargs['name']].item, am.ChatData):
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
            if isinstance(self.data[kwargs['name']].item, am.ChatData):
                self.data[kwargs['name']].item.add_cornell(kwargs['movie_conversations_path'],
                                                           kwargs['movie_lines_path'])
            else:
                raise KeyError("Data \"{0}\" is not a ChatbotData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def chatbot_data_add_files(self, **kwargs):
        """
        Add text files to a chatbot data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *x_path* (``str``) -- Path to a UTF-8 file containing a raw sentence input on each line
        * *y_path* (``str``) -- Path to a UTF-8 file containing a raw sentence output on each line
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'x_path', 'y_path'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.ChatData):
                self.data[kwargs['name']].item.add_files(kwargs['x_path'], kwargs['y_path'])
            else:
                raise KeyError("Data \"{0}\" is not a ChatbotData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def chatbot_data_add_input(self, **kwargs):
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
            if isinstance(self.data[kwargs['name']].item, am.ChatData):
                self.data[kwargs['name']].item.add_input(kwargs['x'])
            else:
                raise KeyError("Data \"{0}\" is not a ChatbotData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def chatbot_data_set_input(self, **kwargs):
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
            if isinstance(self.data[kwargs['name']].item, am.ChatData):
                self.data[kwargs['name']].item.set_input(kwargs['x'])
            else:
                raise KeyError("Data \"{0}\" is not a ChatbotData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def intentNER_data_set_intent_folder(self, **kwargs):
        """
        Set folder for IntentNER Data.

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *path* (``str``) -- Path to the intent folder
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.IntentNERData):
                self.data[kwargs['name']].item.set_intent_folder(kwargs['path'])
            else:
                raise KeyError("Data \"{0}\" is not a IntentNERData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def intentNER_data_add_input(self, **kwargs):
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
                self.data[kwargs['name']].item.add_input(kwargs['x'])
            else:
                raise KeyError("Data \"{0}\" is not a IntentNERData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def intentNER_data_set_input(self, **kwargs):
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
                self.data[kwargs['name']].item.set_input(kwargs['x'])
            else:
                raise KeyError("Data \"{0}\" is not a IntentNERData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def speakerVerification_data_add_folder(self, **kwargs):
        """
        Add folder to a speaker verification data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *path* (``str``) -- Path to folder to add on
        * *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction.
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'],
                                soft_requirements=['y'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.SpeakerVerificationData):
                self.data[kwargs['name']].item.add_folder(kwargs['paths'], kwargs['y'])
            else:
                raise KeyError("Data \"{0}\" is not a SpeakerVerificationData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def speakerVerification_data_set_folder(self, **kwargs):
        """
        Set folder to a speaker verification data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *path* (``str``) -- Path to folder to set
        * *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction.
        """
        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'],
                                soft_requirements=['y'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.SpeakerVerificationData):
                self.data[kwargs['name']].item.set_folder(kwargs['paths'], kwargs['y'])
            else:
                raise KeyError("Data \"{0}\" is not a SpeakerVerificationData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def speakerVerification_data_add_wav_file(self, **kwargs):
        """
        Add wav file to a speaker verification data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *path* (``str``) -- Path to wav file to add on
        * *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction.
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'],
                                soft_requirements=['y'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.SpeakerVerificationData):
                self.data[kwargs['name']].item.add_wav_file(kwargs['paths'], kwargs['y'])
            else:
                raise KeyError("Data \"{0}\" is not a SpeakerVerificationData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def speakerVerification_data_set_wav_file(self, **kwargs):
        """
        Set wav file to a speaker verification data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *path* (``str``) -- Path to wav file to set
        * *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction.
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'],
                                soft_requirements=['y'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.SpeakerVerificationData):
                self.data[kwargs['name']].item.set_wav_file(kwargs['paths'], kwargs['y'])
            else:
                raise KeyError("Data \"{0}\" is not a SpeakerVerificationData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def speakerVerification_data_add_text_file(self, **kwargs):
        """
        Add text file to a speaker verification data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *path* (``str``) -- Path to text file to add on
        * *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction.
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'],
                                soft_requirements=['y'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.SpeakerVerificationData):
                self.data[kwargs['name']].item.add_text_file(kwargs['paths'], kwargs['y'])
            else:
                raise KeyError("Data \"{0}\" is not a SpeakerVerificationData.".format(kwargs['name']))
        else:
            raise KeyError("Data \"{0}\" not found.".format(kwargs['name']))

    def speakerVerification_data_set_text_file(self, **kwargs):
        """
        Set wav file to a speaker verification data

        :param kwargs:

        :Keyword Arguments:
        * *name* (``str``) -- Name of data to add on
        * *path* (``str``) -- Path to wav file to set
        * *y* (``bool``) -- The label (True for is speaker and vice versa) of the audio files. Optional. Include for training, leave out for prediction.
        """

        Console.check_arguments(kwargs,
                                hard_requirements=['name', 'path'],
                                soft_requirements=['y'])

        if kwargs['name'] in self.data:
            if isinstance(self.data[kwargs['name']].item, am.SpeakerVerificationData):
                self.data[kwargs['name']].item.set_text_file(kwargs['paths'], kwargs['y'])
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

    def get_system_info(self):
        """
        Get System Info such as CPU usage, Memory usage, etc.
        """

        # {'cpu_percent': 25.5, 'cpu_count': 6, 'mem_total': 16307, 'mem_available': 11110, 'mem_percent': 31.9,
        # 'disk_total': 339338, 'disk_used': 237581, 'disk_percent': 70.0, 'boot_time': 1556635120.0,
        # 'gpu_driver_version': '430.39', 'gpu_device_list': [
        # {'gpu_name': 'GeForce GTX 1060 6GB', 'gpu_mem_total': 6144, 'gpu_mem_used': 449, 'gpu_mem_percent': 0}]}

        system_info = am.Utils.get_system_info()

        for key in system_info:
            print(key, ':', system_info[key])

        return system_info

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
                                soft_requirements=['local', 'password', 'max_clients'])

        if self.socket_server is not None:
            raise ValueError("A server is already running on this console.")

        if kwargs['max_clients'] is None:
            kwargs['max_clients'] = 10

        if kwargs['password'] is None:
            kwargs['password'] = ''

        if kwargs['local'] is None:
            kwargs['local'] = True

        self.socket_server = \
            self.server(self, kwargs['port'], kwargs['local'], kwargs['password'], kwargs['max_clients'])

    def server(self, console, port, local=True, password='', max_clients=10):
        from .SocketServer import _ServerThread
        thread = _ServerThread(console, port, local, password, max_clients)
        thread.daemon = True
        thread.start()
        return thread

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
        try:
            print(request.id, request.command)
            # initialize commands first
            if self.commands is None:
                self.init_commands()

            # command must be pre-defined
            if request.command not in self.commands:
                raise Exception('Invalid command')

            method_to_call = self.commands[request.command][0]

            if request.arguments == '':
                result = method_to_call.__call__()
            else:
                result = method_to_call.__call__(**request.arguments)

            if result is None:
                result = {}
            return request.id, 0, 'success', result
        except ArgumentError as exc:
            return request.id, 1, str(exc), {}
        except Exception as exc:  # all other errors
            return request.id, 2, str(exc), {}

    def handle_command(self, user_input):

        try:
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

                        result = self.commands[command][0].__call__(**kwargs)
                        if result is not None:
                            print(result)

                else:
                    print('Invalid command')

        except Exception as exc:
            print('{0}: {1}'.format(type(exc).__name__, exc))

    @staticmethod
    def start():
        import readline
        TaskQueue = queue.Queue(0)
        ResultQueue = queue.Queue(0)
        console = am.Console()

        thread = _ClientThread(console, TaskQueue, ResultQueue)
        thread.daemon = True
        thread.start()

        def completer(user_input, state):
            options = [i for i in console.commands if i.startswith(user_input)]
            if state < len(options):
                return options[state]
            else:
                return None

        readline.parse_and_bind("tab: complete")
        readline.set_completer(completer)

        print("Animius. Type 'help' or '?' to list commands. Use 'exit' to break")

        while True:
            user_input = input('Input: ')

            if user_input.lower() == 'exit':
                break
            TaskQueue.put(user_input)


class _ClientThread(threading.Thread):
    def __init__(self, console, TaskQueue, ResultQueue):
        super(_ClientThread, self).__init__()

        self.console = console
        self.console.init_commands()
        self.console.queue = [TaskQueue, ResultQueue]

    def run(self):
        while True:
            if not self.console.queue[0].empty():
                task = self.console.queue[0].get()
                if isinstance(task, str):
                    self.console.handle_command(task)
                else:
                    id, status, result, data = self.console.handle_network(task)
                    self.console.queue[1].put({"id": id, "status": status, "result": result, "data": data})

                self.console.queue[0].task_done()

    def stop(self):
        self.console.queue = None
