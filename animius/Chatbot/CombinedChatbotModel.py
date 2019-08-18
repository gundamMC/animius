import copy

import tensorflow as tf

import animius as am
from .ChatbotModel import ChatbotModel


# A chatbot network built upon an intent-ner model, using its embedding tensor and thus saving VRAM.
# This model is meant for training. Once training is complete, it is advised to freeze the model
# and use the CombinedPredictionModel class instead.
class CombinedChatbotModel(ChatbotModel):

    def __init__(self):

        super().__init__()

        self.intent_ner_model = None
        self.intent_ner_initialized = False
        self.train_intent_ner = None
        self.predict_intent_ner = None
        self.predict_chatbot = super().predict

        self.init_vars = None

    def build_graph(self, model_config, data, graph=None, embedding_tensor=None, intent_ner=None):
        # graph and embedding_tensor arguments doesn't really do anything

        # if data is None or 'embedding' not in data.values:
        #     raise ValueError('When creating a new model, data must contain a word embedding')

        def copy_embedding(new_data):
            if (new_data is None or 'embedding' not in new_data.values) and 'embedding' in data.values:
                new_data.add_embedding_class(data.values['embedding'])

        # intent_ner arg can be IntentNERModel, model config for intent ner, string, tuple of string
        # tuple of model config and data and/or model, or none for a new intent ner model
        if intent_ner is None:

            if 'intent_ner' in model_config.config:
                # storing intent ner in model config, most liekly used for saving / restoring
                intent_ner = model_config.config['intent_ner']
            else:
                # create a new intent ner model
                self.intent_ner_model = am.IntentNER.IntentNERModel()

        if isinstance(intent_ner, am.ModelConfig):
            self.intent_ner_model = am.IntentNER.IntentNERModel()
            intent_ner_data = am.IntentNERData()
            intent_ner_data.add_embedding_class(data.values['embedding'])
            self.intent_ner_model.build_graph(intent_ner, intent_ner_data)

        elif isinstance(intent_ner, am.IntentNER.IntentNERModel):
            self.intent_ner_model = am.IntentNER.IntentNERModel()

            copy_embedding(self.intent_ner_model.data)

            if self.intent_ner_model.cost is None:  # check if model has already been built
                self.intent_ner_model.build_graph(am.ModelConfig(cls='IntentNER'), am.IntentNERData())
            elif self.intent_ner_model.sess is not None:
                self.intent_ner_initialized = True  # we don't need to initialize the model

        elif isinstance(intent_ner, str):
            self.intent_ner_model = am.Model.load(intent_ner)
            copy_embedding(self.intent_ner_model.data)
            self.intent_ner_initialized = True

        elif isinstance(intent_ner, tuple) or isinstance(intent_ner, list):
            if len(intent_ner) == 3:
                self.intent_ner_model, mc, new_data = intent_ner
                copy_embedding(new_data)
                if self.intent_ner_model.cost is None:  # check if model has already been built
                    self.intent_ner_model.build_graph(mc, new_data)

            elif len(intent_ner) == 2:
                if isinstance(intent_ner[0], str):
                    # tuple of string, pair of (directory, name)
                    self.intent_ner_model = am.Model.load(intent_ner[0], intent_ner[1])
                    copy_embedding(self.intent_ner_model.data)
                    self.intent_ner_initialized = True
                else:
                    # assume tuple of model config and data
                    mc, new_data = intent_ner
                    copy_embedding(new_data)
                    self.intent_ner_model = am.IntentNER.IntentNERModel()
                    self.intent_ner_model.build_graph(mc, data)
            else:
                raise ValueError("Unexpected tuple of intent_ner")

        else:
            raise TypeError("Unexpected type of intent_ner")

        # at this point, self.intent_ner_model should be built
        self.train_intent_ner = self.intent_ner_model.train
        self.predict_intent_ner = self.intent_ner_model.predict

        with self.intent_ner_model.graph.as_default():
            intent_vars = set(tf.all_variables())

        super().build_graph(model_config,
                            data,
                            embedding_tensor=self.intent_ner_model.word_embedding,
                            graph=self.intent_ner_model.graph)
        with self.graph.as_default():
            all_vars = set(tf.all_variables())
        self.init_vars = all_vars - intent_vars

        self.config = dict(model_config.config)
        self.config['class'] = 'CombinedChatbot'
        self.model_structure = dict(model_config.model_structure)
        self.hyperparameters = dict(model_config.hyperparameters)
        self.data = data

    def init_tensorflow(self, graph=None, init_param=True, init_sess=True):
        if init_param and self.intent_ner_initialized:
            # we can only initialize the chatbot vars

            self.sess = self.intent_ner_model.sess
            self.graph = self.intent_ner_model.graph

            super().init_tensorflow(graph=self.graph, init_param=False, init_sess=False)

            with self.graph.as_default():

                self.sess.run(
                    tf.variables_initializer(self.init_vars)
                )

        else:
            super().init_tensorflow(graph=graph, init_param=init_param, init_sess=init_sess)

            self.intent_ner_model.sess = self.sess

    @classmethod
    def load(cls, directory, name='model', data=None):

        model = CombinedChatbotModel()
        model.restore_config(directory, name)
        if data is not None:
            model.data = data
        else:
            model.data = am.ChatData()

        model.build_graph(model.model_config(), model.data)  # automatically builds intent ner in model config
        model.init_word_embedding = False  # prevent initializing the word embedding again
        model.init_tensorflow(init_param=False, init_sess=True)

        checkpoint = tf.train.get_checkpoint_state(directory)
        input_checkpoint = checkpoint.model_checkpoint_path

        with model.graph.as_default():
            model.saver.restore(model.sess, input_checkpoint)

        model.saved_directory = directory
        model.saved_name = name

        return model

    def save(self, directory=None, name='model', meta=True, graph=False):

        if self.intent_ner_model.saved_directory is None and self.intent_ner_model.saved_name is None:
            # new model
            self.intent_ner_model.save(directory=directory, name=name + '_intent_ner')
            self.config['intent_ner'] = (directory, name + '_intent_ner')
        else:
            self.intent_ner_model.save()  # default to model save
            self.config['intent_ner'] = (self.intent_ner_model.saved_directory, self.intent_ner_model.saved_name)

        super().save(directory, name, meta, graph)

    def add_embedding(self, embedding):
        # shortcut for adding embedding
        self.data.add_embedding_class(embedding)
        self.intent_ner_model.data.add_embedding_class(embedding)

    def predict_combined(self, input_sentences=None, save_path=None):

        if input_sentences is None:
            input_sentences = self.data

        if isinstance(input_sentences, am.IntentNERData) or isinstance(input_sentences, am.ChatData):
            input_sentences = input_sentences.values['input']

        # package str in a list
        if isinstance(input_sentences, str):
            input_sentences = [input_sentences]

        sentences_cache = copy.copy(input_sentences)

        intent_ner_results = self.predict_intent_ner(input_sentences, raw=False)

        results = []
        chat_indexes = []

        for i in range(len(intent_ner_results)):
            intent = intent_ner_results[i][0]
            if intent > 0:  # not chat
                results.append(intent_ner_results[i])
            else:  # chat
                chat_indexes.append(i)
                results.append(None)  # add tmp placeholder

        if len(chat_indexes) > 0:  # there are chat responses, proceed with chatbot prediction
            chat_results = super().predict([sentences_cache[i] for i in chat_indexes], raw=False)

            for i in range(len(chat_results)):
                results[chat_indexes[i]] = (0, chat_results[i])

        # saving
        if save_path is not None:
            with open(save_path, "w") as file:
                for instance in results:
                    file.write('{0}, {1}\n'.format(*instance))

        return results

    def predict(self, input_data=None, save_path=None, raw=False, combined=True):
        # automatically selects a prediction function
        if combined:
            # use combined
            return self.predict_combined(input_sentences=input_data, save_path=save_path)
        elif isinstance(input_data, am.IntentNERData):
            return self.predict_intent_ner(input_data, save_path, raw)
        else:
            return self.predict_chatbot(input_data, save_path, raw)
