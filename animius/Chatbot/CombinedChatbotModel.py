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

        self.init_vars = None

    def build_graph(self, model_config, data, graph=None, embedding_tensor=None, intent_ner=None):
        # graph and embedding_tensor arguments doesn't really do anything

        if data is None or 'embedding' not in data.values:
            raise ValueError('When creating a new model, data must contain a word embedding')

        def copy_embedding(new_data):
            if new_data is None or 'embedding' not in new_data.values:
                new_data.add_embedding_class(data.values['embedding'])

        # intent_ner arg can be IntentNERModel, model config for intent ner, string, tuple of string
        # tuple of model config and data and/or model, or none for a new intent ner model
        if intent_ner is None:
            self.intent_ner_model = am.IntentNER.IntentNERModel()
            intent_ner_data = am.IntentNERData()
            intent_ner_data.add_embedding_class(data.values['embedding'])  # copy embedding over
            self.intent_ner_model.build_graph(am.ModelConfig(cls='IntentNER'), intent_ner_data)

        elif isinstance(intent_ner, am.ModelConfig):
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

        elif isinstance(intent_ner, tuple):
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

        graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        model.sess = tf.Session(config=config, graph=graph)

        checkpoint = tf.train.get_checkpoint_state(directory)
        input_checkpoint = checkpoint.model_checkpoint_path

        with graph.as_default():
            model.saver = tf.train.import_meta_graph(input_checkpoint + '.meta')
            model.saver.restore(model.sess, input_checkpoint)

        # set up self vars and ops for predict/training
        model.x = model.sess.graph.get_tensor_by_name('input_x:0')
        model.x_length = model.sess.graph.get_tensor_by_name('input_x_length:0')
        model.y = model.sess.graph.get_tensor_by_name('train_y:0')
        model.y_length = model.sess.graph.get_tensor_by_name('train_y_length:0')
        model.y_target = model.sess.graph.get_tensor_by_name('train_y_target:0')
        model.train_op = model.sess.graph.get_operation_by_name('train_op')
        model.cost = model.sess.graph.get_tensor_by_name('train_cost/truediv:0')
        model.infer = model.sess.graph.get_tensor_by_name('decode_1/output_infer:0')

        # initialize tensorboard and hyperdash
        model.init_tensorflow(graph, init_param=False, init_sess=False)

        model.saved_directory = directory
        model.saved_name = name

        return model
