import tensorflow as tf

from .ChatbotModel import ChatbotModel


# A chatbot network built upon an intent-ner model, using its embedding tensor and thus saving VRAM.
# This model is meant for training. Once training is complete, it is advised to freeze the model
# and use the CombinedPredictionModel class instead.
class CombinedChatbotModel(ChatbotModel):

    def __init__(self):

        super().__init__()

    def build_graph(self, model_config, data, graph=None, embedding_tensor=None):
        # graph and embedding_tensor arguments doesn't really do anything

        # creating new graph
        intent_ner_path = model_config.config['intent_ner_path']

        intent_ner_graph_def = tf.GraphDef()
        with tf.gfile.Open(intent_ner_path, "rb") as f:
            intent_ner_graph_def.ParseFromString(f.read())

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(intent_ner_graph_def, name='intent')

        super().build_graph(model_config,
                            data,
                            embedding_tensor=self.graph.get_tensor_by_name('intent/word_embedding:0'),
                            graph=self.graph)

        self.config['class'] = 'CombinedChatbot'

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
