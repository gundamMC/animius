import tensorflow as tf
from .ChatbotModel import ChatbotModel


# A chatbot network built upon an intent-ner model, using its embedding tensor and thus saving VRAM.
# This model is meant for training. Once training is complete, it is advised to freeze the model
# and use the CombinedPredictionModel class instead.
class CombinedChatbotModel(ChatbotModel):

    def __init__(self, model_config, data, restore_directory=None):

        # restoring graph
        if restore_directory is not None:

            self.restore_config(restore_directory)
            self.data = data

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            self.sess = tf.Session(graph=tf.Graph(), config=config)

            checkpoint = tf.train.get_checkpoint_state(restore_directory)
            input_checkpoint = checkpoint.model_checkpoint_path

            with self.sess.graph.as_default():

                self.saver = tf.train.import_meta_graph(input_checkpoint + '.meta')

                self.saver.restore(self.sess, input_checkpoint)

                # set up self vars and ops for predict/training
                self.x = self.sess.graph.get_tensor_by_name('input_x:0')
                self.x_length = self.sess.graph.get_tensor_by_name('input_x_length:0')
                self.y = self.sess.graph.get_tensor_by_name('train_y:0')
                self.y_length = self.sess.graph.get_tensor_by_name('train_y_length:0')
                self.y_target = self.sess.graph.get_tensor_by_name('train_y_target:0')
                self.train_op = self.sess.graph.get_operation_by_name('train_op')
                self.cost = self.sess.graph.get_tensor_by_name('train_cost/truediv:0')
                self.infer = self.sess.graph.get_tensor_by_name('decode_1/output_infer:0')

                # initialize tensorboard and hyperdash
                self.init_tensorflow(graph=self.sess.graph, init_param=False, init_sess=False)
                self.init_hyerdash(self.config['hyperdash'])

            return

        # creating new graph
        intent_ner_path = model_config.config['intent_ner_path']

        intent_ner_graph_def = tf.GraphDef()
        with tf.gfile.Open(intent_ner_path, "rb") as f:
            intent_ner_graph_def.ParseFromString(f.read())

        intent_ner_graph = tf.Graph()
        with intent_ner_graph.as_default():
            tf.import_graph_def(intent_ner_graph_def, name='intent')

        super().__init__()

        self.build_graph(model_config,
                         data,
                         embedding_tensor=intent_ner_graph.get_tensor_by_name('intent/word_embedding:0'),
                         graph=intent_ner_graph)

        self.init_tensorflow()
