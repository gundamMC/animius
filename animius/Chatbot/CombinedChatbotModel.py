import animius as am
import tensorflow as tf
from .ChatbotNetwork import ChatbotModel


class CombinedChatbotModel(ChatbotModel):

    def __init__(self, model_config, data, restore_path=None):

        # restoring graph
        if restore_path is not None:

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            self.sess = tf.Session(graph=tf.Graph(), config=config)

            checkpoint = tf.train.get_checkpoint_state(restore_path)
            input_checkpoint = checkpoint.model_checkpoint_path

            with self.sess.graph.as_default():

                self.saver = tf.train.import_meta_graph(input_checkpoint + '.meta')

                self.saver.restore(self.sess, input_checkpoint)

            super().__init__(None, data, restore_path=restore_path,
                             embedding_tensor=None,
                             graph=self.sess.graph)

            return

        # creating new graph
        intent_ner_path = model_config.config['intent_ner_path']

        intent_ner_graph_def = tf.GraphDef()
        with tf.gfile.Open(intent_ner_path, "rb") as f:
            data2read = f.read()
            intent_ner_graph_def.ParseFromString(data2read)

        intent_ner_graph = tf.Graph()
        with intent_ner_graph.as_default():
            tf.import_graph_def(intent_ner_graph_def, name='intent')

        super().__init__(model_config, data, restore_path=None,
                         embedding_tensor=intent_ner_graph.get_tensor_by_name('intent/word_embedding:0'),
                         graph=intent_ner_graph)


    def predict(self, input_data, save_path=None):

        intent, ner = self.sess.run([self.sess.graph.get_tensor_by_name('intent/output_intent:0'),
                                     self.sess.graph.get_tensor_by_name('intent/output_ner:0')],
                                    feed_dict={
                                        self.x: input_data.values['x'],
                                        self.x_length: input_data.values['x_length']
                                    })
