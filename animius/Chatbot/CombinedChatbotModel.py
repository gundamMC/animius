import animius as am
import tensorflow as tf
from .ChatbotNetwork import ChatbotModel


class CombinedChatbotModel(ChatbotModel):

    def __init__(self, model_config, data, restore_path=None):

        intent_ner_path = model_config.config['intent_ner_path']

        intent_ner_graph_def = tf.GraphDef()
        with tf.gfile.Open(intent_ner_path, "rb") as f:
            data2read = f.read()
            intent_ner_graph_def.ParseFromString(data2read)

        intent_ner_graph = tf.Graph()
        with intent_ner_graph.as_default():
            tf.import_graph_def(intent_ner_graph_def, name='intent')

        super().__init__(model_config, data, restore_path=None,
                         embedding_tensor=intent_ner_graph.get_tensor_by_name('intent/word_embedding:0'), graph=intent_ner_graph)
