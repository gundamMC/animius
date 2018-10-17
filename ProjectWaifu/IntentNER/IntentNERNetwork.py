import tensorflow as tf
import numpy as np
from ProjectWaifu.IntentNER.ParseData import get_data, sentence_to_vec
from ProjectWaifu import Utils
from ProjectWaifu.Model import Model
import os


class IntentNERModel(Model):

    # default values
    @staticmethod
    def DEFAULT_HYPERPARAMETERS():
        return {
            'learning_rate': 0.001,
            'batch_size': 1024,
            'optimizer': 'adam'
        }

    @staticmethod
    def DEFAULT_MODEL_STRUCTURE():
        return {
            'max_sequence': 20,
            'n_hidden': 128,
            'gradient_clip': 5.0,
            'node': 'gru',
            'n_intent_output': 15,
            'n_entities_output': 8
        }

    def __init__(self, model_config, data, restore_path=None):

        super().__init__(model_config, data, restore_path=restore_path)

        def test_model_structure(key, lambda_value):
            if key in self.model_structure:
                return self.model_structure[key]
            else:
                if self.data is None or 'embedding' not in self.data.values:
                    raise ValueError('When creating a new model, data must contain a word embedding')
                self.model_structure[key] = lambda_value()
                return lambda_value()

        self.n_vector = test_model_structure('n_vector', lambda: len(self.data["embedding"].embedding[0]))
        self.word_count = test_model_structure('word_count', lambda: len(self.data["embedding"].words))

        # Tensorflow placeholders
        self.x = tf.placeholder(tf.int32, [None, self.model_structure['max_sequence']])  # [batch size, sequence length]
        self.y_intent = tf.placeholder("float", [None, self.model_structure['n_intent_output']])               # [batch size, intent]
        self.y_entities = tf.placeholder("float", [None, self.model_structure['max_sequence'], self.model_structure['n_entities_output']])
        self.word_embedding = tf.Variable(tf.constant(0.0, shape=(self.word_count, self.n_vector)), trainable=False)

        # Network parameters
        self.weights = {  # LSTM weights are created automatically by tensorflow
            "out_intent": tf.Variable(tf.random_normal([self.model_structure['n_hidden'], self.model_structure['n_intent_output']])),
            "out_entities": tf.Variable(tf.random_normal([self.model_structure['n_hidden'] + self.model_structure['n_intent_output'], self.model_structure['n_entities_output']]))
        }

        self.biases = {
            "out_intent": tf.Variable(tf.random_normal([self.model_structure['n_intent_output']])),
            "out_entities": tf.Variable(tf.random_normal([[self.model_structure['n_entities_output']]]))
        }

        self.cell_fw = tf.contrib.rnn.GRUCell(self.model_structure['n_hidden'])
        self.cell_bw = tf.contrib.rnn.GRUCell(self.model_structure['n_hidden'])

        # Optimization
        logits_intent, logits_ner = self.network()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_intent, labels=self.y_intent)) + \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_ner, labels=self.y_entities))

        optimizer = tf.train.AdamOptimizer(self.hyperparameters['learning_rate'])
        gradients, variables = zip(*optimizer.compute_gradients(self.cost))
        gradients, _ = tf.clip_by_global_norm(gradients, self.model_structure['gradient_clip'])
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        # Tensorflow initialization
        self.sess.run(tf.global_variables_initializer())

        self.data_set = False

    def network(self):

        embedded_x = tf.nn.embedding_lookup(self.word_embedding, self.x)

        # X has shape of x ([batch size, sequence length, input length])
        seqlen = Utils.get_length(self.x)
        batch_size = tf.shape(self.x)[0]

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw,
                                                     self.cell_bw,
                                                     inputs=embedded_x,
                                                     dtype=tf.float32,
                                                     sequence_length=seqlen,
                                                     swap_memory=True)
        # see https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
        # swap_memory is optional.

        outputs_fw, output_bw = outputs  # outputs is a tuple (output_fw, output_bw)

        # get last time steps
        indexes = tf.reshape(tf.range(0, batch_size), [batch_size, 1])
        last_time_steps = tf.reshape(tf.add(seqlen, -1), [tf.range, 1])
        last_time_step_indexes = tf.concat([indexes, last_time_steps], axis=1)

        # apply linear
        outputs_intent = tf.matmul(
            tf.gather_nd(outputs_fw, last_time_step_indexes),
            self.weights["out_intent"]) + self.biases["out_intent"]

        entities = tf.concat(
            [output_bw, tf.tile(tf.expand_dims(outputs_intent, 1), [1, 30, 1])], -1
        )
        outputs_entities = tf.einsum('ijk,kl->ijl', entities, self.weights["out_entities"]) + self.biases["out_entities"]

        return outputs_intent, outputs_entities  # linear/no activation as there will be a softmax layer

    ###############################
    # refactored up to this point #
    ###############################

    def train(self, epochs=200):
        if not self.data_set:
            Utils.printMessage("Error: Training data not set")
            return

        # get data
        input_data, ner_data, intent_data = get_data(self.intents_folder, Utils.wordsToIndex)

        # Start training
        Utils.printMessage("Starting training")
        for epoch in range(epochs + 1):  # since range is exclusive

            mini_batches_X, mini_batches_Y_intent, mini_batches_Y_entities \
                = Utils.random_mini_batches([input_data, intent_data, ner_data], self.batch_size)

            for i in range(0, len(mini_batches_X)):

                batch_x = mini_batches_X[i]
                batch_y_intent = mini_batches_Y_intent[i]
                batch_y_entities = mini_batches_Y_entities[i]

                self.sess.run(self.train_op, feed_dict={self.x: batch_x,
                                                                self.y_intent: batch_y_intent,
                                                                self.y_entities: batch_y_entities})

                if epoch % display_step == 0:
                    cost_value = self.sess.run([self.cost],
                                               feed_dict={self.x: batch_x,
                                                          self.y_intent: batch_y_intent,
                                                          self.y_entities: batch_y_entities})

                    Utils.printMessage(
                        'epoch ' + str(epoch) + ' (' + str(i + 1) + '/' + str(len(mini_batches_X)) + ') - cost' + str(
                            cost_value))

    def predict(self, sentence):

        response_data = sentence_to_vec(Utils.wordsToIndex, sentence.lower().split())

        intent, ner = self.sess.run([tf.argmax(tf.reshape(self.prediction_intent, [self.n_intent_output])),
                                     tf.argmax(tf.reshape(self.prediction_ner, [self.max_sequence,
                                                                                self.n_entities_output]), axis=1)
                                     [:len(str.split(sentence))]],
                                    feed_dict={self.x: np.expand_dims(response_data, 0)})

        return intent, ner

    def predictAll(self, path, savePath=None):
        result = []

        if not os.path.isfile(path):
            Utils.printMessage("Error: path in PredictAll for Intent & NER Networks must be a single file")
            return
        with open(path) as file:
            lines = file.read().splitlines()

        for line in lines:
            result.append(self.predict(line))

        if savePath is not None:
            with open(savePath, "a") as file:
                for i in range(len(lines)):
                    file.write(str(result[i][0]) + " " + str(result[i][1]) + " " + lines[i] + "\n")

        return result
