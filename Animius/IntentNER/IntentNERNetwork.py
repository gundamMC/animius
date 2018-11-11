import tensorflow as tf
import Animius as PW


class IntentNERModel(PW.Model):

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
            'n_ner_output': 8
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
        self.x_length = tf.placeholder(tf.int32, [None])
        self.y_intent = tf.placeholder("float", [None, self.model_structure['n_intent_output']])               # [batch size, intent]
        self.y_ner = tf.placeholder("float", [None, self.model_structure['max_sequence'], self.model_structure['n_ner_output']])
        self.word_embedding = tf.Variable(tf.constant(0.0, shape=(self.word_count, self.n_vector)), trainable=False)

        # Network parameters
        self.weights = {  # LSTM weights are created automatically by tensorflow
            "out_intent": tf.Variable(tf.random_normal([self.model_structure['n_hidden'], self.model_structure['n_intent_output']])),
            "out_ner": tf.Variable(tf.random_normal([self.model_structure['n_hidden'] + self.model_structure['n_intent_output'], self.model_structure['n_ner_output']]))
        }

        self.biases = {
            "out_intent": tf.Variable(tf.random_normal([self.model_structure['n_intent_output']])),
            "out_ner": tf.Variable(tf.random_normal([self.model_structure['n_ner_output']]))
        }

        self.cell_fw = tf.contrib.rnn.GRUCell(self.model_structure['n_hidden'])
        self.cell_bw = tf.contrib.rnn.GRUCell(self.model_structure['n_hidden'])

        # Optimization
        logits_intent, logits_ner = self.network()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_intent, labels=self.y_intent)) + \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_ner, labels=self.y_ner))

        optimizer = tf.train.AdamOptimizer(self.hyperparameters['learning_rate'])
        gradients, variables = zip(*optimizer.compute_gradients(self.cost))
        gradients, _ = tf.clip_by_global_norm(gradients, self.model_structure['gradient_clip'])
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        self.prediction = tf.nn.softmax(logits_intent), tf.nn.softmax(logits_ner)

        # Tensorboard
        if self.config['tensorboard'] is not None:
            tf.summary.scalar('cost', self.cost)
            self.merged = tf.summary.merge_all()

        self.init_tensorflow()

        self.init_hyerdash(self.config['hyperdash'])

        # restore model data values
        self.init_restore(restore_path, self.word_embedding)

    def network(self):

        embedded_x = tf.nn.embedding_lookup(self.word_embedding, self.x)
        batch_size = tf.shape(self.x)[0]

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.cell_fw,
                                                     self.cell_bw,
                                                     inputs=embedded_x,
                                                     dtype=tf.float32,
                                                     sequence_length=self.x_length,
                                                     swap_memory=True)
        # see https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn
        # swap_memory is optional.

        outputs_fw, output_bw = outputs  # outputs is a tuple (output_fw, output_bw)

        # get last time steps
        indexes = tf.reshape(tf.range(0, batch_size), [batch_size, 1])
        last_time_steps = tf.reshape(tf.add(self.x_length, -1), [batch_size, 1])
        last_time_step_indexes = tf.concat([indexes, last_time_steps], axis=1)

        # apply linear
        outputs_intent = tf.matmul(
            tf.gather_nd(outputs_fw, last_time_step_indexes),
            self.weights["out_intent"]) + self.biases["out_intent"]

        entities = tf.concat(
            [output_bw, tf.tile(tf.expand_dims(outputs_intent, 1), [1, self.model_structure['max_sequence'], 1])], -1
        )
        outputs_entities = tf.einsum('ijk,kl->ijl', entities, self.weights["out_ner"]) + self.biases["out_ner"]

        return outputs_intent, outputs_entities  # linear/no activation as there will be a softmax layer

    def train(self, epochs=200):

        for epoch in range(epochs):

            mini_batches_x, mini_batches_x_length, mini_batches_y_intent, mini_batches_y_ner \
                = PW.Utils.get_mini_batches(
                    PW.Utils.shuffle([
                        self.data['x'],
                        self.data['x_length'],
                        self.data['y_intent'],
                        self.data['y_ner']
                    ]),
                    self.hyperparameters['batch_size'])

            for batch in range(len(mini_batches_x)):

                batch_x = mini_batches_x[batch]
                batch_x_length = mini_batches_x_length[batch]
                batch_y_intent = mini_batches_y_intent[batch]
                batch_y_ner = mini_batches_y_ner[batch]

                if (self.config['display_step'] == 0 or
                    self.config['epoch'] % self.config['display_step'] == 0 or
                    epoch == epochs) and \
                        (batch % 100 == 0 or batch == 0):
                    _, cost_value = self.sess.run([self.train_op, self.cost], feed_dict={
                        self.x: batch_x,
                        self.x_length: batch_x_length,
                        self.y_intent: batch_y_intent,
                        self.y_ner: batch_y_ner
                    })

                    print("epoch:", self.config['epoch'], "- (", batch, "/", len(mini_batches_x), ") -", cost_value)

                    if self.config['hyperdash'] is not None:
                        self.hyperdash.metric("cost", cost_value)

                else:
                    self.sess.run([self.train_op], feed_dict={
                        self.x: batch_x,
                        self.x_length: batch_x_length,
                        self.y_intent: batch_y_intent,
                        self.y_ner: batch_y_ner
                    })

            if self.config['tensorboard'] is not None:
                summary = self.sess.run(self.merged, feed_dict={
                    self.x: mini_batches_x[0],
                    self.x_length: mini_batches_x_length[0],
                    self.y_intent: mini_batches_y_intent[0],
                    self.y_ner: mini_batches_y_ner[0]
                })
                self.tensorboard_writer.add_summary(summary, self.config['epoch'])

            self.config['epoch'] += 1

    def predict(self, input_data, save_path=None):

        intent, ner = self.sess.run(self.prediction,
                                    feed_dict={
                                        self.x: input_data.values['x'],
                                        self.x_length: input_data.values['x_length']
                                    })

        ner = [ner[i, :int(input_data.values['x_length'][i])] for i in range(len(ner))]

        if save_path is not None:
            with open(save_path, "w") as file:
                for i in range(len(intent)):
                    file.write(str(intent[i]) + ' - ' + ', '.join(str(x) for x in ner[i]) + '\n')

        return intent, ner
