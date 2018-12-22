import animius as am


class Waifu:

    def __init__(self, models, word_embedding):

        if 'CombinedPrediction' not in models:
            raise ValueError('A CombinedPrediction model is required')

        self.combined_prediction = models['CombinedPrediction']

        self.input_data = am.ModelData.CombinedPredictionData(models['CombinedPrediction'].model_config)
        self.input_data.add_embedding_class(word_embedding)

    def predict(self, sentence):

        self.input_data.set_parse_input(sentence)

        return self.combined_prediction.predict(self.input_data)
