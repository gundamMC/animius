class ModelConfig:
    def __init__(self, cls, config=None, hyperparameters=None, model_structure=None):
        if config is None:
            self.config = {}
        else:
            self.config = config

        if hyperparameters is None:
            self.hyperparameters = {}
        else:
            self.hyperparameters = hyperparameters

        if model_structure is None:
            self.model_structure = {}
        else:
            self.model_structure = model_structure

        self.apply_defaults(cls)

    def apply_defaults(self, cls):

        for key, default_value in cls.DEFAULT_CONFIG().items():
            if key not in self.config:
                self.config[key] = default_value

        for key, default_value in cls.DEFAULT_HYPERPARAMETERS().items():
            if key not in self.hyperparameters:
                self.hyperparameters[key] = default_value

        for key, default_value in cls.DEFAULT_MODEL_STRUCTURE().items():
            if key not in self.model_structure:
                self.model_structure[key] = default_value
