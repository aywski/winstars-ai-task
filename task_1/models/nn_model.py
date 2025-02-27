from sklearn.neural_network import MLPClassifier
from interface import MnistClassifierInterface

class NeuralNetworkModel(MnistClassifierInterface):
    def __init__(self):
        # initializing a MLP model with hyperparameters
        self.model = MLPClassifier(
            hidden_layer_sizes=(128,), # number of neurons in the first layer
            early_stopping = True,
            max_iter = 1000,
            verbose=True    
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
