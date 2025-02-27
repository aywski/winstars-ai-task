from models.cnn_model import CNNModel
from models.rf_model import RandomForestModel
from models.nn_model import NeuralNetworkModel

class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == 'rf':
            self.model = RandomForestModel()
        elif algorithm == 'nn':
            self.model = NeuralNetworkModel()
        elif algorithm == 'cnn':
            self.model = CNNModel()
        else:
            raise ValueError("Algorithm not recognized")
        
    def train(self, X_train, y_train):
        self.model.train(X_train, y_train)
        
    def predict(self, X_test):
        return self.model.predict(X_test)
