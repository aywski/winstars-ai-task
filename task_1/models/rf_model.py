from sklearn.ensemble import RandomForestClassifier
from interface import MnistClassifierInterface

class RandomForestModel(MnistClassifierInterface):
    def __init__(self):
        # initializing a RF model with hyperparameters
        self.model = RandomForestClassifier(
            max_depth=None,
            n_estimators=500,
            verbose=1
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
