import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------

    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter:int
        Passes over the training dataset

    random_state: int
        Random number generator seed for random weight initialization.

    Attributes
    ------------
    w_:1d-array
        Weights after fitting.
    erroes_ :list
    Number of misclassifications (updates) in each epoch."""

    def __init__(self, eta, n_iter, random_state=1):
        print("init called")
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        print("fit called")
        """"Fit training data.
         Parameters
        ------------
        X: {array-like}, shape = [n_smaples, n_features]
        Training vectors, where n_smaples is the number samples.
        n_features is the number of features.
        y: {array-like}, shape = [n_samples]
        Target values.

        Returns
        -------
        self: object """

        rgen = np.random.RandomState(self.random_state)
        print(rgen, "rgen")
        print("fit self.random_state", self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        print("self.w_", self.w_)
        self.errors_ = []
        print("self.errors", self.errors_)

        for _ in range(self.n_iter):
            print("self.n_iter", self.n_iter)
            print("_", _)
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target-self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        # print("return value from net_input",
        #       np.dot(X, self.w_[1:]) + self.w_[0])
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """return class label after unit step"""
        # print("return value form predict", np.where(
        #     self.net_input(X) >= 0.0, 1, -1))
        return np.where(self.net_input(X) >= 0.0, 1, -1)


df = pd.read_csv('iris.csv')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('updates')
# plt.show()
