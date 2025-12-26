import os
import polars as pl
import pandas as pd
import numpy as np
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Datasets
from sklearn import datasets

# My Models
# from adaline_sgd import AdalineSGD

# ------------------------------------
# Constants
# ------------------------------------


# ------------------------------------
# Function to plot decision regions
# ------------------------------------
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')

    # plot class examples
    # for idx, cl in enumerate(np.unique(y)):
    #     plt.scatter(x=X[y == cl, 0],
    #                 y=X[y == cl, 1],
    #                 alpha=0.8,
    #                 c=colors[idx],
    #                 marker=markers[idx],
    #                 label=f'Class {cl}',
    #                 edgecolor='black')

class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier.
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.
    Attributes
    -----------
    w_ : 1d-array
      Weights after training.
    b_ : Scalar
      Bias unit after fitting.
    losses_ : list
      Mean squared error loss function values in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self, X, y):
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the 
          number of examples and n_features is the 
          number of features.
        y : array-like, shape = [n_examples]
          Target values.
        Returns
        -------
        self : Instance of LogisticRegressionGD
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output))
                   - ((1 - y).dot(np.log(1 - output)))
                    / X.shape[0])
            self.losses_.append(loss)
        return self
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def iris_svc():

    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print('Class labels:', np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # svm = SVC(kernel='rbf', random_state=1, gamma=10.0, C=1.0)
    # svm.fit(X_train_std, y_train)

    forest = RandomForestClassifier(n_estimators=25,
                                    random_state=1,
                                    n_jobs=2)
    forest.fit(X_train_std, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=forest,
                          test_idx=range(105, 150))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('results/random_forest.png')


def random_forest():

    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print('Class labels:', np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # svm = SVC(kernel='rbf', random_state=1, gamma=10.0, C=1.0)
    # svm.fit(X_train_std, y_train)

    forest = RandomForestClassifier(n_estimators=25,
                                    random_state=1,
                                    n_jobs=2)
    forest.fit(X_train_std, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=forest,
                          test_idx=range(105, 150))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('results/random_forest.png')




# ------------------------------------
# Main functions
# ------------------------------------

def main():
    # iris_svc()

    csv_data = "A, B, C, D\n1.0, 2.0, 3.0, 4.0\n5.0, 6.0,,8.0\n10.,11.0,12.0,"

    df = pd.read_csv(StringIO(csv_data))
    print(df)

    print(df.isnull().sum())

    # random_forest()
    


    

if __name__ == "__main__":
    main()
