import os
import polars as pl
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
# Datasets
from sklearn import datasets

# My Models
from adaline_sgd import AdalineSGD



# ------------------------------------
# Function to plot decision regions
# ------------------------------------
def plot_decision_regions(X, y, classifier, resolution=0.02):
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
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')



# ------------------------------------
# Main functions
# ------------------------------------

def main():

    # Load Iris dataset
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print(f'From URL: {s}')
    df = pd.read_csv(s, header=None, encoding='utf-8')
    print(df.tail())
   

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    X = df.iloc[0:100, [0,2]].values
    
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


    # Train Adaline SGD
    ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada_sgd.fit(X_std, y)


    # Plot Results
    plt.style.use('seaborn-v0_8')
    fig1, ax1 = plt.subplots()
    plot_decision_regions(X_std, y, classifier=ada_sgd)
    ax1.set_title('Adaline - Stochastic gradient descent')
    ax1.set_xlabel('Sepal length [standardized]')
    ax1.set_ylabel('Petal length [standardized]')
    ax1.legend(loc='upper left')
    fig1.tight_layout()
    fig1.savefig(os.path.join('results', 'adaline_sgd_iris1.png'))
    plt.close(fig1)

    plt.style.use('seaborn-v0_8')
    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_,
             marker='o')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Average loss')
    fig2.tight_layout()
    fig2.savefig(os.path.join('results', 'adaline_sgd_iris2.png'))
    plt.close(fig2)


if __name__ == "__main__":
    main()
