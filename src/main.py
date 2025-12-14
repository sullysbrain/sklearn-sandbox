import os
import polars as pl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():

    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print(f'From URL: {s}')
    # df = pl.read_csv(s, has_header=False, encoding='utf-8')
    # df = df.filter(pl.any_horizontal(pl.all().is_not_null()))
    df = pd.read_csv(s, header=None, encoding='utf-8')
    print(df.tail())

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    X = df.iloc[0:100, [0,2]].values
    
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')

    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    # plt.show()
    plt.savefig("results/plot.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
