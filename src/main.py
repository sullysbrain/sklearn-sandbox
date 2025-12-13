import polars as pl
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():

    # Load the Iris dataset
    iris = load_iris()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

    # Create a logistic regression model
    model = LogisticRegression(max_iter=20)

    # Train the model using the training set
    model.fit(X_train, y_train)

    # Evaluate the model using the testing set
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")



if __name__ == "__main__":
    main()
