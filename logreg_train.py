import numpy as np
import pandas as pd
from utils import read_dataset, get_file_path, mean_score, standard_deviation
from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()


def gradient_descent(X, y, learning_rate, num_iters):
    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(num_iters):
        # Compute the difference between predicted values and acutal values
        error = sigmoid(np.dot(X, theta.T)) - y
        # Compute the partial derivate
        gradient = np.dot(X.T, error) / m
        # Update de theta
        theta -= learning_rate * gradient

    return theta


def train_logistic_regression(X, Y, num_classes, learning_rate, num_iters):
    weights = []
    # To do a one vs all approche, loop through each class  
    for i in range(num_classes):
        # Construct a binary list with 1 for our class and 0 for other classes
        y_one_vs_all = (Y == i).astype(int)
        theta = gradient_descent(X, y_one_vs_all, learning_rate, num_iters)
        weights.append(theta)
    return np.array(weights)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def preproccess(dataset: pd.DataFrame, train: bool):
    drop_columns = ["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand", "Arithmancy", "Astronomy", "Care of Magical Creatures"]
    X = dataset
    if train:
        Y = dataset["Hogwarts House"]
        Y = label_encoder.fit_transform(Y)
    else:
        Y = None

    X.drop(columns=drop_columns, inplace=True)
    for course in X:
        X[course] = X[course].transform(lambda x: x.fillna(x.mean()))
        values = X[course]
        X[course] = (values - mean_score(values)) / standard_deviation(values)
  
    X = X.to_numpy()

    return X, Y


def predict(X, weights):
    predictions = sigmoid(np.dot(X, weights.T))
    return np.argmax(predictions, axis=1)


if __name__ == "__main__":
    file_path = get_file_path()
    dataset = read_dataset(file_path)
    X, Y = preproccess(dataset, train=True)

    weights = train_logistic_regression(X, Y, len(np.unique(Y)), 0.01, 1000)
    np.save('weights.npy', weights)

    dataset_test = pd.read_csv("./datasets/dataset_test.csv", index_col="Index")
    x_test, y_test = preproccess(dataset_test, train=False)

    predictions = predict(x_test, weights)
    predictions = label_encoder.inverse_transform(predictions)

    houses = pd.DataFrame({'Index': range(len(predictions)), 'Hogwarts House': predictions})
    houses.to_csv('houses.csv', index=False)
