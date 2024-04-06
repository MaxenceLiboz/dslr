import pandas as pd
import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import LabelEncoder

COURSES = ["Arithmancy", "Astronomy", "Herbology",
           "Defense Against the Dark Arts", "Divination",
           "Muggle Studies", "Ancient Runes", "History of Magic",
           "Transfiguration", "Potions", "Care of Magical Creatures",
           "Charms", "Flying"]


label_encoder = LabelEncoder()




def get_file_path():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="data_file", help="Open datasets/dataset_train.csv file")

    args = parser.parse_args()
    file_path = args.data_file
    if (file_path is None):
        print("Correct format: python3 describe.py -f {file_path}")
        exit(1)
    return file_path


def read_dataset(filename, do_drop_na):
    try:
        dataset = pd.read_csv(filename, index_col="Index")

        drop_count = []
        for column in dataset.columns[5:]:
            drop_count.append(dataset[column].isna().sum())

        return (dataset.dropna(), drop_count) if do_drop_na else (dataset, None)
    except Exception:
        print("Don't change the format of the csv file.")
        exit(1)


def get_table_data(data, drop_count):

    tableData = {}

    for index, serie_name in enumerate(data):
        tableData[serie_name] = get_all_fields(data.get(serie_name), drop_count[index])

    return tableData


def get_all_fields(column, drop_count):

    feature_column = []

    column_sort = column.tolist()
    column_sort.sort()

    feature_column.append(get_count(column_sort))
    feature_column.append(get_mean(column_sort))
    feature_column.append(get_std(column_sort))
    feature_column.append(get_min(column_sort))
    feature_column.append(get_25_percentile(column_sort))
    feature_column.append(get_50_percentile(column_sort))
    feature_column.append(get_75_percentile(column_sort))
    feature_column.append(get_max(column_sort))
    feature_column.append(drop_count)

    return feature_column


def get_count(column):

    return len(column)


def get_mean(column):
    try:
        mean_column = sum(column) / get_count(column)
    except Exception:
        print("Don't change the format of the csv file.")
        exit(1)

    return mean_column


def get_std(column):
    try:
        n = get_count(column) - 1
        total = 0

        for i in range(n):
            total += (column[i] - get_mean(column)) * (column[i] - get_mean(column))

        std = total / n

    except Exception:
        print("Don't change the format of the csv file.")
        exit(1)

    return std


def get_min(column):

    min = column[0]
    length = len(column)

    for i in range(length):
        if column[i] < min:
            min = column[i]

    return min


def get_25_percentile(column):

    percent = 0.25
    index_percentile = int(percent * get_count(column))

    return column[index_percentile]


def get_50_percentile(column):

    percent = 0.5
    index_percentile = int(percent * get_count(column))

    return column[index_percentile]


def get_75_percentile(column):

    percent = 0.75
    index_percentile = int(percent * get_count(column))
    
    return column[index_percentile]


def get_max(column):

    max = column[0]
    length = len(column)

    for i in range(length):
        if column[i] > max:
            max = column[i]

    return max


def standard_deviation(scores):
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / n
    return variance ** 0.5


def mean_score(scores):
    return sum(scores) / len(scores)


def standardization(dataset, drop_columns):
    if (drop_columns):
        dataset.drop(columns=drop_columns, inplace=True) 

    try:
        for course in dataset:
            if (dataset[course].dtypes != 'float64'):
                continue
            dataset[course] = dataset[course].transform(lambda x: x.fillna(x.mean()))
            values = dataset[course]
            dataset[course] = (values - mean_score(values)) / standard_deviation(values)
    except Exception as exp:
        print(f"An error happend during the standarization: {exp}")
        exit(1)

    return dataset


def predict(X, weights):
    predictions = sigmoid(np.dot(X, weights.T))
    return np.argmax(predictions, axis=1)


def gradient_descent(X, y, learning_rate, num_iters):
    m, n = X.shape
    theta = np.zeros(n)

    for i in range(num_iters):
        # Compute the difference between predicted values and acutal values
        error = sigmoid(np.dot(X, theta.T)) - y
        # Compute the partial derivate
        gradient = np.dot(X.T, error) / m
        # Update de theta
        theta -= learning_rate * gradient

    return theta

def stochastic_gradient_descent(X, y, learning_rate, num_iters):
    m, n = X.shape
    theta = np.zeros(n)

    num_iter = 0
    while (num_iter <= num_iters):

        for j in range(n):

            for i in range(m):

                # Compute the difference between predicted values and acutal values
                error = sigmoid(np.dot(X[i], theta.T)) - y[i]
                # Compute the partial derivate
                gradient = np.dot(X[i].T, error) / m
                # Update de theta
                theta -= learning_rate * gradient
        
        num_iter += n

    return theta


def train_logistic_regression_stochastic(X, Y, num_classes, learning_rate, num_iters):
    weights = []
    # To do a one vs all approche, loop through each class  
    for i in range(num_classes):
        # Construct a binary list with 1 for our class and 0 for other classes
        y_one_vs_all = (Y == i).astype(int)
        theta = stochastic_gradient_descent(X, y_one_vs_all, learning_rate, num_iters)
        weights.append(theta)

    return np.array(weights)
    

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
        Y = X["Hogwarts House"]
        Y = encode(Y)
    else:
        Y = None

    X = standardization(X, drop_columns) 

    X = X.to_numpy()

    return X, Y

def encode(housesList):
    houses = {"Ravenclaw": 0, "Slytherin": 1, "Gryffindor": 2, "Hufflepuff": 3}
    return np.array([houses.get(house) for house in housesList])

def decode(indexList):
    houses = {0: "Ravenclaw", 1: "Slytherin", 2: "Gryffindor", 3: "Hufflepuff"}
    return np.array([houses.get(index) for index in indexList])

def calcCost(X, Y, weights):
    sum = 0
    prediction = predict(X, weights)
    sum = (prediction - Y) ** 2

    cost = sum / (2 * len(Y))
    return cost