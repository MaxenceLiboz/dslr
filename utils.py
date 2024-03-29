import pandas as pd
from argparse import ArgumentParser

COURSES = ["Arithmancy", "Astronomy", "Herbology",
           "Defense Against the Dark Arts", "Divination",
           "Muggle Studies", "Ancient Runes", "History of Magic",
           "Transfiguration", "Potions", "Care of Magical Creatures",
           "Charms", "Flying"]


def get_file_path():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="data_file", help="Open datasets/dataset_train.csv file")

    args = parser.parse_args()
    file_path = args.data_file
    if (file_path is None):
        print("Correct format: python3 describe.py -f {file_path}")
        exit(1)
    return file_path


def read_dataset(filename):
    try:
        dataset = pd.read_csv(filename, index_col="Index")
        return dataset.dropna()
    except Exception:
        print("Don't change the format of the csv file.")
        exit(1)


def get_table_data(data):

    tableData = {}

    for serie_name, serie in data.items():
        tableData[serie_name] = get_all_fields(serie)

    return tableData


def get_only_numeric_values(data_frame):
    return (data_frame.iloc[:, 6:])  # Select From 7th to end


def get_all_fields(column):

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


def standardization(dataset):
    try:
        for course in COURSES:
            values = dataset[course]
            dataset[course] = (values - mean_score(values)) / standard_deviation(values)
    except Exception as e:
        print("An error happend during the standarization    " + e)
        exit(1)

    return dataset
