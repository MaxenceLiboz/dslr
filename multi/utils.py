import pandas as pd
from argparse import ArgumentParser

def read_dataset(filename):
    try:
        dataset = pd.read_csv(filename, header=None)

        return dataset
    except Exception as exp:
        print("Don't change the format of the csv file." + exp)
        exit(1)

def get_file_path():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="data_file", help="Open datasets/data.csv file")

    args = parser.parse_args()
    file_path = args.data_file
    if (file_path is None):
        print("Correct format: python3 describe.py -f {file_path}")
        exit(1)
    return file_path

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
