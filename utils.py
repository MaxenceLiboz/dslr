import pandas as pd

COURSES = ["Arithmancy", "Astronomy", "Herbology",
           "Defense Against the Dark Arts", "Divination",
           "Muggle Studies", "Ancient Runes", "History of Magic",
           "Transfiguration", "Potions", "Care of Magical Creatures",
           "Charms", "Flying"]


def read_dataset(filename):
    try:
        dataset = pd.read_csv(filename, index_col="Index")
        return dataset.dropna()
    except Exception:
        print("Don't change the format of the csv file.")
        exit(1)


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
    except Exception:
        print("An error happend during the standarization")
        exit(1)

    return dataset