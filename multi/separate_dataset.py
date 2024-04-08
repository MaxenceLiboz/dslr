#!/usr/bin/python3

from utils import read_dataset, get_file_path
import pandas as pd

def create_separate_dataset(dataset, training_percentage):
    # Les column drop sont trouvé grace au program describe.py
    dataset.drop(columns=[0, 6, 11, 13, 16, 20], inplace=True)
    dataset_maligant = dataset[dataset[1] == 'M']
    dataset_benign = dataset[dataset[1] == 'B']

    print(dataset_maligant.shape)
    print(dataset_benign.shape)

    dataset_maligant_training = dataset_maligant.iloc[:int(training_percentage * dataset_maligant.shape[0])]
    dataset_benign_training = dataset_benign.iloc[:int(training_percentage * dataset_benign.shape[0])]
    dataset_training = pd.concat([dataset_maligant_training, dataset_benign_training])
    dataset_training = dataset_training.sample(frac=1, random_state=1)

    # The rest of the set but without the first column (start:end(row), start:end(column))
    dataset_maligant_testing = dataset_maligant.iloc[int(training_percentage * dataset_maligant.shape[0]):, 1:]
    dataset_benign_testing = dataset_benign.iloc[int(training_percentage * dataset_benign.shape[0]):, 1:]

    dataset_testing = pd.concat([dataset_maligant_testing, dataset_benign_testing])
    dataset_testing = dataset_testing.sample(frac=1, random_state=1)

    return dataset_training, dataset_testing


if __name__ == "__main__":
    print("Separate dataset in two parts: training and testing")
    
    file_path = get_file_path()
    dataset = read_dataset(file_path)

    # The percentage of the dataset that will be used for training
    training_percentage = 0.8
    dataset_training, dataset_testing = create_separate_dataset(dataset, training_percentage)

    pd.DataFrame.to_csv(dataset_training, 'datasets/dataset_train.csv', index=False, header=False)
    pd.DataFrame.to_csv(dataset_testing, 'datasets/dataset_test.csv', index=False, header=False)