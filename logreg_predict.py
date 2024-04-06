from utils import read_dataset, get_file_path, preproccess, predict, decode
import pandas as pd
import numpy as np

if __name__ == "__main__":
    file_path = get_file_path()
    dataset_test, _ = read_dataset(file_path, False)
    wieghts = weights = np.load('./weights.npy')
    x_test, _ = preproccess(dataset_test, train=False)

    predictions = predict(x_test, weights)
    predictions = decode(predictions)

    houses = pd.DataFrame({'Index': range(len(predictions)), 'Hogwarts House': predictions})
    houses.to_csv('houses.csv', index=False)
