from utils import read_dataset, get_file_path, preproccess, train_logistic_regression
import numpy as np


if __name__ == "__main__":
    file_path = get_file_path()
    dataset = read_dataset(file_path, False)
    X, Y = preproccess(dataset, train=True)

    weights = train_logistic_regression(X, Y, len(np.unique(Y)), 0.01, 1000)
    np.save('weights.npy', weights)