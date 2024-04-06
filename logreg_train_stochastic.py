from utils import read_dataset, get_file_path, preproccess, train_logistic_regression_stochastic
import numpy as np


if __name__ == "__main__":
    file_path = get_file_path()
    dataset, _ = read_dataset(file_path, False)
    X, Y = preproccess(dataset, train=True)

    weights = train_logistic_regression_stochastic(X, Y, len(np.unique(Y)), 0.01, 1000)
    np.save('weights.npy', weights)