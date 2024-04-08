#!/usr/bin/python3

import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from utils import read_dataset, get_file_path, standardization


def get_values(dataset, Y, result, index):
    data_value = dataset[Y == result]
    return data_value[[index]]


def draw_histogram(dataset, Y, isRaw):
    # Calculate number of rows and columns for the subplot grid
    m = dataset.shape[1]
    num_rows = (m + 2) // 3  # Round up to the nearest multiple of 3 for rows
    num_cols = min(3, m)     # Maximum of 3 columns

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows*5))

    # Iterate over each course and corresponding subplot
    for index in range(m):
        real_index = index + 2 if isRaw else index + 1
        row = index // num_cols
        col = index % num_cols

        # Plot histogram for each house
        axs[row, col].hist(get_values(dataset, Y, "M", real_index), bins=50, alpha=0.65, label='Gry', color='r')
        axs[row, col].hist(get_values(dataset, Y, "B", real_index), bins=50, alpha=0.65, label='Rav', color='b')

        # Set title and legend
        axs[row, col].set_title(real_index)
        axs[row, col].legend()

    # # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file = get_file_path()
    dataset = read_dataset(file)
    Y = dataset[1 if file == 'datasets/data.csv' else 0]
    dataset = standardization(dataset, [0, 1 if file == 'datasets/data.csv' else 0])
    print(Y.value_counts())
    print(dataset.describe())
    draw_histogram(dataset, Y, True if file == 'datasets/data.csv' else False)
    