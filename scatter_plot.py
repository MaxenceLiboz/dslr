import matplotlib.pyplot as plt
from utils import read_dataset, standardization


def draw_scatter_plot(dataset):
    plt.scatter(dataset['Astronomy'], dataset['Defense Against the Dark Arts'], label='grade per student')
    plt.title("Similar feature")
    plt.xlabel("Astronomy")
    plt.ylabel("Defense Against the Dark Arts")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dataset = read_dataset("./datasets/dataset_train.csv")
    standardization(dataset)
    draw_scatter_plot(dataset)
