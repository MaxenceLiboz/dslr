import matplotlib.pyplot as plt
from utils import read_dataset, standardization, get_file_path


def draw_scatter_plot(dataset):
    plt.scatter(dataset['Astronomy'], dataset['Defense Against the Dark Arts'], label='grade per student')
    plt.title("Similar feature")
    plt.xlabel("Astronomy")
    plt.ylabel("Defense Against the Dark Arts")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    file_path = get_file_path()
    dataset = read_dataset(file_path)
    standardization(dataset)
    draw_scatter_plot(dataset)
