import matplotlib.pyplot as plt
from utils import read_dataset, standardization, get_file_path
import seaborn as sb


if __name__ == "__main__":
    file_path = get_file_path()
    dataset = read_dataset(file_path)
    standardization(dataset)
    sb.pairplot(dataset, hue='Hogwarts House')
    plt.show()
