import matplotlib.pyplot as plt
from utils import read_dataset, COURSES, standardization


def get_grades(dataset, house, course):
    house_data = dataset[dataset["Hogwarts House"] == house]
    return house_data[[course]].dropna()


def draw_histogram(dataset):
    # Calculate number of rows and columns for the subplot grid
    num_courses = len(COURSES)
    num_rows = (num_courses + 2) // 3  # Round up to the nearest multiple of 3 for rows
    num_cols = min(3, num_courses)     # Maximum of 3 columns

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows*5))

    # Iterate over each course and corresponding subplot
    for i, course in enumerate(COURSES):
        row = i // num_cols
        col = i % num_cols

        # Plot histogram for each house
        axs[row, col].hist(get_grades(dataset, "Gryffindor", course), bins=50, alpha=0.65, label='Gry', color='r')
        axs[row, col].hist(get_grades(dataset, "Ravenclaw", course), bins=50, alpha=0.65, label='Rav', color='b')
        axs[row, col].hist(get_grades(dataset, "Slytherin", course), bins=50, alpha=0.65, label='Sly', color='g')
        axs[row, col].hist(get_grades(dataset, "Hufflepuff", course), bins=50, alpha=0.65, label='Huf', color='y')

        # Set title and legend
        axs[row, col].set_title(course)
        axs[row, col].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = read_dataset("./datasets/dataset_train.csv")
    standardization(dataset)
    draw_histogram(dataset)
