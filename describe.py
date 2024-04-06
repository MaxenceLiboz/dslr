from utils import get_file_path, read_dataset, standardization, get_table_data, COURSES
import pandas as pd


def describe(standardized_data, drop_count):

    index_labels=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', "NaN Count"]
    table_data = get_table_data(standardized_data, drop_count)
    data_frame = pd.DataFrame(table_data, index_labels)

    pd.options.display.max_columns = 13 # set the max displayable columns
    print(data_frame)


if __name__ == "__main__":
    file_path = get_file_path()
    dataset, drop_count = read_dataset(file_path, True)
    drop_columns = ["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"]
    standardization(dataset, drop_columns)
    describe(dataset, drop_count)
