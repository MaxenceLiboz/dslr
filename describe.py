from utils import get_file_path, read_dataset, standardization, get_table_data
import pandas as pd


def describe(standardizedData):

    index_labels=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    table_data = get_table_data(standardizedData)
    data_frame = pd.DataFrame(table_data, index_labels)

    pd.options.display.max_columns = 13 # set the max displayable columns
    print(data_frame)


if __name__ == "__main__":
    file_path = get_file_path()
    dataset = read_dataset(file_path)
    standardization(dataset)
    describe(dataset)
