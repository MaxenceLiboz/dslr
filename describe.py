from utils import read_dataset, get_table_data, get_file_path, standardization
import pandas as pd


def describe(standardizedData):

    index_labels=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    data_frame = pd.DataFrame(standardizedData, index_labels)

    pd.options.display.max_columns = 13 # set the max displayable columns
    print(data_frame)


if __name__ == "__main__":
    file_path = get_file_path()
    dataset = read_dataset(file_path)
    dataset.columns = dataset.columns.str.replace(' ', '_')
    standardizedData = get_table_data(dataset)
    describe(standardizedData)
