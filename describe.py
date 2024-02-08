from utils import read_dataset, get_table_data, get_file_path
import pandas as pd


def main():
    file_path = get_file_path()
    data = read_dataset(file_path)
    data.columns = data.columns.str.replace(' ', '_')
    table_data = get_table_data(data)


    index_labels=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    data_frame = pd.DataFrame(table_data, index_labels)

    pd.options.display.max_columns = 13 # set the max displayable columns
    print(data_frame)


if __name__ == "__main__":
    main()
