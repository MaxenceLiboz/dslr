from utils import read_dataset, get_table_data
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="data_file", help="Open datasets/dataset_train.csv file")

args = parser.parse_args()
data_file = args.data_file

def main():

    data = read_dataset(data_file)
    data.columns = data.columns.str.replace(' ', '_')
    table_data = get_table_data(data)


    index_labels=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    data_frame = pd.DataFrame(table_data, index_labels)

    pd.options.display.max_columns = 13 # set the max displayable columns
    print(data_frame)

if __name__ == "__main__":
    main()