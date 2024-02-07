from utils import read_dataset, get_table_data
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="myFile", help="Open datasets/dataset_train.csv file")

args = parser.parse_args()
myFile = args.myFile

def main():

    data = read_dataset(myFile)
    data.columns = data.columns.str.replace(' ', '_')
    tableData = get_table_data(data)
    print(myFile)


    indexLabels=['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    dataFrame = pd.DataFrame(tableData, indexLabels)

    pd.options.display.max_columns = 13 # set the max displayable columns
    print(dataFrame)

if __name__ == "__main__":
    main()