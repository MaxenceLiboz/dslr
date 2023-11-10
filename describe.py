from utils import *

def main():
    dataset = read_dataset("datasets/dataset_train.csv")
    for i in dataset:
        print(i)
    print(dataset)
    # print(dataset.isnull().sum())
    print(dataset.dtypes)
    print(dataset.describe())
    ftDescribe(dataset)

    return

if __name__=="__main__":
    main()