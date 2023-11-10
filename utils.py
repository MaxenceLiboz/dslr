import numpy as np
import pandas as pd

def get_count(feature):
    return len(feature)

def get_mean(feature):
    return sum(feature) / len(feature)

def get_std(feature):
    mean = get_mean(feature)
    variance = sum((x - mean) ** 2 for x in feature) / len(feature)
    return np.sqrt(variance)

def get_min(feature):
    minimum = feature[0]
    for item in feature[1:]:
        if (item < minimum):
            minimum = item  
    return minimum

def get_percentile(feature, percentile):
    sorted_feature = sorted(feature)
    index = int(percentile * len(sorted_feature))
    return sorted_feature[index]

def get_25th_percentile(feature):
    return get_percentile(feature, 0.25)

def get_50th_percentile(feature):
    return get_percentile(feature, 0.5)

def get_75th_percentile(feature):
    return get_percentile(feature, 0.75)

def get_max(feature):
    maximum = feature[0]
    for item in feature[1:]:
        if (item > maximum):
            maximum = item  
    return maximum

def read_dataset(filename):
    try:
        dataset = pd.read_csv(filename)   
        return dataset
    except:
        print("Don't change the format of the csv file.")
        exit(1) 
        

def ftDescribe(dataset):
    indexList = ["", "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Flying"]
    list = []
    list.append(indexList)
    list.append([get_count(dataset[index]) for index in indexList[1:]])
    
    formatPrintAsTable(list)
    
    # result = {
    #     'count': get_count(feature),
    #     'mean': get_mean(feature),
    #     'std': get_std(feature),
    #     'min': get_min(feature),
    #     '25%': get_25th_percentile(feature),
    #     '50%': get_50th_percentile(feature),
    #     '75%': get_75th_percentile(feature),
    #     'max': get_max(feature)
    # }
    
def formatPrintAsTable(data):
    # Find the maximum width of each column
    column_widths = [max(len(str(item)) for item in col) for col in zip(*data)]

    # Print the table
    for row in data:
        row_str = "|".join(f"{item: <{width}}" for item, width in zip(row, column_widths))
        print(row_str)