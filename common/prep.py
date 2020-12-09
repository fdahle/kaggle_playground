import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype

#print all entries of a dataframe (really all!)
def printAll(pdFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(pdFrame)

#drop columns from dataframe
def dropColumns(pdFrame, columns):

    if not isinstance(columns, list):
        columns=[columns]

    pdFrame = pdFrame.drop(columns=columns)

    return pdFrame

#merge dataframes together
def merge_dataframes(pdFrame1, pdFrame2):
    pdFrame = pd.concat([pdFrame1, pdFrame2], axis=1)
    return pdFrame

def reshape_dataframe(pdFrame, newShape):
    data = pdFrame.values.reshape(newShape)
    return data

#one hot encode columns
def one_hot_encode(pdFrame, columnName):
    encoded = pd.get_dummies(pdFrame[columnName], prefix= columnName)
    pdFrame = dropColumns(pdFrame, [columnName])
    pdFrame = merge_dataframes(pdFrame, encoded)
    return pdFrame

#binary encode columns
def binary_encode(pdFrame, columnName):
    unique_vals = pdFrame[columnName].unique()
    if len(unique_vals) > 2:
        raise Exception("too many values (>2) for binary encoding")

    pdFrame[columnName] = pdFrame[columnName].replace(unique_vals[0], 0)
    pdFrame[columnName] = pdFrame[columnName].replace(unique_vals[1], 1)
    return pdFrame

#convert boolean to numeric
def bool_to_num(pdFrame, columnName):
    pdFrame[columnName] = pdFrame[columnName]*1
    return pdFrame

#delete, split and append the splitted columns, keep describes if only one column
#should be kept (all if -1) and title the new column name
def split_column(pdFrame, columnName, delimiter, keep=-1, title=""):
    splitted = pdFrame[columnName].str.split(delimiter, expand=True)

    #keep all columns
    if keep == -1:
        columnNames = []
        for i in range(splitted.shape[1]):
            columnNames.append(columnName + "_" + str(i + 1))
        splitted.columns = columnNames
    else:
        splitted = splitted[keep].to_frame()
        if title == "":
            splitted.columns = [columnName]
        else:
            splitted.columns = [title]

    pdFrame = dropColumns(pdFrame, [columnName])

    pdFrame = merge_dataframes(pdFrame, splitted)
    return pdFrame

#returns True if row has a value in a column
#entries can be str or list (to search for multiple entries at once)
def has_Entry(pdFrame, columnName, entries, delete=False):

    #if only one entry is given convert it to a lists
    if isinstance(entries, str):
        entries = [entries]

    for entry in entries:
        contains = pdFrame[columnName].str.contains(entry).to_frame()
        contains.columns = ["has_" + entry]
        contains = bool_to_num(contains, "has_" + entry)
        pdFrame = merge_dataframes(pdFrame, contains)

    if delete:
        pdFrame = dropColumns(pdFrame, [columnName])

    return pdFrame

#standardize values
def calc_scale(pdFrame, columns=[]):

    scaler = StandardScaler()

    if len(columns) == 0:
        columns = pdFrame.columns

    for elem in columns:
        pdFrame[elem] = pdFrame[elem].astype('float64')
        pdFrame[elem] = scaler.fit_transform(pdFrame[elem].values.reshape(-1, 1))

    return pdFrame

#convert to binary based on threshold
def calc_threshold(input, threshold):

    if torch.is_tensor(input):
        data = input.detach().numpy()

    data[data >= threshold] = 1
    data[data < threshold] = 0

    return data

def handle_nan(pdFrame, type="auto", val=""):

    if type == "auto":

        #save columns that should be dropped
        to_drop = []
        for column in pdFrame:

            #check for NaN
            if pdFrame[column].isnull().values.any():

                sum_nan = pdFrame[column].isnull().values.sum()
                percentage_nan = sum_nan / pdFrame[column].shape[0] * 100

                #take average if below 50% are nan values
                if percentage_nan < 50:
                    pdFrame[column] = handle_nan(pdFrame[column].to_frame(), "avg")

                #otherwise drop the column
                else:
                    to_drop.append(column)

        #drop columns with Nan
        pdFrame = dropColumns(pdFrame, to_drop)

    #asap a nan value is in the column drop the column
    elif type == "drop":

        #save columns that should be dropped
        to_drop = []

        #iterate all columns
        for column in pdFrame:

            #check for NaN
            if pdFrame[column].isnull().values.any():
                to_drop.append(column)

        #drop columns with Nan
        pdFrame = dropColumns(pdFrame, to_drop)

    #replace value with avg values
    #or for nun numeric values the one that can be found most in data
    elif type == "avg":

        #iterate all columns
        for column in pdFrame:

            #check for NaN
            if pdFrame[column].isnull().values.any():

                #check type of column
                if is_numeric_dtype(pdFrame[column]):

                    #check if it may be boolean
                    uniqueList = pdFrame[column].unique().tolist()
                    if (uniqueList == [0,1] or uniqueList == [1,0]):
                        most_common_val = pdFrame[column].mode()[0]
                        pdFrame[column] = handle_nan(pdFrame[column].to_frame(), "val", val=most_common_val)

                    #here we can calulate the avg
                    else:
                        avg_val = pdFrame[column].mean()
                        pdFrame[column] = handle_nan(pdFrame[column].to_frame(), "val", val=avg_val)

                #objects, so take most common value
                else:
                    most_common_val = pdFrame[column].mode()[0]
                    pdFrame[column] = handle_nan(pdFrame[column].to_frame(), "val", val=most_common_val)

    #replace Nan values with a certain value
    elif type == "val":
        if val == "":
            raise Exception("value cannot be empty")

        pdFrame = pdFrame.fillna(val)

    return pdFrame
