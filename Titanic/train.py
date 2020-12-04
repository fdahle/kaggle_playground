import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam

#load data
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#get ids of passengers
Y_train = data_train['Survived']

#drop columns
data_train = data_train.drop(columns=['PassengerId','Survived', 'Ticket', 'Cabin'])
data_test = data_test.drop(columns=['PassengerId', 'Ticket', 'Cabin'])

X_train = data_train
X_test = data_test

#one hot encode some columns
def one_hot_encode(pdFrame, columnName):
    encoded = pd.get_dummies(pdFrame[columnName], prefix= columnName)
    pdFrame = pdFrame.drop(columns=[columnName])
    pdFrame = merge_dataframes(pdFrame, encoded)
    return pdFrame

def binary_encode(pdFrame, columnName):
    unique_vals = pdFrame[columnName].unique()
    if len(unique_vals) > 2:
        raise Exception("too many values (>2) for binary encoding")

    pdFrame[columnName] = pdFrame[columnName].replace(unique_vals[0], 0)
    pdFrame[columnName] = pdFrame[columnName].replace(unique_vals[1], 1)
    return pdFrame

def bool_to_num(pdFrame, columnName):
    pdFrame[columnName] = pdFrame[columnName]*1
    return pdFrame

def split_column(pdFrame, delimiter):
    splitted = pdFrame.str.split(delimiter, expand=True)
    return splitted

def merge_dataframes(pdFrame1, pdFrame2):
    pdFrame = pd.concat([pdFrame1, pdFrame2], axis=1)
    return pdFrame

def printAll(pdFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(pdFrame)


X_train = binary_encode(X_train, "Sex")
X_train = one_hot_encode(X_train, "Embarked")

#extract the title from the names
#titles = split_column(X_train["Name"], ",")
#titles = titles[1].to_frame()
#titles.columns = ["Title"]
#X_train = merge_dataframes(X_train, titles)
X_train = X_train.drop(columns=['Name'])
#printAll(X_train)
#exit()
#X_train = one_hot_encode(X_train, "Title")


#print(X_train)
#exit()

X_test = binary_encode(X_test, "Sex")
X_test = one_hot_encode(X_test, "Embarked")

X_test = X_test.drop(columns=['Name'])


print(X_train)

X_train = X_train.fillna(-1)
X_test = X_test.fillna(-1)

X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)
Y_train = torch.Tensor(Y_train.values)

def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)

def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

class NN(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(9, 8)
        self.linear2 = nn.Linear(8,4)
        self.linear3= nn.Linear(4,1)

        self.leaky = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky(x)
        x = self.linear2(x)
        x = self.leaky(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x

NN = NN()
criterion = nn.BCELoss()
optm = Adam(NN.parameters(), lr = 0.001)

epochs = 20000
for epoch in range(epochs):
    y_pred = NN(X_train)
    y_pred = torch.squeeze(y_pred)

    train_loss = criterion(y_pred, Y_train)
    train_acc = calculate_accuracy(Y_train, y_pred)

    optm.zero_grad()

    train_loss.backward()

    optm.step()

    if epoch % 100 == 0:
        print(
            f'''epoch {epoch}
            Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
            ''')
