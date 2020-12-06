import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam

import sys
sys.path.insert(0, '../common')
from prep import *

#load data
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#get ids of passengers
Y_train = data_train['Survived']

#drop columns
data_train = data_train.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Survived'])
data_test = data_test.drop(columns=['PassengerId', 'Ticket', 'Cabin'])

#rename
X_train = data_train
X_test = data_test

#encode values
X_train = binary_encode(X_train, "Sex")
X_train = one_hot_encode(X_train, "Embarked")
X_test = binary_encode(X_test, "Sex")
X_test = one_hot_encode(X_test, "Embarked")

#extract the title from the names
X_train = split_column(X_train, "Name", ",", keep=1, title="Title")
X_train = split_column(X_train, "Title", " ", keep=1)
X_train = has_Entry(X_train, "Title", ["Dr", "Rev"], delete=True)
X_test = split_column(X_test, "Name", ",", keep=1, title="Title")
X_test = split_column(X_test, "Title", " ", keep=1)
X_test = has_Entry(X_test, "Title", ["Dr", "Rev"], delete=True)

#scale values

X_train = calc_scale(X_train, columns=["Pclass", "Age", "SibSp", "Parch", "Fare"])
X_test = calc_scale(X_test, columns=["Pclass", "Age", "SibSp", "Parch", "Fare"])

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

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

        self.linear1 = nn.Linear(11, 8)
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


test = NN(X_test)
test = calc_threshold(test, 0.5)
print(test)
