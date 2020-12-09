import numpy as np

import sys
sys.path.insert(0, '../../common')
from prep import *
from plot import *
from load_save import *
from train import *

#load_data
data_train = load_data("train.csv")
data_test = load_data("test.csv")

Y_train = data_train["label"]

data_train = dropColumns(data_train, "label")

X_train = data_train
X_train = reshape_dataframe(X_train,[-1,28,28])
X_train = np.expand_dims(X_train, axis=1)

class NN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.drop = nn.Dropout2d()
        self.linear1 = nn.Linear(320, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3= nn.Linear(128,10)
        self.leaky = nn.LeakyReLU()
        self.softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.pool(x)
        x = x.view(-1, 320)
        x = self.linear1(x)
        x = self.leaky(x)
        x = self.linear2(x)
        x = self.leaky(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x

NN = NN()

params = {}
params["n_epochs"] = 3
params["optimizer"] = "Adam"
params["criterion"] = "CrossEntropy"
params["print_epochs"] = 1
params["print_batches"] = 10
params["num_workers"] = 0

NN = train_NN(NN, X_train, Y_train, params, device="gpu")
