import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

def train_NN(NN, x, y, params, device="auto"):

    print("TODO - ADD PREMATURE ENDING IF LOSS NOT DECREASING")

    #set backup learning_rate
    if 'learning_rate' not in params:
        print("No learning rate was specified. Default value is set to 0.01")
        params["learning_rate"] = 0.01

    # TODO: fix this criterion
    if 'criterion' not in params:
        print("No loss criterion was specified. Default value is set to Cross Entropy")
        params["criterion"] = nn.CrossEntropyLoss()

    if 'optimizer' not in params:
        params["optimizer"] = None

    if 'eval_params' not in params:
        params["eval_params"] = []

    if 'print_epochs' not in params:
        params["print_epochs"] = None

    if 'print_batches' not in params:
        params["print_batches"] = None

    if 'validation_split' not in params:
        params["validation_split"] = 0

    if 'seed' not in params:
        params["seed"] = 123

    if 'num_workers' not in params:
        params["num_workers"] = 0

    if 'batch_size' not in params:
        print("No batch_size was specified. Default value is set to 64")
        params["batch_size"] = 64

    if 'shuffle' not in params:
        params["shuffle"] = True

    #check the logical consistency of the params
    check_params_consistency(params)

    #extract variables from params dict
    n_epochs = params["n_epochs"]
    lr = params["learning_rate"]
    criterion = params["criterion"]
    eval_params = params["eval_params"]
    optm = params["optimizer"].lower()
    print_epochs = params["print_epochs"]
    print_batches = params["print_batches"]
    val_split = params["validation_split"]
    seed = params["seed"]


    #set the device
    device = setDevice(device)
    if device == "cuda:0":
        NN.cuda()

    #init optimizer if wished
    if (optm == "adam"):
        optm = Adam(NN.parameters(), lr=lr)

    if (criterion == "BCE"):
        criterion = nn.BCELoss()
    elif (criterion == "CrossEntropy"):
        criterion = nn.CrossEntropyLoss()

    #convert inputData to tensors
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        x = x.float()
    elif isinstance(x, pd.DataFrame) or isinstance(y, pd.Series):
        x = torch.FloatTensor(x.values)
    else:
        raise Exception("this format is not yet supported for conversion to Tensor")

    #convert inputData to tensors
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    elif isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = torch.tensor(y.values, dtype=torch.long)
    else:
        raise Exception("this format is not yet supported for conversion to Tensor")

    #convert data to dataset
    dataset = TensorDataset(x, y)

    #make split between validation and train set
    if val_split > 0:
        train_dataset, val_dataset = random_split(100 - val_split, val_split)
    elif val_split == 0:
        train_dataset = dataset

    #create generator
    gen_params = {'batch_size': params["batch_size"],
                  'shuffle': params["shuffle"],
                  'num_workers': params["num_workers"]}
    train_generator = torch.utils.data.DataLoader(train_dataset, **gen_params)
    if val_split > 0:
        val_generator = torch.utils.data.DataLoader(val_dataset, **gen_params)


    train_losses = []


    #train
    for epoch in range(n_epochs):

        batch_counter = 0
        #iterate batch
        for local_batch, local_labels in train_generator:

            batch_counter = batch_counter + 1

            #transfer to gpu
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            if (optm != None):
                optm.zero_grad()

            #get the predicted values
            pred = NN(local_batch)

            #shrinken the dimension
            pred = torch.squeeze(pred)

            train_loss = criterion(pred, local_labels)

            train_loss.backward()

            if (optm != None):
                optm.step()

            if print_batches != None and batch_counter % print_batches == 0:
                print(
                    f'''batch {batch_counter}
                    Train set - loss: {round_tensor(train_loss)}
                    ''')


        if print_epochs != None and epoch % print_epochs == 0:
            print(
                f'''epoch {epoch}
                Train set - loss: {round_tensor(train_loss)}
                ''')

    return NN

def check_params_consistency(params):

    if params["n_epochs"] < 1:
        raise Exception("n_epochs must be at least 1")

    if params["learning_rate"] < 0:
        raise Exception("learning_rate must be higher than 0")

    if params["validation_split"] < 0 or params["validation_split"] > 99:
         raise Exception("validation_split must between 0 and 99. (0 = No validation set)")

def setDevice(device):

    #force cpu use
    if device == "cpu":
        pass
    #force gpu
    elif device == "gpu":
        device = "cuda:0"

    #use gpu if available and cpu otherwise
    elif device == "auto" and torch.cuda.is_available():
        device = "cuda:0"

    return device

def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)
