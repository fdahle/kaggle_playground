import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

def train_NN(NN, x, y, params, device="auto"):

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

    if 'validation_split' not in params:
        params["validation_split"] = 0

    if 'seed' not in params:
        params["seed"] = 123

    #check the logical consistency of the params
    check_params_consistency(params)

    #extract variables from params dict
    n_epochs = params["n_epochs"]
    lr = params["learning_rate"]
    criterion = params["criterion"]
    eval_params = params["eval_params"]
    optm = params["optimizer"].lower()
    print_epochs = params["print_epochs"]
    val_split = params["validation_split"]
    seed = params["seed"]


    #set the device
    device = setDevice(device)

    #init optimizer if wished
    if (optm == "adam"):
        optm = Adam(NN.parameters(), lr=lr)

    if (criterion == "BCE"):
        criterion = nn.BCELoss()

    #convert inputData to tensors
    x = torch.FloatTensor(x.values)
    y = torch.FloatTensor(y.values)

    dataset = TensorDataset(x, y)

    if val_split > 0:
        train_dataset, val_dataset = random_split(100 - val_split, val_split)

    train_losses = []


    #train
    for epoch in range(n_epochs):

        #get the predicted values
        y_pred = NN(x)

        #shrinken the dimension
        y_pred = torch.squeeze(y_pred)

        train_loss = criterion(y_pred, y)

        if (optm != None):
            optm.zero_grad()

        train_loss.backward()

        if (optm != None):
            optm.step()

        if print_epochs != None and epoch % print_epochs == 0:
            print(
                f'''epoch {epoch}
                Train set - loss: {round_tensor(train_loss)}
                ''')

    return NN

def check_params_consistency(params):

    if params["validation_split"] < 0 or params["validation_split"] > 99:
         raise Exception("Validation_split must between 0 and 99. (0 = No validation set)")


def setDevice(device):

    #default device
    device = "cpu"

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
