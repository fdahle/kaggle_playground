import matplotlib.pyplot as plt
import numpy as np

def plot_curve(data, titles):
    #titles is the name of the curves

    if len(data) != len(titles):
        raise Exception("please provide the same number of entries in data and titles")

    for i, lst in enumerate(data):

        #convert to numpy array for easier accessability
        arr = np.array(lst)

        #get x and y
        x = arr[:,0]
        y = arr[:,1]

        plt.plot(x,y, label=titles[i])

    plt.legend(loc="upper left")
    plt.show()
