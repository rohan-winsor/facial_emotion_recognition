import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.utils import to_categorical

def get_data():
    data = pd.read_csv("fer2013.csv", index_col=False)
    X = [[int(a) for a in i.split()] for i in data['pixels']]
    Y = data['emotion']
    X, Y = np.array(X) / 255.0, np.array(Y)
    training_len = int(len(X) * 0.8)
    X, Y = shuffle(X, Y)
    Xtrain, Ytrain = X[:training_len], Y[:training_len]
    Xvalid, Yvalid = X[training_len:], Y[training_len:]
    X0, Y0 = Xtrain[Ytrain!=1, :], Ytrain[Ytrain!=1]
    X1 = Xtrain[Ytrain==1, :]
    X1 = np.repeat(X1, 9, axis=0)
    Xtrain = np.vstack([X0, X1])
    Ytrain = np.concatenate((Y0, [1]*len(X1)))
    Xtrain = Xtrain.reshape(-1, 48,48, 1)
    Xvalid = Xvalid.reshape(-1, 48,48, 1)
    
    return Xtrain, to_categorical(Ytrain, 7), Xvalid, to_categorical(Yvalid, 7)

