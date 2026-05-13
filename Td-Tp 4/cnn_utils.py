import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def initialize_with_zeros(dim):
    w = np.zeros([dim,1])
    b = 0.0
    return w, b

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        ## Calcul du coût et du gradient
        grads, cost = propagate(w, b, X, Y)
        
        ## Récupérer les dérivés de grads
        dw = grads["dw"]
        db = grads["db"]
        
        ## mise à jour
        w += -learning_rate * dw
        b += -learning_rate * db
        
        ## Enregistrer les coûts
        if i % 100 == 0:
            costs.append(cost)
        
            ## Imprimer le coût toutes les 100 itérations d'entraînement
            if print_cost:
                print ("Coût après itération %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w, b, X):
    n = X.shape[1]
    Y_prediction = np.zeros((1, n))
    w = w.reshape(X.shape[0], 1)
    
    ## Calculer le vecteur "hat_Y" prédisant les probabilités qu'un chat soit présent dans l'image
    hat_Y = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(hat_Y.shape[1]):
        ## Convertir les probabilités hat_Y[0,i] en prédictions réelles p[0,i]
        if hat_Y[0, i] > 0.5 :
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    
    return Y_prediction
