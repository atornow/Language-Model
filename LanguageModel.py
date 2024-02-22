import random

import numpy as np
import math

vocab = 40
data = 492324
hiddenSize = 100
dataSize = len(data)

U = np.random.uniform(low=-0.3, high=0.3, size=(hiddenSize, hiddenSize))
V = np.random.uniform(low=-0.3, high=0.3, size=(hiddenSize, hiddenSize))
W = np.random.uniform(low=-0.3, high=0.3, size=(hiddenSize, hiddenSize))
Inputs = np.random.uniform(low=-0.3, high=0.3, size=(dataSize, vocab))

h = np.random.uniform(low=-0.3, high=0.3, size=(1, hiddenSize))
b = np.zeros(1, hiddenSize)
c = np.zeros(1, hiddenSize)

def hiddenFunction(a):
    return np.tanh(a)

def softmax(x):
    # Compute softmax values for each sets of scores in x.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


Epochs = 1000

for epoch in Epochs:
    savedHidden = np.zeros(dataSize, hiddenSize)
    savedError = np.zeros(dataSize, vocab)
    nextHidden = np.zeros(1, 100)

    for char in range(0, dataSize):
        x = Inputs[char] * U
        a = b + (h * W) + x
        nextHidden = hiddenFunction(a)

        savedHidden[char] = nextHidden
        output = c + nextHidden * V

        savedError[char] = softmax(output) - Inputs[char + 1]
    
