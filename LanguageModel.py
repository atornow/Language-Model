import numpy as np
import ModelFunctions

vocab = 75
hiddenSize = 100
file_path = 'SciFiData.txt'
load = False
learningRate = 0.1
percentData = 0.01
epochs = 100
Inputs = ModelFunctions.inputLoader(file_path, load, percentData)
dataSize = len(Inputs)

U = np.random.uniform(low=-0.1, high=0.1, size=(vocab, hiddenSize))
V = np.random.uniform(low=-0.1, high=0.1, size=(hiddenSize, vocab))
W = np.random.uniform(low=-0.1, high=0.1, size=(hiddenSize, hiddenSize))

h = np.random.uniform(low=-0.3, high=0.3, size=(1, hiddenSize))
b = np.zeros((1, hiddenSize))
c = np.zeros((1, vocab))
savedHidden = np.zeros((dataSize, hiddenSize))
savedError = np.zeros(epochs)
savedLogits = np.zeros((dataSize, vocab))

for epoch in range(0, epochs):

    # Forward pass
    extraError = np.zeros(dataSize)
    for char in range(0, dataSize - 1):
        x = np.dot(Inputs[char], U)
        a = b + np.dot(h, W) + x
        h = ModelFunctions.hiddenFunction(a)

        savedHidden[char] = h
        output = c + np.dot(h, V)
        savedLogits[char] = output

        loss = ModelFunctions.crossEntropy(ModelFunctions.softMax(output), Inputs[char + 1])
        savedError[epoch] += loss
        extraError[char] = loss

    lossOfC = np.zeros((1, vocab))
    lossOfB = np.zeros((1, hiddenSize))
    lossOfU = np.random.uniform(low=-0.3, high=0.3, size=(vocab, hiddenSize))
    lossOfV = np.random.uniform(low=-0.3, high=0.3, size=(hiddenSize, vocab))
    lossOfW = np.random.uniform(low=-0.3, high=0.3, size=(hiddenSize, hiddenSize))

    nextLOfH = np.zeros((1, hiddenSize))
    nextH = np.zeros((1, hiddenSize))

    # Backwards pass through sequence
    for char in range(dataSize, 0):
        hOfH = np.dot(np.transpose(W), np.diag(1 - np.square(nextH)))
        lossOfO = ModelFunctions.derivativeLofO(savedLogits[char], Inputs[char])

        lossOfH = np.dot(nextLOfH, hOfH) + np.dot(lossOfO, np.transpose(V))
        nextLOfH = lossOfH
        nextH = savedHidden[char]

        lossOfC += lossOfO
        lossOfB += np.dot(lossOfH, np.diag(1 - (np.square(savedHidden[char]))))
        lossOfV += np.dot(np.transpose(savedHidden[char]), lossOfO)
        lossOfW += np.dot(np.transpose(savedHidden[char - 1]), np.dot(lossOfH, np.diag(1 - np.square(savedHidden[char]))))
        lossOfU += np.dot(np.transpose(Inputs[char]), np.dot(lossOfH, np.diag(1 - np.square(savedHidden[char]))))

    c = c - learningRate * (lossOfC / dataSize)
    b = b - learningRate * (lossOfB / dataSize)
    V = V - learningRate * (lossOfV /dataSize)
    W = W - learningRate * (lossOfW / dataSize)
    U = U - learningRate * (lossOfU / dataSize)

savedError = savedError / dataSize

print(dataSize)

ModelFunctions.plot_error_set(savedError)
ModelFunctions.plot_error_set(extraError)