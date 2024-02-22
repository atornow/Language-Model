import TrainingDataLoader
import numpy as np

vocab = 75
hiddenSize = 100
file_path = 'SciFiData.txt'
load = False

U = np.random.uniform(low=-0.3, high=0.3, size=(vocab, hiddenSize))
V = np.random.uniform(low=-0.3, high=0.3, size=(hiddenSize, vocab))
W = np.random.uniform(low=-0.3, high=0.3, size=(hiddenSize, hiddenSize))

h = np.random.uniform(low=-0.3, high=0.3, size=(1, hiddenSize))
b = np.zeros(hiddenSize,)
c = np.zeros(vocab,)
b = b.reshape((1, hiddenSize))
c = c.reshape((1, vocab))

if load:
    Inputs = TrainingDataLoader.file_to_binary_matrix(file_path)
    np.save('SavedInputs.npy', Inputs)

else:
    Inputs = np.load('SavedInputs.npy')
    Inputs = Inputs[:int(len(Inputs) * 0.001)]

dataSize = len(Inputs)

def hiddenFunction(a):
    # Compute tanh of input vector, add non-linearity.
    return np.tanh(a)

def softmax(x):
    # Compute softmax values for each sets of scores in x.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def derivativeLofO(Actual, Desired):
    # Ensure Actual and Desired are NumPy arrays to use NumPy operations
    Actual = np.array(Actual)
    Desired = np.array(Desired)

    # Compute the derivative
    dL_dP = Actual - Desired

    return dL_dP


def crossEntropy(Actual, Desired):
    # Ensure Actual and Desired are NumPy arrays to use NumPy operations
    Actual = np.array(Actual)
    Desired = np.array(Desired)
    # Calculate the cross entropy loss for given array compared to desired array
    loss = -np.sum(Desired * np.log(Actual))
    return loss

Epochs = 1
print(dataSize)

for epoch in range(0, Epochs):
    savedHidden = np.random.uniform(low=-0.3, high=0.3, size=(dataSize, hiddenSize))
    savedError = np.zeros(dataSize)
    savedLogits = np.random.uniform(low=-0.3, high=0.3, size=(dataSize, vocab))

    for char in range(0, dataSize - 1):
        x = np.dot(Inputs[char], U)
        a = b + np.dot(h, W) + x
        h = hiddenFunction(a)

        savedHidden[char] = h
        output = c + np.dot(h, V)
        savedLogits[char] = output

        loss = crossEntropy(softmax(output), Inputs[char + 1])
        savedError[char] = loss

    lossOfC = np.zeros(vocab)
    lossOfC = lossOfC.reshape(1, vocab)
    lossOfB = np.zeros(hiddenSize)
    lossOfB = lossOfB.reshape(1, hiddenSize)
    lossOfU = np.random.uniform(low=-0.3, high=0.3, size=(vocab, hiddenSize))
    lossOfV = np.random.uniform(low=-0.3, high=0.3, size=(hiddenSize, vocab))
    lossOfW = np.random.uniform(low=-0.3, high=0.3, size=(hiddenSize, hiddenSize))

    nextLOfH = np.zeros(hiddenSize)
    nextLOfH = nextLOfH.reshape(1, hiddenSize)
    nextH = np.zeros(hiddenSize)
    nextH = nextH.reshape(1, hiddenSize)

    for char in range (dataSize, 0):
        hOfH = np.dot(np.transpose(W), np.diag(1 - np.square(nextH)))
        lossOfO = derivativeLofO(savedLogits[char], Inputs[char])

        lossOfH = np.dot(nextLOfH, hOfH) + np.dot(lossOfO, np.transpose(V))
        nextLOfH = lossOfH
        nextH = savedHidden[char]

        lossOfC += lossOfO
        lossOfB += np.dot(lossOfH, np.diag(1 - (np.square(savedHidden[char]))))
        lossOfV += np.dot(np.transpose(savedHidden[char]), lossOfO)
        lossOfW += np.dot(np.transpose(savedHidden[char - 1]), np.dot(lossOfH, np.diag(1 - np.square(savedHidden[char]))))
        lossOfU += np.dot(np.transpose(Inputs[char]), np.dot(lossOfH, np.diag(1 - np.square(savedHidden[char]))))


    

print(savedError[1])