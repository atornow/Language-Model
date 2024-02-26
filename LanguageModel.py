import numpy as np
import ModelFunctions

# Hyperparameters of model
hiddenSize = 100
learningRate = 0.00001
batchSize = 30
size = 100
decay = 0.9

# Load data from txt and create dictionaries for one-hot encoding
data = open('ScifiData.txt', 'r').read()
chars = list(set(data))
charToIndex = {ch: i for i, ch in enumerate(chars)}
indexToChar = {i: ch for i, ch in enumerate(chars)}
dataSize, vocab = len(data), len(chars)

# Initialize weight matrices and bias vectors
U = np.random.uniform(low=-0.01, high=0.01, size=(vocab, hiddenSize))
V = np.random.uniform(low=-0.01, high=0.01, size=(hiddenSize, vocab))
W = np.random.uniform(low=-0.01, high=0.01, size=(hiddenSize, hiddenSize))
b = np.zeros((1, hiddenSize))
c = np.zeros((1, vocab))

# Initialize weight and vector derivative memories for adagrad
memdOfB = np.zeros_like(b)
memdOfC = np.zeros_like(c)
memdOfU = np.zeros_like(U)
memdOfW = np.zeros_like(W)
memdOfV = np.zeros_like(V)

# Get starting loss point
average_loss = -np.log(1.0 / vocab) * batchSize  # Initial loss at iteration 0
iteration_counter, data_pointer = 0, 0
lossOverItterations = []
while iteration_counter < 100000:
    # Keep track of current iteration and place in batch

    if data_pointer + batchSize + 1 >= len(data) or iteration_counter == 0:
        prevHidden = np.zeros((1, hiddenSize))
        data_pointer = 0

    # Create a batch of batchSize and equal sized target array to compare
    Batch = [charToIndex[char] for char in data[data_pointer:data_pointer + batchSize]]
    Targets = [charToIndex[char] for char in data[data_pointer + 1:data_pointer + batchSize + 1]]

    # Run batch through full forward/backward pass to get derivatives
    loss, dOfU, dOfW, dOfV, dOfB, dOfC, prevHidden = ModelFunctions.mainPass(Batch, Targets, prevHidden, U, W, V, b, c)

    # Average loss over batches and print it every 100 iterations
    average_loss = average_loss * 0.999 + loss * 0.001
    if iteration_counter % 100 == 0:
        print('Iteration ' + str(iteration_counter) + ' loss: ' + str(average_loss))

    if iteration_counter % 100 == 0:
        seed = Batch[-1]
        returnString = ModelFunctions.sampleLanguage(chars, seed, prevHidden, size, U, W, V, b, c)
        print(returnString)
        lossOverItterations.append(average_loss)
        print(U[1][1])

    # Update the weights and bias with adagrad method
    for parameter, gradient, memory in zip([U, W, V, b, c], [dOfU, dOfW, dOfV, dOfB, dOfC],
                                           [memdOfU, memdOfW, memdOfV, memdOfB, memdOfC]):
        memory = decay * memory + (1 - decay) * (gradient ** 2)
        parameter += -learningRate * gradient / np.sqrt(memory + 1e-8)


    data_pointer += batchSize
    iteration_counter += 1

ModelFunctions.plot_error_set(lossOverItterations)