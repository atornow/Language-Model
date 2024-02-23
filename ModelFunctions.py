import numpy as np
import TrainingDataLoader
import matplotlib.pyplot as plt

def hiddenFunction(a):
    # Compute tanh of input vector, add non-linearity.
    return np.tanh(a)

def softMax(x):
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


def mainPass(Batch, Targets, prevHidden, U, W, V, b, c):

    oneHot = {}
    savedHiddens = {}
    savedLogits = {}
    savedProbs = {}
    savedHiddens[-1] = np.copy(prevHidden)
    loss = 0

    # Forward pass through Batch
    for char in range(len(Batch)):

        oneHot[char] = np.zeros_like(c)  # encode in 1-of-vocab representation
        oneHot[char][0, Batch[char]] = 1

        savedHiddens[char] = hiddenFunction(b + np.dot(savedHiddens[char - 1], W) + np.dot(oneHot[char], U))
        savedLogits[char] = c + np.dot(savedHiddens[char], V)
        savedProbs[char] = softMax(savedLogits[char])

        prob_array_for_char = savedProbs[char]
        target_prob = prob_array_for_char[0][Targets[char]]
        loss += -np.log(target_prob)

    dOfB = np.zeros_like(b)
    dOfC = np.zeros_like(c)
    dOfU = np.zeros_like(U)
    dOfW = np.zeros_like(W)
    dOfV = np.zeros_like(V)

    dNextHidden = np.zeros_like(savedHiddens[0])

    # Backwards pass through sequence
    for char in reversed(range(len(Batch))):

        # Helper variables for getting derivatives over batch
        dLogit = np.copy(savedProbs[char])
        dLogit[0][Targets[char]] -= 1
        dOfH = dNextHidden + np.dot(dLogit, np.transpose(V))
        dOfHRaw = np.dot(dOfH, np.diag((1 - (np.square(savedHiddens[char]))).flatten()))

        # Calculate and sum derivatives over batch
        dOfB += dOfHRaw
        dOfC += dLogit
        dOfU += np.dot(np.transpose(oneHot[char]), dOfHRaw)
        dOfW += np.dot(np.transpose(savedHiddens[char - 1]), dOfHRaw)
        dOfV += np.dot(np.transpose(savedHiddens[char]), dLogit)

        # Update change in hidden with respect to previous hidden
        dNextHidden = np.dot(dOfHRaw, W.T)

    # Clip values in derivatives to prevent explosions in gradient
    for dParam in [dOfU, dOfW, dOfV, dOfB, dOfC]:
        np.clip(dParam, -5, 5, out=dParam)

    # Return summed loss and derivatives aswell as the last state vector for next pass
    return loss, dOfU, dOfW, dOfV, dOfB, dOfC, savedHiddens[len(Batch) - 1]


def plot_error_set(ErrorSet):
    # Determine the size of ErrorSet
    size = len(ErrorSet)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting
    ax.plot(ErrorSet, label='Error')

    # Scaling the x-axis based on the size of ErrorSet
    if size > 10000000:
        ax.set_xscale('log')
        ax.set_xlabel('Epoch (log scale)')
    else:
        ax.set_xlabel('Epoch')

    # Enhancements for readability
    ax.set_ylabel('Error')
    ax.set_title('Averaged error over all epochs')
    ax.grid(True)
    ax.legend()

    # Show the plot
    plt.show()
