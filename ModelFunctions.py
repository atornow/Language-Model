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

def inputLoader(file_path, load, percentData):
    if load:
        Inputs = TrainingDataLoader.file_to_binary_matrix(file_path)
        np.save('SavedInputs.npy', Inputs)
        return Inputs

    else:
        Inputs = np.load('SavedInputs.npy')
        Inputs = Inputs[:int(len(Inputs) * percentData)]
        return Inputs

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
