<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Project README</title>
<body>
<h1>Project README</h1>
<h2>Project Overview</h2>
<p>This project is designed to train a neural network model on a dataset contained in <code>SciFiData.txt</code>. The model uses numpy for operations and demonstrates the process of loading data, performing forward and backward passes, and updating weights using gradient descent.</p>

<h2>Files in the Project</h2>
<ul>
    <li><strong>main.py</strong>: The main script that initializes the model, runs the training loop, and saves the error and weight matrices.</li>
    <li><strong>ModelFunctions.py</strong>: Contains all the necessary functions for the model's operation, including <code>inputLoader</code> for loading the dataset, <code>hiddenFunction</code> for calculating the hidden layer's activations, <code>crossEntropy</code> and <code>softMax</code> for loss and output layer calculations, and <code>derivativeLofO</code> for calculating derivatives during the backpropagation.</li>
    <li><strong>TrainingLoader</strong> (within <code>ModelFunctions.py</code>): A utility for handling the loading of training cases. It's an integral part of <code>ModelFunctions</code> used by the <code>inputLoader</code> function.</li>
</ul>

<h2>Project Setup</h2>
<p>Before running the project, ensure you have numpy installed in your environment. You can install numpy using pip:</p>
<pre><code>pip install numpy</code></pre>

<h2>Running the Code</h2>
<p>To run the project, execute the <code>main.py</code> script. This script will automatically load the dataset, perform the training over the specified number of epochs, and save the output matrices and errors as .npy files.</p>
<p>Adjustable parameters within <code>main.py</code> include:</p>
<ul>
    <li>Vocabulary size (<code>vocab</code>)</li>
    <li>Hidden layer size (<code>hiddenSize</code>)</li>
    <li>File path for the dataset (<code>file_path</code>)</li>
    <li>Loading mode (<code>load</code>)</li>
    <li>Learning rate (<code>learningRate</code>)</li>
    <li>Percentage of data used (<code>percentData</code>)</li>
    <li>Number of epochs (<code>epochs</code>)</li>
    <li>Batch size (<code>batchSize</code>)</li>
</ul>
<p>After training, the script will save the weight matrices (<code>U</code>, <code>V</code>, <code>W</code>, <code>b</code>, <code>c</code>) and the training errors to .npy files. It will also output the final batch size used and plot the error set using a function from <code>ModelFunctions</code>.</p>

<h2>Dependencies</h2>
<ul>
    <li>Numpy</li>
</ul>
</body>
</html>