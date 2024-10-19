import math

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define a function to calculate the weighted sum for neurons
def weighted_sum(weights, inputs, bias):
    return sum([weights[i] * inputs[i] for i in range(len(inputs))]) + bias

# Feedforward function
def feedforward(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    # Input to hidden layer
    hidden_layer_inputs = []
    for j in range(2):  # Two hidden layer neurons
        hidden_input = weighted_sum(weights_input_hidden[j], inputs, bias_hidden[j])
        hidden_layer_inputs.append(sigmoid(hidden_input))
    
    # Hidden layer to output
    output_input = weighted_sum(weights_hidden_output, hidden_layer_inputs, bias_output)
    predicted_output = sigmoid(output_input)
    
    return hidden_layer_inputs, predicted_output

# Backpropagation function
def backpropagation(inputs, outputs, hidden_layer_inputs, predicted_output, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate):
    # Calculate error for the output
    error = outputs - predicted_output

    # Calculate gradients for the output layer
    delta_output = error * sigmoid_derivative(predicted_output)
    
    # Update weights and bias for the output layer
    for j in range(2):
        weights_hidden_output[j] += learning_rate * delta_output * hidden_layer_inputs[j]
    bias_output += learning_rate * delta_output
    
    # Calculate gradients for the hidden layer
    delta_hidden = []
    for j in range(2):
        delta = delta_output * weights_hidden_output[j] * sigmoid_derivative(hidden_layer_inputs[j])
        delta_hidden.append(delta)
        
    # Update weights and biases for the hidden layer
    for j in range(2):
        for k in range(2):
            weights_input_hidden[j][k] += learning_rate * delta_hidden[j] * inputs[k]
        bias_hidden[j] += learning_rate * delta_hidden[j]
    
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Training function that integrates feedforward and backpropagation
def train(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, outputs, learning_rate, epochs):
    for epoch in range(epochs):
        for i in range(len(inputs)):
            # Perform feedforward pass
            hidden_layer_inputs, predicted_output = feedforward(inputs[i], weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
            
            # Perform backpropagation
            weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = backpropagation(
                inputs[i], outputs[i], hidden_layer_inputs, predicted_output, 
                weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate)
    
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Define a function to test the neural network after training
def test(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    predictions = []
    for i in range(len(inputs)):
        # Perform feedforward pass
        hidden_layer_inputs, predicted_output = feedforward(inputs[i], weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
        
        # Round the predicted output to 0 or 1
        predictions.append(1 if predicted_output >= 0.5 else 0)     # Threshold = 0.5
    
    return predictions

# Inputs for 2-input XOR Gate
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Initialize weights and biases
weights_input_hidden = [[0.5, 0.2], [0.3, 0.7]]     # 2 input neurons, 2 hidden neurons
weights_hidden_output = [0.6, 0.9]                  # 2 hidden neurons, 1 output neuron
bias_hidden = [0.5, 0.6]                            # Biases for hidden layer neurons
bias_output = 0.5                                   # Bias for output neuron

# Setting learning rate and epochs
learning_rate = 0.1
epochs = 10000

# Training and Testing 2-input XOR Gate
print('''BACK PROPAGATION TRAINING ON MULTILAYER FEED FORWARD NETWORK 
TRAINING 2-Input XOR GATE
[0, 0] -> 0
[0, 1] -> 1
[1, 0] -> 1
[1, 1] -> 0
''')

weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, [0, 1, 1, 0], learning_rate, epochs)

# Test XOR Gate
predictions_xor = test([[0, 0], [0, 1], [1, 0], [1, 1]], weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
print("Testing XOR Gate\nOutput Predictions:", predictions_xor)

# Reinitialize weights and biases for XNOR Gate training
weights_input_hidden = [[0.5, 0.2], [0.3, 0.7]]     # 2 input neurons, 2 hidden neurons
weights_hidden_output = [0.6, 0.9]                  # 2 hidden neurons, 1 output neuron
bias_hidden = [0.5, 0.6]                            # Biases for hidden layer neurons
bias_output = 0.5                                   # Bias for output neuron

print('''TRAINING 2-Input XNOR GATE
[0, 0] -> 1
[0, 1] -> 0
[1, 0] -> 0
[1, 1] -> 1
''')

weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, [1, 0, 0, 1], learning_rate, epochs)

# Test XNOR Gate
predictions_xnor = test([[0, 0], [0, 1], [1, 0], [1, 1]], weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
print("Testing XNOR Gate\nOutput Predictions:", predictions_xnor)
