def accumulator(inputs, weights, bias):
    return sum([inputs[i] * weights[i] for i in range(len(inputs))]) + bias

def activation(weighted_sum, threshold):
    return 1 if weighted_sum >= threshold else 0

def difference(expected_output, model_output):
    return expected_output - model_output

def update_weights(weights, difference, inputs, learning_rate):
    return [weights[i] + learning_rate * difference * inputs[i] for i in range(len(weights))]

def perceptron(inputs, weights, expected_output, threshold, bias, learning_rate):
    actual_model_output = []
    for i in range(len(inputs)):
        weighted_sum = accumulator(inputs[i], weights, bias)
        model_output = activation(weighted_sum, threshold)
        actual_model_output.append(model_output)
        D = difference(expected_output[i], model_output)
        if D != 0:
            weights = update_weights(weights, D, inputs[i], learning_rate)
    return actual_model_output, weights

def train_model(inputs, expected_output, weights, threshold, bias, learning_rate):
    iteration_counter = 0
    actual_model_output = []
    while actual_model_output != expected_output:
        actual_model_output = []
        iteration_counter += 1
        actual_model_output, weights = perceptron(inputs, weights, expected_output, threshold, bias, learning_rate)
    print(f'Number of Epochs for training: {iteration_counter}')
    return weights

def test_model(test_inputs, weights, threshold, bias):
    test_outputs = []
    for i in range(len(test_inputs)):
        weighted_sum = accumulator(test_inputs[i], weights, bias)
        model_output = activation(weighted_sum, threshold)
        test_outputs.append(model_output)
    return test_outputs

trained_weights = train_model([[0,0],
                               [0,1],
                               [1,1]],   
                              [0, 0,1], [0.1, 0.1], 0.5, 0, 0.1)

print('THE PERCEPTRON MODEL\n')
# 3 input AND Gate Training
print('''Training AND Gate...
      Training Data
      [0, 1, 0] -> 0
      [0, 1, 1] -> 0
      [1, 0, 0] -> 0
      [1, 0, 1] -> 0
      [1, 1, 0] -> 0
      [1, 1, 1] -> 1
      ''')
trained_weights = train_model([[0, 1, 0],
                               [0, 1, 1],
                               [1, 0, 0],
                               [1, 0, 1],
                               [1, 1, 0],
                               [1, 1, 1]], 
                              [0, 0, 0, 0, 0, 1], [0.1, 0.1, 0.1], 0.5, 0, 0.1)

# 3 input AND Gate Testing
print('\nTesting AND Gate...')
test_outputs = test_model([[0, 0, 0], [0, 0, 1]], trained_weights, 0.5, 0)
print(f'''Test Input:
      [0, 0, 0]
      [0, 0, 1]
      Test Outputs: {test_outputs}''')

# 3 input OR Gate Training
print('''\nTraining OR Gate...
      Training Data
      [0, 0, 0] -> 0
      [0, 0, 1] -> 1
      [0, 1, 0] -> 1
      [0, 1, 1] -> 1
      [1, 0, 0] -> 1
      [1, 0, 1] -> 1
      ''')
trained_weights = train_model([[0, 0, 0],
                               [0, 0, 1],
                               [0, 1, 0],
                               [0, 1, 1],
                               [1, 0, 0],
                               [1, 0, 1]], 
                              [0, 1, 1, 1, 1, 1], [0.1, 0.1, 0.1], 0.5, 0, 0.1)

# 3 input OR Gate Testing
print('\nTesting OR Gate...')
test_outputs = test_model([[1, 1, 0], [1, 1, 1]], trained_weights, 0.5, 0)
print(f'''Test Input:
      [1, 1, 0]
      [1, 1, 1]
      Test Outputs: {test_outputs}''')

# 3 input NOR Gate Training
print('''\nTraining NOR Gate...
      Training Data
      [0, 0, 0] -> 1
      [0, 0, 1] -> 0
      [0, 1, 0] -> 0
      [0, 1, 1] -> 0
      [1, 0, 0] -> 0
      [1, 0, 1] -> 0
      ''')
trained_weights = train_model([[0, 0, 0],
                               [0, 0, 1],
                               [0, 1, 0],
                               [0, 1, 1],
                               [1, 0, 0],
                               [1, 0, 1]], 
                              [1, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1], 0.5, 0.6, 0.1)

# 3 input NOR Gate Testing
print('\nTesting NOR Gate...')
test_outputs = test_model([[1, 1, 0], [1, 1, 1]], trained_weights, 0.5, 0.6)
print(f'''Test Input:
      [1, 1, 0]
      [1, 1, 1]
      Test Outputs: {test_outputs}''')

# 3 input NAND Gate Training
print('''\nTraining NAND Gate...
      Training Data
      [0, 1, 0] -> 1
      [0, 1, 1] -> 1
      [1, 0, 0] -> 1
      [1, 0, 1] -> 1
      [1, 1, 0] -> 1
      [1, 1, 1] -> 0
      ''')
trained_weights = train_model([[0, 1, 0],
                               [0, 1, 1],
                               [1, 0, 0],
                               [1, 0, 1],
                               [1, 1, 0],
                               [1, 1, 1]], 
                              [1, 1, 1, 1, 1, 0], [0.1, 0.1, 0.1], 0.5, 0.8, 0.1)

# 3 input NAND Gate Testing
print('\nTesting NAND Gate...')
test_outputs = test_model([[0, 0, 0], [0, 0, 1]], trained_weights, 0.5, 0.8)
print(f'''Test Input:
      [0, 0, 0]
      [0, 0, 1]
      Test Outputs: {test_outputs}''')