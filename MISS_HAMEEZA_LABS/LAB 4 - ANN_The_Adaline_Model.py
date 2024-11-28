def accumulator(inputs, weights, bias):
    return sum([inputs[i] * weights[i] for i in range(len(inputs))]) + bias

def activation(weighted_sum, threshold):
    return 1 if weighted_sum >= threshold else -1
    
def difference(expected_output, weighted_sum):
    return expected_output - weighted_sum

def update_weights(weights, difference, inputs, learning_rate):
    return [weights[i] + learning_rate * difference * inputs[i] for i in range(len(weights))]

def update_bias(bias, difference, learning_rate):
    return [bias + learning_rate * difference]

def adaline(inputs, weights, expected_output, actual_model_output, threshold, bias, learning_rate):
    for i in range(len(inputs)):
        weighted_sum = accumulator(inputs[i], weights, bias)
        model_output = activation(weighted_sum, threshold)
        actual_model_output.append(model_output)
        D = difference(expected_output[i], weighted_sum)
        if D != 0:
            weights = update_weights(weights, D, inputs[i], learning_rate)
            bias = update_bias(bias, D, learning_rate)
    return actual_model_output, weights, bias

def train_model(inputs, expected_output, weights, threshold, bias, learning_rate):
    iteration_counter = 0
    actual_model_output = []
    while actual_model_output != expected_output:
        actual_model_output = []
        #print(f'Iteration # {iteration_counter + 1}')
        iteration_counter += 1
        actual_model_output, weights, bias = adaline(inputs, weights, expected_output, actual_model_output, threshold, bias, learning_rate)
        #print(f'Actual Model Output : {actual_model_output}, Expected Output : {expected_output}\n')
    print(f'Number of Iterations : {iteration_counter}')
    return inputs, expected_output, weights, threshold, bias, learning_rate

# OR
print('OR')
train_model([[0, 0],[0, 1],[1, 0], [1, 1]], [0, 1, 1, 1], [0.1, 0.1], 0.5, 0, 0.2)
train_model([[0, 0],[0, 1],[1, 0], [1, 1]], [0, 1, 1, 1], [0.9, 0.8], 0.5, 0, 0.2)
train_model([[0, 0],[0, 1],[1, 0], [1, 1]], [0, 1, 1, 1], [0.5, 0.5], 0.5, 0, 0.2)
train_model([[0, 0],[0, 1],[1, 0], [1, 1]], [0, 1, 1, 1], [0.2, 0.4], 0.5, 0, 0.2)
train_model([[0, 0],[0, 1],[1, 0], [1, 1]], [0, 1, 1, 1], [-0.3, -0.5], 0.5, 0, 0.2)


