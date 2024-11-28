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

def test_model(inputs, weights, threshold, bias):
    outputs = []
    for i in range(len(inputs)):
        weighted_sum = accumulator(inputs[i], weights, bias)
        model_output = activation(weighted_sum, threshold)
        outputs.append(model_output)
    return outputs

def print_results(gate_name, inputs, outputs):
    print(f"\n{gate_name} Gate Results:")
    print("Inputs\t\tOutput")
    for i in range(len(inputs)):
        print(f"{inputs[i]}\t{outputs[i]}")

# All possible 3-input combinations
all_inputs = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
]

# AND Gate
print("Training AND Gate...")
and_weights = train_model(all_inputs, [0, 0, 0, 0, 0, 0, 0, 1], [0.1, 0.1, 0.1], 0.5, 0, 0.1)
and_outputs = test_model(all_inputs, and_weights, 0.5, 0)
print_results("AND", all_inputs, and_outputs)

# OR Gate
print("\nTraining OR Gate...")
or_weights = train_model(all_inputs, [0, 1, 1, 1, 1, 1, 1, 1], [0.1, 0.1, 0.1], 0.5, 0, 0.1)
or_outputs = test_model(all_inputs, or_weights, 0.5, 0)
print_results("OR", all_inputs, or_outputs)

# NOR Gate
print("\nTraining NOR Gate...")
nor_weights = train_model(all_inputs, [1, 0, 0, 0, 0, 0, 0, 0], [0.1, 0.1, 0.1], 0.5, 0.6, 0.1)
nor_outputs = test_model(all_inputs, nor_weights, 0.5, 0.6)
print_results("NOR", all_inputs, nor_outputs)

# NAND Gate
print("\nTraining NAND Gate...")
nand_weights = train_model(all_inputs, [1, 1, 1, 1, 1, 1, 1, 0], [0.1, 0.1, 0.1], 0.5, 0.8, 0.1)
nand_outputs = test_model(all_inputs, nand_weights, 0.5, 0.8)
print_results("NAND", all_inputs, nand_outputs)
