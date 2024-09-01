# ANSI escape sequences for colors and bold text
BOLD = '\033[1m'
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
ENDC = '\033[0m'  # Resets to the default color

input_x1 = [0, 0, 1, 1]
input_x2 = [0, 1, 0, 1]
expected_output = [0, 1, 1, 1]

weighted_sum = [0] * 4
actual_output = [0] * 4
difference = [1] * 4

initial_weight_1 = [-0.3] * 4
initial_weight_2 = [-0.5] * 4

final_weight_1 = [0] * 4
final_weight_2 = [0] * 4

iterations = 0
learning_rate = 0.2
bias = 0

print(f'{BOLD}{BLUE}\nTWO INPUTS PERCEPTRON MODEL{ENDC}')

while sum(difference) and iterations < 10:
    print(f'{BOLD}{GREEN}ITERATION {iterations+1}{ENDC}')
    for i in range(4):
        
        # Accumulation
        weighted_sum[i] = input_x1[i] * initial_weight_1[i] + input_x2[i] * initial_weight_2[i] + bias
        
        # Activation Function
        actual_output[i] = 1 if weighted_sum[i] >= 0.5 else 0
        
        # Calculating difference
        difference[i] = expected_output[i] - actual_output[i]
        
        # Updating weights
        final_weight_1[i] = initial_weight_1[i] + learning_rate * difference[i] * input_x1[i]
        final_weight_2[i] = initial_weight_2[i] + learning_rate * difference[i] * input_x2[i]
        initial_weight_1[i] = round(final_weight_1[i], 4)
        initial_weight_2[i] = round(final_weight_2[i], 4)
    
    print(f"{BOLD}{RED}difference:{ENDC}  {difference}")
    print(f"{BOLD}{BLUE}W1:{ENDC} {initial_weight_1}")
    print(f"{BOLD}{GREEN}W2:{ENDC} {initial_weight_2}")
    print(f"{BOLD}{RED}actual_output:{ENDC}  {actual_output}\n")
    
    iterations += 1

print(f'{BOLD}{BLUE}Total iterations: {iterations}{ENDC}')



