import numpy as np
import random
from tensorflow import keras
# Ensure the data is reshaped correctly
mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# Normalize and split the data
X_train_full, X_test = X_train_full / 255.0, X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


X_train = X_train.reshape(-1, 28, 28)  # Reshape if necessary
X_valid = X_valid.reshape(-1, 28, 28)  # Reshape if necessary


# Step 1: Define search space
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = {
            "neurons1": random.randint(32, 512),
            "neurons2": random.randint(32, 512),
            "learning_rate": 10 ** random.uniform(-4, -2)  # Log-scale
        }
        population.append(individual)
    return population

# Step 2: Create the model
def create_model(neurons1, neurons2, lr):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),  # Adjust input shape for MNIST
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(neurons1, kernel_initializer='lecun_normal', activation='selu'),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(neurons2, kernel_initializer='lecun_normal', activation='selu'),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(10, activation="softmax")  # 10 classes for digit classification
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    return model


# Step 3: Fitness function

def fitness(individual, X_train, y_train, X_valid, y_valid):
    

    model = create_model(individual["neurons1"], individual["neurons2"], individual["learning_rate"])
    early_stopping = keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid),
                    batch_size=32, verbose=0, callbacks=[early_stopping])
    # history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid), batch_size=32, verbose=0)
    val_loss = history.history["val_loss"][-1]  # Use final validation loss as fitness
    val_accuracy = history.history["val_accuracy"][-1]  # Final validation accuracy
    return val_loss, val_accuracy
    
# Step 4: Selection (Tournament Selection)
def select_parents(population, fitness_scores):
    parents = random.choices(population, weights=1/np.array(fitness_scores), k=2)  # Lower loss = higher fitness
    return parents

# Step 5: Crossover
def crossover(parent1, parent2):
    child = {
        "neurons1": random.choice([parent1["neurons1"], parent2["neurons1"]]),
        "neurons2": random.choice([parent1["neurons2"], parent2["neurons2"]]),
        "learning_rate": random.choice([parent1["learning_rate"], parent2["learning_rate"]])
    }
    return child

# Step 6: Mutation
def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        individual["neurons1"] = random.randint(32, 512)
    if random.random() < mutation_rate:
        individual["neurons2"] = random.randint(32, 512)
    if random.random() < mutation_rate:
        individual["learning_rate"] = 10 ** random.uniform(-4, -2)
    return individual

# Step 7: Genetic Algorithm
def genetic_algorithm(X_train, y_train, X_valid, y_valid, generations, pop_size, mutation_rate):
    population = initialize_population(pop_size)
    for generation in range(generations):
        print(f"Generation {generation+1}/{generations}")
        
        # Evaluate fitness
        results = [fitness(ind, X_train, y_train, X_valid, y_valid) for ind in population]
        fitness_scores = [result[0] for result in results]  # Extract val_loss
        accuracy_scores = [result[1] for result in results]  # Extract val_accuracy
        
        # Log the best results of the current generation
        best_loss = min(fitness_scores)
        best_accuracy = max(accuracy_scores)
        print(f"Best fitness (val_loss): {best_loss}")
        print(f"Best accuracy (val_accuracy): {best_accuracy}")
        
        # Select next generation
        new_population = []
        for _ in range(pop_size // 2):  # Each iteration produces 2 children
            parent1, parent2 = select_parents(population, fitness_scores)
            child1 = mutate(crossover(parent1, parent2), mutation_rate)
            child2 = mutate(crossover(parent1, parent2), mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population
    
    # Return the best hyperparameters
    best_index = np.argmin(fitness_scores)
    # print(f"Final Best Accuracy (val_accuracy): {accuracy_scores[best_index]}")
    return population[best_index], best_loss, accuracy_scores[best_index]

#  Run Genetic Algorithm
best_hyperparameters, best_loss, best_accuracy = genetic_algorithm(
    X_train, y_train, X_valid, y_valid, 
    generations=5, 
    pop_size=5, 
    mutation_rate=0.1
)

print("Best Hyperparameters:", best_hyperparameters)
print(f"Best Validation Loss: {best_loss}")
print(f"Best Validation Accuracy: {best_accuracy}")