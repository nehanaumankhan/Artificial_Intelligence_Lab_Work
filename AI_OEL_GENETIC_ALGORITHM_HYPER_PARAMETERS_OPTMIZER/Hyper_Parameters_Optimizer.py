import numpy as np
import random
from tensorflow import keras
import streamlit as st

# Load and preprocess MNIST data
mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
X_train_full, X_test = X_train_full / 255.0, X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_train = X_train.reshape(-1, 28, 28)
X_valid = X_valid.reshape(-1, 28, 28)

# Step 1: Initialize population
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = {
            "neurons1": random.randint(32, 512),
            "neurons2": random.randint(32, 512),
            "learning_rate": 10 ** random.uniform(-4, -2)
        }
        population.append(individual)
    return population

# Step 2: Create a model
def create_model(neurons1, neurons2, lr):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(neurons1, kernel_initializer='lecun_normal', activation='selu'),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(neurons2, kernel_initializer='lecun_normal', activation='selu'),
        keras.layers.Dropout(rate=0.1),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="sparse_categorical_crossentropy", 
metrics=["accuracy"])
    return model

# Step 3: Fitness function
def fitness(individual, X_train, y_train, X_valid, y_valid):
    model = create_model(individual["neurons1"], individual["neurons2"], individual["learning_rate"])
    early_stopping = keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid),
                        batch_size=32, verbose=0, callbacks=[early_stopping])
    val_loss = history.history["val_loss"][-1]
    val_accuracy = history.history["val_accuracy"][-1]
    return val_loss, val_accuracy

# Step 4: Selection
def select_parents(population, fitness_scores):
    parents = random.choices(population, weights=1/np.array(fitness_scores), k=2)
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

# Step 7: Genetic Algorithm with Streamlit Integration
def genetic_algorithm(X_train, y_train, X_valid, y_valid, generations=10, pop_size=10, mutation_rate=0.1):
    population = initialize_population(pop_size)
    all_generations = []

    for generation in range(generations):
        st.markdown(f"<h3 style='color: teal;'>Generation {generation+1}/{generations}: Evaluating individuals...</h3>", unsafe_allow_html=True)
        progress = st.progress(0)

        generation_results = []
        for idx, ind in enumerate(population):
            val_loss, val_accuracy = fitness(ind, X_train, y_train, X_valid, y_valid)
            ind_results = {
                "neurons1": ind["neurons1"],
                "neurons2": ind["neurons2"],
                "learning_rate": ind["learning_rate"],
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }
            generation_results.append(ind_results)
            progress.progress((idx + 1) / len(population))

        best_individual = min(generation_results, key=lambda x: x["val_loss"])
        st.markdown(f"<div style='color: navy; background-color: #dff0d8; padding: 10px; border-radius: 5px;'>"
                    f"Best in Generation {generation+1}: {best_individual}</div>", unsafe_allow_html=True)
        all_generations.append(generation_results)

        fitness_scores = [result["val_loss"] for result in generation_results]
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1 = mutate(crossover(parent1, parent2), mutation_rate)
            child2 = mutate(crossover(parent1, parent2), mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    return all_generations

# Streamlit Interface
st.title("🎯 Genetic Algorithm for Neural Network Hyperparameter Tuning")
st.markdown("""
<style>
    body {
        background-color: #f8f9fa;
        color: #343a40;
    }
    .stButton>button {
        color: white;
        background-color: teal;
        font-size: large;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

if st.button("🚀 Run Optimization"):
    with st.spinner("Running Genetic Algorithm..."):
        all_generations = genetic_algorithm(
            X_train, y_train, X_valid, y_valid,
            generations=5,
            pop_size=5, 
            mutation_rate=0.1
        )
    st.success("Optimization Complete! 🎉")
    st.write("### 📊 Hyperparameter Evolution")
    for gen_idx, generation in enumerate(all_generations, 1):
        st.write(f"#### Generation {gen_idx}")
        st.table(generation)
