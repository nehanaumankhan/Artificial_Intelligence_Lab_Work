import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Take paragraph input from the user
text = input("Enter a paragraph for summarization:\n")

# Step 2: Preprocess the text into sentences
sentences = text.split(". ")
sentences = [s.strip() for s in sentences if s.strip()]

# Step 3: Compute TF-IDF similarity between sentences
vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(sentences)
similarity_matrix = cosine_similarity(sentence_vectors)

# Compute sentence importance (relevance scores)
sentence_scores = similarity_matrix.sum(axis=1)

# Step 4: Define the Genetic Algorithm
POPULATION_SIZE = 10
GENERATIONS = 30
MUTATION_RATE = 0.2

# Fitness function: Maximize the relevance score of selected sentences
def fitness_function(chromosome):
    selected_sentences = [sentences[i] for i in range(len(chromosome)) if chromosome[i] == 1]
    if not selected_sentences:
        return 0

    # Calculate relevance of selected sentences
    selected_indices = [i for i in range(len(chromosome)) if chromosome[i] == 1]
    relevance_score = sum(sentence_scores[i] for i in selected_indices)

    # Penalty for too short or too long summaries
    penalty = abs(len(selected_sentences) - (len(sentences) // 2))  # Target ~50% of the original
    return relevance_score - penalty

# Generate a random chromosome (selection of sentences)
def generate_chromosome():
    return [random.choice([0, 1]) for _ in range(len(sentences))]

# Initialize the population
def initialize_population():
    return [generate_chromosome() for _ in range(POPULATION_SIZE)]

# Selection: Choose two parents based on fitness
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [f / total_fitness for f in fitness_scores]
    return random.choices(population, weights=probabilities, k=2)

# Crossover: Combine two parents to create offspring
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]

# Mutation: Flip a random bit
def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, len(chromosome) - 1)
        chromosome[index] = 1 - chromosome[index]
    return chromosome

# Genetic Algorithm
def genetic_algorithm():
    population = initialize_population()

    for generation in range(GENERATIONS):
        # Evaluate fitness
        fitness_scores = [fitness_function(chromosome) for chromosome in population]
        best_chromosome = population[fitness_scores.index(max(fitness_scores))]
        best_fitness = max(fitness_scores)

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}")

        # Create new population
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            offspring1 = mutate(crossover(parent1, parent2))
            offspring2 = mutate(crossover(parent1, parent2))
            new_population.extend([offspring1, offspring2])

        population = new_population

    # Decode the best chromosome into a summary
    return [sentences[i] for i in range(len(best_chromosome)) if best_chromosome[i] == 1]

# Run the algorithm
summary_sentences = genetic_algorithm()

# Combine sentences into a single paragraph for the summary
summary = " ".join(summary_sentences)

# Word count for input and output
input_word_count = len(text.split())
output_word_count = len(summary.split())

# Display results
print("\nSummary:")
print(summary)
print(f"\nOriginal Paragraph Word Count: {input_word_count}")
print(f"Summarized Paragraph Word Count: {output_word_count}")
