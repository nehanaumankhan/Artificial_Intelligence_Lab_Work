import random

# Genetic Algorithm Implementation
class GeneticAlgorithm:
    def _init_(self, population_size, mutation_rate, crossover_rate, elite_count, generations, fitness_function, search_range):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.generations = generations
        self.fitness_function = fitness_function
        self.search_range = search_range

    def initialize_population(self):
        # Randomly generate the initial population
        return [random.randint(self.search_range[0], self.search_range[1]) for _ in range(self.population_size)]

    def evaluate_fitness(self, population):
        # Calculate the fitness for each individual; The fitness function is varies from problem to problem. It reflects the goal of the optimization problem.
        return [self.fitness_function(individual) for individual in population]

    def select_parents(self, population, fitness_scores):
        # Select parents using a roulette-wheel selection
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            probabilities = [1 / len(fitness_scores)] * len(fitness_scores)
        else:
            probabilities = [score / total_fitness for score in fitness_scores]
        return random.choices(population, weights=probabilities, k=self.population_size)

    def crossover(self, parent1, parent2):
        # Perform single-point crossover
        if random.random() < self.crossover_rate:
            return (parent1 + parent2) // 2
        return parent1

    def mutate(self, individual):
        # Apply mutation
        if random.random() < self.mutation_rate:
            return random.randint(self.search_range[0], self.search_range[1])
        return individual

    def create_next_generation(self, population, fitness_scores):
        # Create the next generation
        next_generation = []
        # Add elite individuals
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_count]
        elites = [population[i] for i in elite_indices]
        next_generation.extend(elites)
        # Generate the rest of the population through crossover and mutation
        parents = self.select_parents(population, fitness_scores)
        for _ in range(self.population_size - self.elite_count):
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            next_generation.append(child)
        return next_generation

    def run(self):
        # Run the genetic algorithm
        population = self.initialize_population()
        for generation in range(self.generations):
            fitness_scores = self.evaluate_fitness(population)
            best_individual = max(population, key=self.fitness_function)
            best_fitness = max(fitness_scores)
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")
            if best_fitness >= self.search_range[1]**2:  # Stop if the best solution is found
                break
            population = self.create_next_generation(population, fitness_scores)
        return best_individual


# Example Usage
def fitness_function(x):
    return x ** 2  # Objective: Maximize x^2

# Parameters
population_size = 20
mutation_rate = 0.1
crossover_rate = 0.8
elite_count = 2
generations = 100
search_range = (0, 50)  # Search space: integers between 0 and 50

# Run Genetic Algorithm
ga = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, elite_count, generations, fitness_function, search_range)
best_solution = ga.run()
print(f"Best solution found: {best_solution}, Fitness: {fitness_function(best_solution)}")