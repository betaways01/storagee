import numpy as np
from deap import base, creator, tools

class sbox:
    def __init__(self, sbox_list):
        self.sbox_list = sbox_list

    def diffusion(self):
        n = len(self.sbox_list)
        output_diffusion = [0] * n
        for i in range(n):
            output_diffusion[self.sbox_list[i]] = 1
        return sum(output_diffusion) / n

    def nonlinearity(self):
        n = len(self.sbox_list)
        wt = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                wt[i, j] = (i & j) == i
        wt = np.dot(wt, self.sbox_list)
        return max([abs(i) for i in wt]) - n

    def algebraic_degree(self):
        n = len(self.sbox_list)
        anf = [0] * n
        for i in range(n):
            for j in range(n):
                if (i & j) == i:
                    anf[self.sbox_list[j]] ^= 1
        return max([i.bit_length() for i in anf]) - 1

    def linear_approximation_probability(self):
        n = len(self.sbox_list)
        ddt = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ddt[i ^ j][self.sbox_list[i] ^ self.sbox_list[j]] += 1
        return max([max(i) for i in ddt]) / n

# Define the fitness function
def sbox_fitness(sbox_list):
    sbox_instance = sbox(sbox_list)
    diffusion = sbox_instance.diffusion()
    nonlinearity = sbox_instance.nonlinearity()
    algebraic_degree = sbox_instance.algebraic_degree()
    linear_approximation_probability = sbox_instance.linear_approximation_probability()
    # Define a weight for each property based on the desired properties of the s-box
    weights = (1, 1, 1, 1)
    # Return the weighted sum of the properties
    return (diffusion*weights[0], nonlinearity*weights[1], algebraic_degree*weights[2], linear_approximation_probability*weights[3])

# Create the individual and population classes
creator.create("Sbox", list, fitness=base.Fitness)
toolbox = base.Toolbox()
# Register the sbox function with the toolbox
toolbox.register("sbox", tools.initIterate, creator.Sbox)
# register the population
toolbox.register("population", tools.initRepeat, list, toolbox.sbox, n=16)

# Define the genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the fine-tuning parameters
pop_size = 50
num_gen = 100
cxpb = 0.5
mutpb = 0.1

# Initialize the population
pop = toolbox.population(n=pop_size)

# Evaluate the fitness of the individuals
fitnesses = list(map(sbox_fitness, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Perform the evolutionary loop
for generation in range(num_gen):
    # Select the next generation
    offspring = toolbox.select(pop, len(pop))
    # Apply crossover and mutation
    offspring = algorithms.varAnd(offspring, toolbox, cxpb=cxpb, mutpb=mutpb)
    # Evaluate the fitness of the offspring
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(sbox_fitness, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # Replace the current population with the offspring
    pop[:] = offspring

# Select the best individual
best_sbox = tools.selBest(pop, k=1)[0]

# Use the best S-box in the permutation function of the sponge algorithm
output = permutation_function(best_sbox)
