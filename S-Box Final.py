# REPLICATION OF THE PAPER
# THE PAPER USED C, WE USE PYTHON
# WE USE THE DEAP LIBRARY INSTEAD OF THE SAGE LIBRARY, TO REPLICATE THE RESULTS CORRECTLY IN PYTHON

# Make sure to initialize device PRN
# otherwise, it will not be properly initialized.

# IMPORT DEAP LIB, AND THE CREATOR TOOLS
from deap import base, creator, tools
import random
import string

# Define the individual and population
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Define the genetic operators
# Registering each of the components
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bool, 16)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Define the evaluation function
def linearity(sbox):
    """Calculate the linearity of a 4x4-bit S-box"""
    sbox = [int(x) for x in format(sbox, '016b')]
    linearity = 0
    for i in range(16):
        for j in range(16):
            linearity += abs((i ^ j) ^ sbox[i ^ j])
    return linearity


def differential_uniformity(sbox):
    """Calculate the differential uniformity of a 4x4-bit S-box"""
    sbox = [int(x) for x in format(sbox, '016b')]
    uniformity = 0
    for i in range(16):
        for j in range(16):
            uniformity += abs(sbox[i ^ j] ^ sbox[i] ^ sbox[j])
    return uniformity


def algebraic_degree(sbox):
    """Calculate the algebraic degree of a 4x4-bit S-box"""
    sbox = [int(x) for x in format(sbox, '016b')]
    degree = 0
    for i in range(16):
        for j in range(16):
            degree += bin(sbox[i] ^ sbox[j]).count("1")
    return degree


def nonlinearity(sbox):
    """Calculate the nonlinearity of a 4x4-bit S-box"""
    sbox = [int(x) for x in format(sbox, '016b')]
    nonlinearity = 0
    for i in range(16):
        for j in range(16):
            nonlinearity += bin(i ^ j ^ sbox[i] ^ sbox[j]).count("1")
    return nonlinearity


def evaluate(individual):
    """Evaluate the cryptographic properties of a 4x4-bit S-box"""
    # Convert the binary string to integer
    sbox = 0
    for bit in individual:
        sbox = (sbox << 1) | bit
    # Calculate the cryptographic properties
    LINEARITY_VAL = linearity(sbox)
    DIFF_UNIFORMITY_VAL = differential_uniformity(sbox)
    ALGEBRAIC_DEGREE_VAL = algebraic_degree(sbox)
    NON_LINEARITY_VAL = nonlinearity(sbox)
    # Return the fitness value
    return (LINEARITY_VAL + DIFF_UNIFORMITY_VAL + ALGEBRAIC_DEGREE_VAL + NON_LINEARITY_VAL),


toolbox.register("evaluate", evaluate)

# Define the selection operator
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# Define the mutation operator
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# Define the parameters of the genetic algorithm
pop = toolbox.population(n=300)
CXPB, MUTPB, NGEN = 0.5, 0.2, 40

# Run the genetic algorithm
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for g in range(NGEN):
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

    # display results
    print(offspring)  # Make sure to initialize device PRN

    print(pop[:])
