from functools import lru_cache
from sage.all import *
from sage.crypto.sbox import SBox

import random

# logging
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.DEBUG)

''' Variables to modify the algorithm '''

# The number of input bits for the sbox (we currently assume the number of output bits is identical)
SBOX_SIZE = 4

# Weights for each of the properties we can calculate (should add up to 1)
BRANCH_WEIGHT = 0.9
NON_LINEARITY_WEIGHT = 0.04
DIFFERENTIAL_UNIFORMITY_WEIGHT = 0.04
COMPLEXITY_WEIGHT = 0.02

# Any s-box with a score at or above this would be considered satisfactory
SATISFACTORY_SCORE = 0.95

# The size of the population for each generation
POP_SIZE = int(10)

# The maximum number of generations to continue for
MAX_GENERATIONS = 1000000

# If there is no improvement for this many generations, stop
BEST_LIMIT = 10 * MAX_GENERATIONS // 100

# When breeding the next generation, keep this many best candidates as possible parents
SELECTION_SIZE = 20 * POP_SIZE // 100
# When creating a new generation, keep this many of the worst solutions unchanged
KEEP_WORST = 5 * POP_SIZE // 100
# When creating a new generation, keep this many of the best solutions unchanged
KEEP_BEST = 5 * POP_SIZE // 100
# When creating a new generation, include this many randomly-generated candidates
KEEP_RANDOM = 10 * POP_SIZE // 100

# Percentage of the time mutation should occur
MUTATION_RATE = 0.1
# Percentage of the time crossover should occur
CROSSOVER_RATE = 0.8

# After this many candidates have been processed, report on the current best solution
REPORT_CALCULATION_LIMIT = 100000


@lru_cache(maxsize=POP_SIZE)
def calculate_branch_number(sbox):
    # TODO: Implement this correctly
    return 3


@lru_cache(maxsize=POP_SIZE)
def calculate_non_linearity(sbox):
    input_size = SBOX_SIZE
    output_size = SBOX_SIZE
    non_linearity = 2 ^ (output_size-1) - \
        max([abs(len(set([sbox[x ^^ y] ^^ sbox[x] ^^ sbox[y] for x in range(2**input_size) \
            for y in range(2**input_size)])) - 2 ^ (output_size-1)) for x in range(2 ^ input_size)])
    return non_linearity


@lru_cache(maxsize=POP_SIZE)
def calculate_differential_uniformity(sbox):
    input_size = SBOX_SIZE
    output_size = SBOX_SIZE
    differential_uniformity = max([max([len(set([sbox[x ^^ y] ^^ sbox[x] ^^ sbox[y] \
        for x in range(2 ^ input_size)])) for y in range(2 ^ input_size)]) for y in range(2 ^ input_size)])
    return differential_uniformity


@lru_cache(maxsize=POP_SIZE)
def calculate_complexity(sbox):
    sbox = SBox(sbox)
    # Calculate the number of fixed points
    fixed_points = sum([1 for x in range(2 ^ sbox.input_size()) if sbox(x) == x])
    # Calculate the number of active S-boxes
    active_sboxes = sum([1 for x in range(2 ^ sbox.input_size()) if sbox(x) != x])
    # Combine the fixed points and active S-boxes into a single complexity score
    complexity = fixed_points * (2 ^ sbox.output_size() - active_sboxes)
    return complexity


@lru_cache
def get_min_branch_number(sbox_size):
    return 0  # TODO: Check this value


@lru_cache
def get_max_branch_number(sbox_size):
    return 3  # TODO: Calculate this value depending on the size of the s-box


@lru_cache
def get_min_non_linearity(sbox_size):
    return 0  # TODO: Check this value
    

@lru_cache
def get_max_non_linearity(sbox_size):
    return 100  # TODO: Calculate this value depending on the size of the s-box


@lru_cache
def get_min_differential_uniformity(sbox_size):
    return 0  # TODO: Check this value
    

@lru_cache
def get_max_differential_uniformity(sbox_size):
    return 100  # TODO: Calculate this value depending on the size of the s-box


@lru_cache
def get_min_complexity(sbox_size):
    return 0  # TODO: Check this value
    

@lru_cache
def get_max_complexity(sbox_size):
    return 100  # TODO: Calculate this value depending on the size of the s-box


# Scales the given value to be between 0 and 1 (inclusive),
# based on the given min_val and max_val.
# If min_val > max_val then inverts the result
# (i.e., makes values closer to min_val better)
def scale(val, min_val = 0, max_val = 1):
    if min_val > max_val:
        return 1 - scale(val, max_val, min_val)
    val = max(val, min_val)
    val = min(val, max_val)
    return (val - min_val) / (max_val - min_val)

    
@lru_cache(maxsize=POP_SIZE)
# Define the fitness function to evaluate each candidate S-box
def evaluate_candidate(sbox):
    result = 0
    
    branch = 0
    if BRANCH_WEIGHT > 0:
        branch = calculate_branch_number(sbox)
        branch = scale(branch, get_min_branch_number(SBOX_SIZE), get_max_branch_number(SBOX_SIZE))
    result += BRANCH_WEIGHT * branch
        
    non_linearity = 0
    if NON_LINEARITY_WEIGHT > 0:
        non_linearity = calculate_non_linearity(sbox)
        non_linearity = scale(non_linearity, get_min_non_linearity(SBOX_SIZE), get_max_non_linearity(SBOX_SIZE))
    result += NON_LINEARITY_WEIGHT * non_linearity
        
    differential_uniformity = 0
    if DIFFERENTIAL_UNIFORMITY_WEIGHT > 0:
        differential_uniformity = calculate_differential_uniformity(sbox)
        differential_uniformity = scale(differential_uniformity, get_min_differential_uniformity(SBOX_SIZE), get_max_differential_uniformity(SBOX_SIZE))
    result += DIFFERENTIAL_UNIFORMITY_WEIGHT * differential_uniformity
        
    complexity = 0
    if COMPLEXITY_WEIGHT > 0:
        complexity = calculate_complexity(sbox)
        complexity = scale(complexity, get_max_complexity(SBOX_SIZE), get_min_complexity(SBOX_SIZE))
    result += COMPLEXITY_WEIGHT * complexity
    
    return result


# Evaluate the fitness of the entire population
@lru_cache
def evaluate(population):
    result = [evaluate_candidate(candidate) for candidate in population]
    return result


# Sort the population based on fitness
def sort_population(population):
    return sorted(population, key=evaluate_candidate, reverse=True)


# Mutate a candidate S-box by swapping two elements
def mutate(candidate):
    candidate_mutate = [i for i in candidate]
    i = random.randrange(len(candidate))
    j = random.randrange(len(candidate))
    while (j == i):
        j = random.randrange(len(candidate))
    candidate_mutate[i], candidate_mutate[j] = candidate_mutate[j], candidate_mutate[j]
    return tuple(candidate_mutate)


# Crossover two candidate S-boxes by merging elements
def crossover(candidate1, candidate2):
    s1 = SBox(candidate1)
    s2 = SBox(candidate2)
    result = [s1(s2(i)) for i in range(2 ** SBOX_SIZE)]
    return tuple(result)


# Generate a random candidate S-box
def get_random_candidate(size):
    candidate = [i for i in range(2**size)]
    random.shuffle(candidate)
    return tuple(candidate)


# Evolving a new population
def evolve_new_population(population):
    new_population = []
    population = sort_population(population)
    scores = evaluate(population)
    
    # Keep some of the best candidates
    for i in range(KEEP_BEST):
        new_population.append(population[i])
        
    # Keep some of the worst candidates
    for i in range(KEEP_WORST):
        new_population.append(population[-1 - i])
        
    # Have some random candidates
    for _ in range(KEEP_RANDOM):
        new_population.append(get_random_candidate(SBOX_SIZE))
        
    # Evolve new candidates
    selected_population = population[:SELECTION_SIZE]
    selected_scores = scores[:SELECTION_SIZE]
    while len(new_population) < len(population):
        # Select parents
        pair = random.choices(selected_population, weights = selected_scores, k = 2)
        # Candidate starts as first parent
        candidate = pair[0]
        # Randomly perform a crossover
        if random.random() < CROSSOVER_RATE:
            candidate = crossover(candidate, pair[1])
        # Randomly perform a mutation
        if random.random() < MUTATION_RATE:
            candidate = mutate(candidate)
        new_population.append(candidate)

    return sort_population(new_population)


# Runs the entire evolution process for the required number of generations
def evolve(initial_population):
    population = initial_population
    best = population[0]
    best_count = 0
    calculation_count = 0

    for i in range(MAX_GENERATIONS):
        # Check if the best is already good enough
        if evaluate_candidate(best) >= SATISFACTORY_SCORE:
            logging.debug(
                f"A satisfactory score at or above {SATISFACTORY_SCORE} has been found.")
            break

        # Check if the best hasn't changed for the required number of generations to exit
        if population[0] == best:
            best_count += 1
        else:
            best = population[0]
            best_count = 0
        if best_count > BEST_LIMIT:
            logging.debug(
                f"Best has remained the same for {BEST_LIMIT} generations.")
            break

        # Generate the new population
        population = evolve_new_population(population)

        # Report on progress every REPORT_CALCULATIONS or so evaluations
        calculation_count += POP_SIZE
        if calculation_count >= REPORT_CALCULATION_LIMIT:
            calculation_count = 0
            logging.debug(
                f"{i}/{MAX_GENERATIONS}:\t{population[0]} {evaluate_candidate(population[0]):.2f}")

    return sort_population(population)


# Conversion of a given s-box
# into the lexicographically earliest representative
# function iterates over all possible combinations of i and j
# (which represent the XOR operation),
# and calculates the resulting permutation-XOR.
# The resulting permutation that is lexicographically
# smallest is returned as the canonical PE.def get_canonical(sbox):
def get_canonical_sbox(s):
    """
    Follows Markku-Juhani O. Saarinen's "Cryptographic Analysis of All 4x4-Bit S-Boxes"
    to convert the given s-box into the one that comes lexographically earliest while
    staying in the same permutation-xor equivalence set.
    """
    s = SBox(s)
    canonical = s
    for ci in range(pow(2, s.input_size())):
        for co in range(pow(2, s.input_size())):
            for Pi in Permutations(s.input_size()):
                for Po in Permutations(s.input_size()):
                    modified = list(s)
                    for i in range(len(modified)):
                        i_ci = i ^^ ci
                        tmp_bin = ("0" * s.input_size() +
                                   bin(i_ci)[2:])[-s.input_size():]
                        pi_i_ci = int(''.join(tmp_bin[i - 1] for i in Pi), 2)
                        s_pi_i_ci = s(pi_i_ci)
                        tmp_bin = ("0" * s.input_size() +
                                   bin(s_pi_i_ci)[2:])[-s.input_size():]
                        po_s_pi_i_ci = int(
                            ''.join(tmp_bin[i - 1] for i in Po), 2)
                        po_s_pi_i_ci_co = s_pi_i_ci ^^ co
                        modified[i] = po_s_pi_i_ci_co
                    modified = SBox(modified)
                    if tuple(modified) < tuple(canonical):
                        canonical = modified
    return tuple(canonical)


# Initialise population
logging.info("Generating initial population...")
population = [get_random_candidate(SBOX_SIZE) for i in range(POP_SIZE)]
population = sort_population(population)

logging.info("Starting evolution...")
population = evolve(population)

''' Output results '''
logging.info("Evolution complete.")

worst = population[-1]
logging.info(f"Worst S-Box:\t{worst}\t{evaluate_candidate(worst)}")

best = population[0]
logging.info(f"Best S-Box:\t{best}\t{evaluate_candidate(best)}")

canonical = get_canonical_sbox(best)
logging.info(f"Canonical S-Box:\t{canonical}\t{evaluate_candidate(canonical)}")

