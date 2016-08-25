"""
@author Konrad Cybulski
@since 14/08/2016
@modified 16/08/2016
Code written utilising pseudocode provided
 in 'Social norms of cooperation in small-scale
 societies' by Santos, Santos, Pacheco
"""
import numpy as np
import time
from numba import jit

"""
    Static Variables
"""

#              B  G
strategies = np.array([[0, 0],  # AllD
                      [1, 0],  # pDisc
                      [0, 1],  # Disc
                      [1, 1]])  # AllC

"""
    Simulation Variables
"""
runs = 0  # number of runs
generations = 0  # number of generations
population_size = 0  # population size
population = []  # vector of all individual strategies
# population[k] : strategy of individual k
# population[k] = [0,0], [1,0], [0,1] or [1,1]
reputation = []  # vector of all individual public reputations
# reputation[k] : public reputation of individual k
# reputation[k] = 0 or 1
mutation_probability = 0  # mutation probability
execution_error = 0  # execution error probability
reputation_assignment_error = 0  # reputation assignment error probability
assessment_error = 0  # private assessment error probability
reputation_update_rate = 0  # reputation update probability
randomseed = 0  # seed used to generate randomness
socialnorm = np.array([[1, 0], [0, 1]])  # matrix determining the reputation dynamic with
# regard to the action taken and the reputation
# of the other agent
cost = 1  # cost defining the payoff matrix cost
benefit = 5  # benefit defined as the payoff matrix benefit

### Tracking Variables
cooperation_count = 0
interaction_count = 0


def fitness_function(x, y_array):
    """
    :param x: the index of agent-x in population
    :param y: the array of indices of agent-y's in population
    :return: the fitness of x after
    """
    # Action of X:
    arr_len = y_array.size

    p_rep = reputation[y_array]

    x_strategy = strategies[population[x]]
    xactionbad = x_strategy[0]
    xactiongood = x_strategy[1]
    """
        Action of X with errors
    """
    cx = xactiongood*p_rep + xactionbad*(1 - p_rep)

    # Adjust for assessment error
    elements_to_change_assessment_error = np.int(arr_len * assessment_error)
    mask_assessment_error = np.random.randint(arr_len, size=elements_to_change_assessment_error)
    cx[mask_assessment_error] = (xactiongood*(1 - p_rep[mask_assessment_error]) + xactionbad*p_rep[mask_assessment_error])

    # Adjust for execution error
    elements_to_change_execution_error = np.int(arr_len * execution_error)
    mask_execution_error = np.random.randint(arr_len, size=elements_to_change_execution_error)
    cx[mask_execution_error] = 0

    """
        Update Reputation of X with errors
    """
    reputation_x_vector = np.insert(socialnorm[(1 - cx, 1 - p_rep)], 0, reputation[x])

    # Reputation update rate:
    elements_to_change_reputation_update_rate = np.int(arr_len * reputation_update_rate)
    mask_reputation_update_rate = np.random.randint(1, arr_len,
                                                    size=elements_to_change_reputation_update_rate)
    reputation_x_vector[mask_reputation_update_rate] = reputation_x_vector[mask_reputation_update_rate - 1]

    # Reputation assignment error:
    elements_to_change_reputation_assignment_error = np.int(arr_len * reputation_assignment_error)
    mask_reputation_assignment_error = np.random.randint(
        1, arr_len, size=elements_to_change_reputation_assignment_error)
    reputation_x_vector[mask_reputation_assignment_error] = 1 - reputation_x_vector[mask_reputation_assignment_error]

    reputation[x] = reputation_x_vector[len(reputation_x_vector)-1]
    mask = np.ones(reputation_x_vector.shape, dtype=bool)
    mask[arr_len] = False
    reputation_x_vector = reputation_x_vector[mask]

    # Action of Y:
    pstratindex = population[y_array]
    pstrategy = strategies[pstratindex]
    pactionbad = pstrategy[:, 0]
    pactiongood = pstrategy[:, 1]

    """
        Action of Y with errors
    """
    cy = np.multiply(reputation_x_vector, pactiongood) + np.multiply((1 - reputation_x_vector), pactionbad)
    # Adjust for assessment error
    elements_to_change_assessment_error = np.int(arr_len * assessment_error)
    mask_assessment_error = np.random.randint(arr_len, size=elements_to_change_assessment_error)
    cy[mask_assessment_error] = np.multiply(pactiongood[mask_assessment_error],
                                            (1 - reputation_x_vector[mask_assessment_error])) +\
                                np.multiply(pactionbad[mask_assessment_error],
                                            reputation_x_vector[mask_assessment_error])
    # Adjust for execution error
    elements_to_change_execution_error = np.int(arr_len * execution_error)
    mask_execution_error = np.random.randint(arr_len, size=elements_to_change_execution_error)
    cy[mask_execution_error] = 0

    """
        Update Reputation of Y with errors
    """
    reputation_y_vector = socialnorm[(1 - cy, 1 - reputation_x_vector)]
    # Reputation update rate:
    elements_to_change_reputation_update_rate = np.int(arr_len * reputation_update_rate)
    mask_reputation_update_rate = np.random.randint(arr_len, size=elements_to_change_reputation_update_rate)
    reputation_y_vector[mask_reputation_update_rate] = reputation[y_array[mask_reputation_update_rate]]
    # Reputation assignment error:
    elements_to_change_reputation_assignment_error = np.int(arr_len * reputation_assignment_error)
    mask_reputation_assignment_error = np.random.randint(
        arr_len, size=elements_to_change_reputation_assignment_error)
    reputation_y_vector[mask_reputation_assignment_error] = 1 - reputation_y_vector[mask_reputation_assignment_error]

    # Track cooperation
    # global interaction_count
    # global cooperation_count
    coops_y = np.sum(cy)
    coops_x = np.sum(cx)
    # interaction_count += 2 * arr_len
    # cooperation_count += coops_y + coops_x

    return (benefit * coops_y) - (cost * coops_x)


def simulate():
    for r in range(0, runs):

        # Initialise random population
        global population
        global reputation
        population = np.random.randint(4, size=population_size)  # equivalent to U(0, 3)
        reputation = np.random.randint(2, size=population_size)  # equivalent to U(0, 1)

        for t in range(0, generations):
            # Update progress
            if t % (generations // 10) == 0:
                progress = (float((t + 1) * 100) / float(generations))
                print("Simulation progress: %d%%     \r" % progress)

            agent_one = np.random.randint(population_size)

            # Random mutation probability
            if np.random.random() < mutation_probability:
                population[agent_one] = strategies[np.random.randint(4)]

            # Make sure B != A
            agent_two = np.random.randint(population_size)
            while agent_two == agent_one:
                agent_two = np.random.randint(population_size)

            #### Creating tournament arrays
            tournament_sample = np.random.randint(population_size, size=2*population_size)
            fitness_a = fitness_function(agent_one, tournament_sample)
            fitness_b = fitness_function(agent_two, tournament_sample)

            fitness_a /= (2 * population_size)
            fitness_b /= (2 * population_size)
            if np.random.random() < np.power(1 + np.exp(fitness_a - fitness_b), -1):
                population[agent_one] = population[agent_two]
    # print("Cooperation index: " + str(float(cooperation_count) / float(interaction_count)))


def run_instance(NumRuns, NumGenerations, PopulationSize, MutationRate,
                 ExecutionError, ReputationAssignmentError,
                 PrivateAssessmentError, ReputationUpdateProbability,
                 RandomSeed, SocialNormMatrix, CostValue, BenefitValue):
    global runs
    global generations
    global population_size
    global population
    global reputation
    global mutation_probability
    global execution_error
    global reputation_assignment_error
    global assessment_error
    global reputation_update_rate
    global randomseed
    global socialnorm
    global cost
    global benefit
    runs = NumRuns
    generations = NumGenerations
    population_size = PopulationSize

    mutation_probability = MutationRate
    execution_error = ExecutionError
    reputation_assignment_error = ReputationAssignmentError
    assessment_error = PrivateAssessmentError
    reputation_update_rate = ReputationUpdateProbability
    randomseed = RandomSeed
    np.random.seed(randomseed)
    socialnorm = np.array(SocialNormMatrix)
    cost = CostValue
    benefit = BenefitValue

    ### Reset tracking variables
    global cooperation_count
    global interaction_count
    cooperation_count = 0
    interaction_count = 0

    start = time.clock()
    print("Simulation beginning...")

    simulate()
    end = time.clock()
    print("Simulation completed in " + str(end - start))
