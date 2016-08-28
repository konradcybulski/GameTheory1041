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

"""
    Static Variables
"""

#              B  G
strategies = [[0, 0],  # AllD
              [1, 0],  # pDisc
              [0, 1],  # Disc
              [1, 1]]  # AllC

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
socialnorm = [[1, 0], [0, 1]]  # matrix determining the reputation dynamic with
# regard to the action taken and the reputation
# of the other agent
cost = 1  # cost defining the payoff matrix cost
benefit = 5  # benefit defined as the payoff matrix benefit

### Tracking Variables
cooperation_index_sum = 0
cooperation_index_min = 1
cooperation_index_max = 0
cooperation_index_average = 0
cooperation_index_zeros = 0


def fitness_function(x, y):
    """
    :param x: the index of agent-x in population
    :param y: the index of agent-y in population
    :return: the fitness of x after
    """
    # Action of X:
    xstrategy = strategies[population[x]]
    if np.random.random() < assessment_error:
        if np.random.random() < execution_error and xstrategy[1 - reputation[y]]:
            cx = 1 - xstrategy[1 - reputation[y]]
        else:
            cx = xstrategy[1 - reputation[y]]
    else:
        if np.random.random() < execution_error and xstrategy[reputation[y]]:
            cx = 1 - xstrategy[reputation[y]]
        else:
            cx = xstrategy[reputation[y]]
    # Action of Y:
    ystrategy = strategies[population[y]]
    if np.random.random() < assessment_error:
        if np.random.random() < execution_error and ystrategy[1 - reputation[x]]:
            cy = 1 - ystrategy[1 - reputation[x]]
        else:
            cy = ystrategy[1 - reputation[x]]
    else:
        if np.random.random() < execution_error and ystrategy[reputation[x]]:
            cy = 1 - ystrategy[reputation[x]]
        else:
            cy = ystrategy[reputation[x]]

    # Update Reputation of X:
    if np.random.random() < reputation_update_rate:
        if np.random.random() < reputation_assignment_error:
            reputation[x] = 1 - socialnorm[1 - cx][1 - reputation[y]]
        else:
            reputation[x] = socialnorm[1 - cx][1 - reputation[y]]
    # Update Reputation of Y:
    if np.random.random() < reputation_update_rate:
        if np.random.random() < reputation_assignment_error:
            reputation[y] = 1 - socialnorm[1 - cy][1 - reputation[x]]
        else:
            reputation[y] = socialnorm[1 - cy][1 - reputation[x]]
    # Track cooperation
    global cooperation_index_sum
    global cooperation_index_min
    global cooperation_index_max
    global cooperation_index_zeros
    coops_x = 1 if cx == 1 else 0
    coops_y = 1 if cy == 1 else 0
    cur_cooperation_index = float(float(coops_y + coops_x)/float(2))
    cooperation_index_sum += cur_cooperation_index
    cooperation_index_min = min(cooperation_index_min, cur_cooperation_index)
    cooperation_index_max = max(cooperation_index_max, cur_cooperation_index)
    if cur_cooperation_index < float(np.power(float(10), float(-5))):
        cooperation_index_zeros += 1
    return (benefit * cy) - (cost * cx)


def simulate():
    for r in range(0, runs):

        # Initialise random population
        global population
        global reputation
        population = np.random.randint(4, size=population_size)  # equivalent to U(0, 3)
        reputation = np.random.randint(2, size=population_size)  # equivalent to U(0, 1)

        for t in range(0, generations):
            # Update progress
            # if t % (generations // 10) == 0:
            #     progress = (float((t + 1) * 100) / float(generations))
            #     print("Simulation progress: %d%%     \r" % progress)

            index_to_mutate = np.random.randint(population_size)

            # Random mutation probability
            if np.random.random() < mutation_probability:
                population[index_to_mutate] = np.random.randint(4)

            # Make sure B != A
            b = np.random.randint(population_size)
            while b == index_to_mutate:
                b = np.random.randint(population_size)

            fitness_a = 0
            fitness_b = 0
            for i in range(0, 2 * population_size):
                c = np.random.randint(population_size)
                while c == index_to_mutate:
                    c = np.random.randint(population_size)
                # Update Fitness of A and Reputation of A & C
                fitness_a += fitness_function(index_to_mutate, c)

                c = np.random.randint(population_size)
                while c == b:
                    c = np.random.randint(population_size)
                # Update Fitness of B and Reputation of B & C
                fitness_b += fitness_function(b, c)
            fitness_a /= (2 * population_size)
            fitness_b /= (2 * population_size)
            if np.random.random() < np.power(1 + np.exp(fitness_a - fitness_b), -1):
                population[index_to_mutate] = population[b]
    global cooperation_index_sum
    global cooperation_index_average
    cooperation_index_average = float(cooperation_index_sum)/float(runs*generations*4*population_size)


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
    socialnorm = SocialNormMatrix
    cost = CostValue
    benefit = BenefitValue

    ### Reset tracking variables
    global cooperation_index_sum
    global cooperation_index_average
    global cooperation_index_min
    global cooperation_index_max
    global cooperation_index_zeros
    cooperation_index_min = 1
    cooperation_index_max = 0
    cooperation_index_zeros = 0
    cooperation_index_sum = 0
    cooperation_index_average = 0

    # start = time.clock()
    # print("Simulation beginning...")

    simulate()
    return_list = [cooperation_index_average,
                   cooperation_index_min,
                   cooperation_index_max,
                   float(cooperation_index_zeros) / float(runs * generations * 4 * population_size),
                   float(cooperation_index_sum) / float((runs *
                                                         generations * 4 * population_size) - cooperation_index_zeros)]
    return return_list # float(cooperation_index_average)
    # end = time.clock()
    # print("Simulation completed in " + str(end - start))
