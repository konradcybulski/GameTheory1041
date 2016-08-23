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
CooperationCount = 0
InteractionCount = 0


def FitnessFunction(x, y):
    """
    :param x: the index of agent-x in population
    :param y: the index of agent-y in population
    :return: the fitness of x after
    """
    # Action of X:
    XStrategy = strategies[population[x]]
    if np.random.random() < assessment_error:
        if np.random.random() < execution_error and XStrategy[1 - reputation[y]]:
            Cx = 1 - XStrategy[1 - reputation[y]]
        else:
            Cx = XStrategy[1 - reputation[y]]
    else:
        if np.random.random() < execution_error and XStrategy[reputation[y]]:
            Cx = 1 - XStrategy[reputation[y]]
        else:
            Cx = XStrategy[reputation[y]]
    # Action of Y:
    YStrategy = strategies[population[y]]
    if np.random.random() < assessment_error:
        if np.random.random() < execution_error and YStrategy[1 - reputation[x]]:
            Cy = 1 - YStrategy[1 - reputation[x]]
        else:
            Cy = YStrategy[1 - reputation[x]]
    else:
        if np.random.random() < execution_error and YStrategy[reputation[x]]:
            Cy = 1 - YStrategy[reputation[x]]
        else:
            Cy = YStrategy[reputation[x]]

    # Update Reputation of X:
    if np.random.random() < reputation_update_rate:
        if np.random.random() < reputation_assignment_error:
            reputation[x] = 1 - socialnorm[1 - Cx][1 - reputation[y]]  #ReputationFunction(socialnorm, Cx, reputation[y])
        else:
            reputation[x] = socialnorm[1 - Cx][1 - reputation[y]]  #ReputationFunction(socialnorm, Cx, reputation[y])
    # Update Reputation of Y:
    if np.random.random() < reputation_update_rate:
        if np.random.random() < reputation_assignment_error:
            reputation[y] = 1 - socialnorm[1 - Cy][1 - reputation[x]]  #ReputationFunction(socialnorm, Cy, reputation[x])
        else:
            reputation[y] = socialnorm[1 - Cy][1 - reputation[x]]  #ReputationFunction(socialnorm, Cy, reputation[x])
    ### Track cooperation
    global InteractionCount
    global CooperationCount
    InteractionCount += 2
    CooperationCount += 1 if Cx == 1 else 0
    CooperationCount += 1 if Cy == 1 else 0
    return (benefit * Cy) - (cost * Cx)


def Simulate():
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

            index_to_mutate = np.random.randint(population_size)

            # Random mutation probability
            if np.random.random() < mutation_probability:
                population[index_to_mutate] = strategies[np.random.randint(4)]

            # Make sure B != A
            b = np.random.randint(population_size)
            while b == index_to_mutate:
                b = np.random.randint(population_size)

            Fa = 0
            Fb = 0
            for i in range(0, 2 * population_size):
                c = np.random.randint(population_size)
                while c == index_to_mutate:
                    c = np.random.randint(population_size)
                # Update Fitness of A and Reputation of A & C
                Fa += FitnessFunction(index_to_mutate, c)

                c = np.random.randint(population_size)
                while c == b:
                    c = np.random.randint(population_size)
                # Update Fitness of B and Reputation of B & C
                Fb += FitnessFunction(b, c)
            Fa /= (2 * population_size)
            Fb /= (2 * population_size)
            if np.random.random() < np.power(1 + np.exp(Fa - Fb), -1):
                population[index_to_mutate] = population[b]
    print("Cooperation index: " + str(float(CooperationCount) / float(InteractionCount)))


def RunInstance(NumRuns, NumGenerations, PopulationSize, MutationRate,
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
    socialnorm = SocialNormMatrix
    cost = CostValue
    benefit = BenefitValue

    ### Reset tracking variables
    global CooperationCount
    global InteractionCount
    CooperationCount = 0
    InteractionCount = 0

    start = time.clock()
    print("Simulation beginning...")

    Simulate()
    end = time.clock()
    print("Simulation completed in " + str(end - start))
