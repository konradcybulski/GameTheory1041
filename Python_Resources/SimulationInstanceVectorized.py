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
CooperationCount = 0
InteractionCount = 0


def FitnessFunction(x, y_array):
    """
    :param x: the index of agent-x in population
    :param y: the array of indices of agent-y's in population
    :return: the fitness of x after
    """
    # Action of X:
    arr_len = len(y_array)

    Prep = reputation[y_array]

    XStrategy = strategies[population[x]]
    XActionBad = XStrategy[0]
    XActionGood = XStrategy[1]
    """
        Action of X with errors
    """
    Cx =  XActionGood*Prep + XActionBad*(1 - Prep)
    # Adjust for assessment error
    elements_to_change_assessment_error = int(arr_len * assessment_error)
    mask_assessment_error = np.random.randint(arr_len, size=elements_to_change_assessment_error)
    Cx[mask_assessment_error] = (XActionGood*(1 - Prep[mask_assessment_error]) + XActionBad*Prep[mask_assessment_error])
    # Adjust for execution error
    elements_to_change_execution_error = int(arr_len * execution_error)
    mask_execution_error = np.random.randint(arr_len, size=elements_to_change_execution_error)
    Cx[mask_execution_error] = 0

    """
        Update Reputation of X with errors
    """
    reputation_x_vector = np.insert(socialnorm[(1 - Cx, 1 - Prep)], 0, reputation[x])
    # Reputation update rate:
    elements_to_change_reputation_update_rate = int(arr_len * reputation_update_rate)
    mask_reputation_update_rate = np.random.randint(1, arr_len, size=elements_to_change_reputation_update_rate)
    reputation_x_vector[mask_reputation_update_rate] = reputation_x_vector[mask_reputation_update_rate - 1]
    # Reputation assignment error:
    elements_to_change_reputation_assignment_error = int(arr_len * reputation_assignment_error)
    mask_reputation_assignment_error = np.random.randint(1, arr_len, size=elements_to_change_reputation_assignment_error)
    reputation_x_vector[mask_reputation_assignment_error] = 1 - reputation_x_vector[mask_reputation_assignment_error]

    reputation[x] = reputation_x_vector[len(reputation_x_vector)-1]
    mask = np.ones(reputation_x_vector.shape, dtype=bool)
    mask[arr_len] = False
    reputation_x_vector = reputation_x_vector[mask] #, len(reputation_x_vector)-1)

    # Action of Y:
    PStratIndex = population[y_array]
    PStrategy = strategies[PStratIndex]
    PActionBad = PStrategy[:, 0]
    PActionGood = PStrategy[:, 1]

    # print(reputation_x_vector)
    # print(PActionGood)

    """
        Action of Y with errors
    """
    Cy = np.multiply(reputation_x_vector, PActionGood) + np.multiply((1 - reputation_x_vector), PActionBad)
    # Adjust for assessment error
    elements_to_change_assessment_error = int(arr_len * assessment_error)
    mask_assessment_error = np.random.randint(arr_len, size=elements_to_change_assessment_error)
    Cy[mask_assessment_error] = np.multiply(PActionGood[mask_assessment_error],
                                            (1 - reputation_x_vector[mask_assessment_error])) +\
                                np.multiply(PActionBad[mask_assessment_error],
                                             reputation_x_vector[mask_assessment_error])
    # Adjust for execution error
    elements_to_change_execution_error = int(arr_len * execution_error)
    mask_execution_error = np.random.randint(arr_len, size=elements_to_change_execution_error)
    Cy[mask_execution_error] = 0

    """
        Update Reputation of Y with errors
    """
    reputation_y_vector = socialnorm[(1 - Cy, 1 - reputation_x_vector)]
    # Reputation update rate:
    elements_to_change_reputation_update_rate = int(arr_len * reputation_update_rate)
    mask_reputation_update_rate = np.random.randint(arr_len, size=elements_to_change_reputation_update_rate)
    reputation_y_vector[mask_reputation_update_rate] = reputation[y_array[mask_reputation_update_rate]]
    # Reputation assignment error:
    elements_to_change_reputation_assignment_error = int(arr_len * reputation_assignment_error)
    mask_reputation_assignment_error = np.random.randint(arr_len, size=elements_to_change_reputation_assignment_error)
    reputation_y_vector[mask_reputation_assignment_error] = 1 - reputation_y_vector[mask_reputation_assignment_error]

    ### Track cooperation
    global InteractionCount
    global CooperationCount
    CoopsY = np.sum(Cy)
    CoopsX = np.sum(Cx)
    InteractionCount += 2 * arr_len
    CooperationCount += CoopsY + CoopsX

    return (benefit * CoopsY) - (cost * CoopsX)


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
            Fa = FitnessFunction(agent_one, tournament_sample)
            Fb = FitnessFunction(agent_two, tournament_sample)

            Fa /= (2 * population_size)
            Fb /= (2 * population_size)
            if np.random.random() < np.power(1 + np.exp(Fa - Fb), -1):
                population[agent_one] = population[agent_two]
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
    socialnorm = np.array(SocialNormMatrix)
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
