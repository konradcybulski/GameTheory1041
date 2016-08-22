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
Z = 0  # population size
population = []  # vector of all individual strategies
# population[k] : strategy of individual k
# population[k] = [0,0], [1,0], [0,1] or [1,1]
reputation = []  # vector of all individual public reputations
# reputation[k] : public reputation of individual k
# reputation[k] = 0 or 1
mu = 0  # mutation probability
epsilon = 0  # execution error probability
alpha = 0  # reputation assignment error probability
Xerror = 0  # private assessment error probability
tau = 0  # reputation update probability
randomseed = 0  # seed used to generate randomness
socialnorm = [[1, 0], [0, 1]]  # matrix determining the reputation dynamic with
# regard to the action taken and the reputation
# of the other agent
cost = 1  # cost defining the payoff matrix cost
benefit = 5  # benefit defined as the payoff matrix benefit

### Tracking Variables
CooperationCount = 0
InteractionCount = 0


def Rand():
    return np.random.random()


def U(a, b):
    return np.random.randint(a, b + 1)


def ReputationFunction(socialnorm_matrix, action_x, rep_y):
    """
    :param socialnorm_matrix: a 2x2 matrix defining the assigned
            reputation given an action and reputation of the
            other agent. Follows the form:
                         G   B
            Cooperate:[[ G   B ]
            Defect:    [ B   G ]]
            Given that G == 1 and B == 0
    :param action_x: the action of agent x (1 : cooperate,
            0 : defect)
    :param rep_y: the reputation of agent y (1 : G, 0 : B)
    :return: the new reputation of agent x given their action
            on agent y with a given reputation.
    """
    return socialnorm_matrix[1 - action_x][1 - rep_y]


def FitnessFunction(x, y):
    """
    :param x: the index of agent-x in population
    :param y: the index of agent-y in population
    :return: the fitness of x after
    """
    # Action of X:
    XStrategy = strategies[population[x]]
    if Rand() < Xerror:
        if Rand() < epsilon and XStrategy[1 - reputation[y]]:
            Cx = 1 - XStrategy[1 - reputation[y]]
        else:
            Cx = XStrategy[1 - reputation[y]]
    else:
        if Rand() < epsilon and XStrategy[reputation[y]]:
            Cx = 1 - XStrategy[reputation[y]]
        else:
            Cx = XStrategy[reputation[y]]
    # Action of Y:
    YStrategy = strategies[population[y]]
    if Rand() < Xerror:
        if Rand() < epsilon and YStrategy[1 - reputation[x]]:
            Cy = 1 - YStrategy[1 - reputation[x]]
        else:
            Cy = YStrategy[1 - reputation[x]]
    else:
        if Rand() < epsilon and YStrategy[reputation[x]]:
            Cy = 1 - YStrategy[reputation[x]]
        else:
            Cy = YStrategy[reputation[x]]

    # Update Reputation of X:
    if Rand() < tau:
        if Rand() < alpha:
            reputation[x] = 1 - ReputationFunction(socialnorm, Cx, reputation[y])
        else:
            reputation[x] = ReputationFunction(socialnorm, Cx, reputation[y])
    # Update Reputation of Y:
    if Rand() < tau:
        if Rand() < alpha:
            reputation[y] = 1 - ReputationFunction(socialnorm, Cy, reputation[x])
        else:
            reputation[y] = ReputationFunction(socialnorm, Cy, reputation[x])
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
        population = np.random.randint(4, size=Z)  # equivalent to U(0, 3)
        reputation = np.random.randint(2, size=Z)  # equivalent to U(0, 1)

        for t in range(0, generations):
            # Update progress
            if t % (generations // 100) == 0:
                progress = (float((t + 1) * 100) / float(generations))
                print("Simulation progress: %d%%     \r" % progress)
            # sys.stdout.flush()
            # sys.stdout.write("Simulation progress: %d%%     \r" % ((t+1)*100/generations))

            index_to_mutate = U(0, Z - 1)

            # Random mutation probability
            if Rand() < mu:
                population[index_to_mutate] = strategies[U(0, 3)]

            # Make sure B != A
            b = U(0, Z - 1)
            while b == index_to_mutate:
                b = U(0, Z - 1)

            Fa = 0
            Fb = 0
            for i in range(0, 2 * Z):
                c = U(0, Z - 1)
                while c == index_to_mutate:
                    c = U(0, Z - 1)
                # Update Fitness of A and Reputation of A & C
                Fa += FitnessFunction(index_to_mutate, c)

                c = U(0, Z - 1)
                while c == b:
                    c = U(0, Z - 1)
                # Update Fitness of B and Reputation of B & C
                Fb += FitnessFunction(b, c)
            Fa /= (2 * Z)
            Fb /= (2 * Z)
            if Rand() < np.power(1 + np.exp(Fa - Fb), -1):
                population[index_to_mutate] = population[b]
    print("Cooperation index: " + str(float(CooperationCount) / float(InteractionCount)))


def RunInstance(NumRuns, NumGenerations, PopulationSize, MutationRate,
                ExecutionError, ReputationAssignmentError,
                PrivateAssessmentError, ReputationUpdateProbability,
                RandomSeed, SocialNormMatrix, CostValue, BenefitValue):
    global runs
    global generations
    global Z
    global population
    global reputation
    global mu
    global epsilon
    global alpha
    global Xerror
    global tau
    global randomseed
    global socialnorm
    global cost
    global benefit
    runs = NumRuns
    generations = NumGenerations
    Z = PopulationSize

    mu = MutationRate
    epsilon = ExecutionError
    alpha = ReputationAssignmentError
    Xerror = PrivateAssessmentError
    tau = ReputationUpdateProbability
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
