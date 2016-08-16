"""
@author Konrad Cybulski
@since 14/08/2016
@modified 16/08/2016
Code written utilising pseudocode provided
 in 'Social norms of cooperation in small-scale
 societies' by Santos, Santos, Pacheco
"""
import numpy as np
from numba import jit, int8, int32, int64
import multiprocessing
import threading
from multiprocessing import Process, Pool
import time
import sys
import queue
"""
    Static Variables
"""
#              B  G
Strategies = [[0, 0],  # AllD
              [1, 0],  # pDisc
              [0, 1],  # Disc
              [1, 1]]  # AllC

"""
    Simulation Variables
"""
Runs = 0                    # number of runs
Generations = 0             # number of generations
Z = 0                       # population size
P = []                      # vector of all individual strategies
                                # P[k] : strategy of individual k
                                # P[k] = [0,0], [1,0], [0,1] or [1,1]
D = []                      # vector of all individual public reputations
                                # D[k] : public reputation of individual k
                                # D[k] = 0 or 1
mu = 0                      # mutation probability
epsilon = 0                 # execution error probability
alpha = 0                   # reputation assignment error probability
Xerror = 0                       # private assessment error probability
tau = 0                     # reputation update probability
randomseed = 0              # seed used to generate randomness
socialnorm = [[1,0],[0,1]]  # matrix determining the reputation dynamic with
                                # regard to the action taken and the reputation
                                # of the other agent
cost = 1                    # cost defining the payoff matrix cost
benefit = 5                 # benefit defined as the payoff matrix benefit

### Tracking Variables
CooperationCount = 0
InteractionCount = 0

def Rand():
    return np.random.random()


def U(a, b):
    return np.random.randint(a, b+1)


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
    :param x: the index of agent-x in P
    :param y: the index of agent-y in P
    :return: the fitness of x after
    """
    # Action of X:
    XStrategy = Strategies[P[x]]
    if Rand() < Xerror:
        if Rand() < epsilon and XStrategy[1 - D[y]]:
            Cx = 1 - XStrategy[1 - D[y]]
        else:
            Cx = XStrategy[1 - D[y]]
    else:
        if Rand() < epsilon and XStrategy[D[y]]:
            Cx = 1 - XStrategy[D[y]]
        else:
            Cx = XStrategy[D[y]]
    # Action of Y:
    YStrategy = Strategies[P[y]]
    if Rand() < Xerror:
        if Rand() < epsilon and YStrategy[1 - D[x]]:
            Cy = 1 - YStrategy[1 - D[x]]
        else:
            Cy = YStrategy[1 - D[x]]
    else:
        if Rand() < epsilon and YStrategy[D[x]]:
            Cy = 1 - YStrategy[D[x]]
        else:
            Cy = YStrategy[D[x]]

    # Update Reputation of X:
    if Rand() < tau:
        if Rand() < alpha:
            D[x] = 1 - ReputationFunction(socialnorm, Cx, D[y])
        else:
            D[x] = ReputationFunction(socialnorm, Cx, D[y])
    # Update Reputation of Y:
    if Rand() < tau:
        if Rand() < alpha:
            D[y] = 1 - ReputationFunction(socialnorm, Cy, D[x])
        else:
            D[y] = ReputationFunction(socialnorm, Cy, D[x])
    ### Track cooperation
    global InteractionCount
    global CooperationCount
    InteractionCount += 2
    CooperationCount += 1 if Cx == 1 else 0
    CooperationCount += 1 if Cy == 1 else 0
    return (benefit * Cy) - (cost * Cx)


def Simulate():
    for r in range(0, Runs):

        # Initialise random population
        global P
        global D
        P = np.random.randint(4, size=Z)  # equivalent to U(0, 3)
        D = np.random.randint(2, size=Z)  # equivalent to U(0, 1)

        for t in range(0, Generations):
            # Update progress
            if t % (Generations//100) == 0:
                progress = (float((t+1)*100)/float(Generations))
                print("Simulation progress: %d%%     \r" % progress)
            #sys.stdout.flush()
            #sys.stdout.write("Simulation progress: %d%%     \r" % ((t+1)*100/Generations))

            a = U(0, Z-1)
            # Random mutation probability
            if Rand() < mu:
                P[a] = Strategies[U(0, 3)]
            # Make sure B != A
            b = U(0, Z-1)
            while b == a:
                b = U(0, Z-1)

            Fa = 0
            Fb = 0
            for i in range(0, 2*Z):
                c = U(0, Z-1)
                while c == a:
                    c = U(0, Z-1)
                # Update Fitness of A and Reputation of A & C
                Fa += FitnessFunction(a, c)

                c = U(0, Z-1)
                while c == b:
                    c = U(0, Z-1)
                # Update Fitness of B and Reputation of B & C
                Fb += FitnessFunction(b, c)
            Fa /= (2*Z)
            Fb /= (2*Z)
            if Rand() < np.power(1 + np.exp(Fa-Fb),-1):
                P[a] = P[b]
    print("Cooperation index: " + str(float(CooperationCount)/float(InteractionCount)))

def RunInstance(NumRuns, NumGenerations, PopulationSize, MutationRate,
                 ExecutionError, ReputationAssignmentError,
                 PrivateAssessmentError, ReputationUpdateProbability,
                 RandomSeed, SocialNormMatrix, CostValue, BenefitValue):
    global Runs
    global Generations
    global Z
    global P
    global D
    global mu
    global epsilon
    global alpha
    global Xerror
    global tau
    global randomseed
    global socialnorm
    global cost
    global benefit
    Runs = NumRuns
    Generations = NumGenerations
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
