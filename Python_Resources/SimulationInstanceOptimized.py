"""
@author Konrad Cybulski
@since 16/08/2016
@modified 16/08/2016
Code written utilising pseudocode provided
 in 'Social norms of cooperation in small-scale
 societies' by Santos, Santos, Pacheco
"""
import numpy as np
from numba import jit, int8, int32, int64, float32
import time
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
# Runs = 0                    # number of runs
# Generations = 0             # number of generations
# Z = 0                       # population size
# P = []                      # vector of all individual strategies
#                                 # P[k] : strategy of individual k
#                                 # P[k] = [0,0], [1,0], [0,1] or [1,1]
# D = []                      # vector of all individual public reputations
#                                 # D[k] : public reputation of individual k
#                                 # D[k] = 0 or 1
# mu = 0                      # mutation probability
# epsilon = 0                 # execution error probability
# alpha = 0                   # reputation assignment error probability
# Xerror = 0                       # private assessment error probability
# tau = 0                     # reputation update probability
# randomseed = 0              # seed used to generate randomness
# socialnorm = [[1,0],[0,1]]  # matrix determining the reputation dynamic with
#                                 # regard to the action taken and the reputation
#                                 # of the other agent
# cost = 1                    # cost defining the payoff matrix cost
# benefit = 5                 # benefit defined as the payoff matrix benefit


def Rand():
    return np.random.random()


def U(a, b):
    return np.random.randint(a, b+1)


#@jit(int8(int8[::], int8, int8), nopython=True, nogil=True)
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
    return socialnorm_matrix[1 - action_x, 1 - rep_y]


@jit(int8(int8, int8, int8[:], int8[:], int64[::], int8[:], int8[::],
          int8, int8, float32, float32, float32, float32), nogil=True)
def FitnessFunction(x, y, P, D, Strategies, CoopList, socialnorm, cost, benefit, Xerror, epsilon, tau, alpha):
    # """
    # :param x: the index of agent-x in P
    # :param y: the index of agent-y in P
    # :return: the fitness of x after
    # """
    # Action of X:
    XStrategy = Strategies[P[x]]
    if np.random.rand() < Xerror:
        if np.random.rand() < epsilon and XStrategy[1 - D[y]] == 1:
            Cx = 1 - XStrategy[1 - D[y]]
        else:
            Cx = XStrategy[1 - D[y]]
    else:
        if np.random.rand() < epsilon and XStrategy[D[y]] == 1:
            Cx = 1 - XStrategy[D[y]]
        else:
            Cx = XStrategy[D[y]]
    # Action of Y:
    YStrategy = Strategies[P[y]]
    if np.random.rand() < Xerror:
        if np.random.rand() < epsilon and YStrategy[1 - D[x]] == 1:
            Cy = 1 - YStrategy[1 - D[x]]
        else:
            Cy = YStrategy[1 - D[x]]
    else:
        if np.random.rand() < epsilon and YStrategy[D[x]] == 1:
            Cy = 1 - YStrategy[D[x]]
        else:
            Cy = YStrategy[D[x]]

    # Update Reputation of X:
    if np.random.rand() < tau:
        if np.random.rand() < alpha:
            D[x] = 1 - socialnorm[1 - Cx, 1 - D[y]]#ReputationFunction(socialnorm, Cx, D[y])
        else:
            D[x] = socialnorm[1 - Cx, 1 - D[y]]#ReputationFunction(socialnorm, Cx, D[y])
    # Update Reputation of Y:
    if np.random.rand() < tau:
        if np.random.rand() < alpha:
            D[y] = 1 - socialnorm[1 - Cy, 1 - D[x]]#ReputationFunction(socialnorm, Cy, D[x])
        else:
            D[y] = socialnorm[1 - Cy, 1 - D[x]]#ReputationFunction(socialnorm, Cy, D[x])
    ### Track cooperation
    CoopList[0] += 2
    CoopList[1] += 1 if Cx == 1 else 0
    CoopList[1] += 1 if Cy == 1 else 0
    return (benefit * Cy) - (cost * Cx)


@jit((int64, int64, int64, float32, float32, float32,
      float32, float32, float32, int8[::], int8, int8, int8[::]), nopython=True, nogil=True)
def Simulate(Runs, Generations, Z, mu, epsilon, alpha,
             Xerror, tau, randomseed, socialnorm, cost, benefit, Strategies):
    # first element is # of interactions, second is cooperations
    CoopList = np.array([0, 0], dtype=int64)
    np.random.seed(randomseed)
    for r in range(0, Runs):

        # Initialise random population
        P = np.random.randint(4, size=Z)  # equivalent to U(0, 3)
        D = np.random.randint(2, size=Z)  # equivalent to U(0, 1)

        for t in range(0, Generations):
            # Update progress
            if t % (Generations//100) == 0:
                print(t)
                progress = (float((t+1)*100)/float(Generations))
                # print("Simulation progress: %d%%     \r" % progress)

            a = np.random.randint(0, Z)
            # Random mutation probability
            if np.random.rand() < mu:
                P[a] = Strategies[np.random.randint(0, 4)]
            # Make sure B != A
            b = np.random.randint(0, Z)
            while b == a:
                b = np.random.randint(0, Z)

            Fa = 0
            Fb = 0
            for i in range(0, 2*Z):
                c = np.random.randint(0, Z)
                while c == a:
                    c = np.random.randint(0, Z)
                # Update Fitness of A and Reputation of A & C
                Fa += FitnessFunction(a, c, P, D, Strategies, CoopList,
                                      socialnorm, cost, benefit, Xerror, epsilon, tau, alpha)

                c = np.random.randint(0, Z)
                while c == b:
                    c = np.random.randint(0, Z)
                # Update Fitness of B and Reputation of B & C
                Fb += FitnessFunction(b, c, P, D, Strategies, CoopList,
                                      socialnorm, cost, benefit, Xerror, epsilon, tau, alpha)
            Fa /= (2*Z)
            Fb /= (2*Z)
            if np.random.rand() < np.power(1 + np.exp(Fa-Fb), -1):
                P[a] = P[b]
    # print("Cooperation index: ")
    print(float(CoopList[1])/float(CoopList[0]))

def RunInstance(NumRuns, NumGenerations, PopulationSize, MutationRate,
                 ExecutionError, ReputationAssignmentError,
                 PrivateAssessmentError, ReputationUpdateProbability,
                 RandomSeed, SocialNormMatrix, CostValue, BenefitValue):
    # global Runs
    # global Generations
    # global Z
    # global P
    # global D
    # global mu
    # global epsilon
    # global alpha
    # global Xerror
    # global tau
    # global randomseed
    # global socialnorm
    # global cost
    # global benefit
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
    socialnorm = np.array([[1, 0],
                            [0, 1]], dtype=int8[:])  #np.array(SocialNormMatrix)
    cost = CostValue
    benefit = BenefitValue
    Strategies = np.array([[0, 0],  # AllD
                          [1, 0],  # pDisc
                          [0, 1],  # Disc
                          [1, 1]], dtype=int8[:])  # AllC

    start = time.clock()
    print("Simulation beginning...")

    Simulate(Runs, Generations, Z, mu, epsilon, alpha,
             Xerror, tau, randomseed, socialnorm, cost, benefit, Strategies)
    end = time.clock()
    print("Simulation completed in " + str(end - start))
