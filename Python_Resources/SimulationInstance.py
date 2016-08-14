"""
@author Konrad Cybulski
@since 14/08/2016
@modified 14/08/2016
Code written utilising pseudocode provided
 in 'Social norms of cooperation in small-scale
 societies' by Santos, Santos, Pacheco
"""
import random
"""
    Static Variables
"""
#              B  G
Strategies = [[0, 0], # AllD
              [1, 0], # pDisc
              [0, 1], # Disc
              [1, 1]] # AllC

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

def Rand():
    return random.uniform(0.0, 1.0)

def U(a, b):
    return random.randint(a, b)

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
    if Rand() < Xerror:
        if Rand() < epsilon and P[x][1 - D[y]]:
            Cx = 1 - P[x][1 - D[y]]
        else:
            Cx = P[x][1 - D[y]]
    else:
        if Rand() < epsilon and P[x][D[y]]:
            Cx = 1 - P[x][D[y]]
        else:
            Cx = P[x][D[y]]
    # Action of Y:
    if Rand() < Xerror:
        if Rand() < epsilon and P[y][1 - D[x]]:
            Cy = 1 - P[y][1 - D[x]]
        else:
            Cy = P[y][1 - D[x]]
    else:
        if Rand() < epsilon and P[y][D[x]]:
            Cy = 1 - P[y][D[x]]
        else:
            Cy = P[y][D[x]]

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

    return (benefit * Cy) - (cost * Cx)

def Simulate():
    for r in range(0, Runs):
        # Initialise random population
        for k in range(0, Z):
            P[k] = Strategies[U(0, 3)]
            D[k] = U(0, 1)
        for t in range(0, Generations):
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
            if Rand() < (1 +





def RunInstance(NumRuns, NumGenerations, PopulationSize, MutationRate,
                 ExecutionError, ReputationAssignmentError,
                 PrivateAssessmentError, ReputationUpdateProbability,
                 RandomSeed, SocialNormMatrix, CostValue, BenefitValue):
    Runs = NumRuns
    Generations = NumGenerations
    Z = PopulationSize
    P = [0]*Z ########## ADAPT FOR NUMPY
    D = [0]*Z ########## ADAPT FOR NUMPY
    mu = MutationRate
    epsilon = ExecutionError
    alpha = ReputationAssignmentError
    Xerror = PrivateAssessmentError
    tau = ReputationUpdateProbability
    randomseed = RandomSeed
    random.seed(randomseed)
    socialnorm = SocialNormMatrix
    cost = CostValue
    benefit = BenefitValue
    Simulate()














