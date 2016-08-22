"""
@author Konrad Cybulski
@since 14/08/2016
@modified 14/08/2016
"""
import numpy as np
import SimulationInstance
import SimulationInstanceOptimized

# STERN JUDGING
SJ = [[1, 0],
      [0, 1]]

# SIMPLE STANDING
SS = [[1, 1],
      [0, 1]]

# SHUNNING
SH = [[1, 0],
      [0, 0]]

# IMAGE SCORING
IS = [[1, 1],
      [0, 0]]


def santos_santos_pacheco(Z=50):
    """
    This class does everything
    :param Z: population size
    :return:
    """

    runs = 1
    generations = 3*np.power(10,5)

    mu = np.power(10*Z, -1)

    epsilon = 0.08
    alpha = 0.01
    Xerror = 0.01
    tau = 0.2
    randomseed = 1
    socialnorm = SJ
    cost = 1
    benefit = 5

    SimulationInstance.RunInstance(runs, generations, Z,
                                   mu, epsilon, alpha, Xerror,
                                   tau, randomseed, socialnorm,
                                   cost, benefit)


def santos_santos_pacheco_optimized(Z=50):
    Runs = 1
    Generations = 3*np.power(10,5)

    mu = np.power(10*Z, -1)
    epsilon = 0.08
    alpha = 0.01
    Xerror = 0.01
    tau = 0.2
    randomseed = 1
    socialnorm = SJ
    cost = 1
    benefit = 5
    SimulationInstanceOptimized.RunInstance(Runs, Generations, Z,
                                   mu, epsilon, alpha, Xerror,
                                   tau, randomseed, socialnorm,
                                   cost, benefit)

if __name__ == '__main__':
    # santos_santos_pacheco()
    santos_santos_pacheco_optimized()
