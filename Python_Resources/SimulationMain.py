"""
@author Konrad Cybulski
@since 14/08/2016
@modified 14/08/2016
"""
import numpy as np
import SimulationInstance
import SimulationInstanceOptimized

SJ = [[1, 0],
      [0, 1]]
SS = [[1, 1],
      [0, 1]]
SH = [[1, 0],
      [0, 0]]
IS = [[1, 1],
      [0, 0]]


def SantosSantosPacheco(Z=50):
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
    SimulationInstance.RunInstance(Runs, Generations, Z,
                                   mu, epsilon, alpha, Xerror,
                                   tau, randomseed, socialnorm,
                                   cost, benefit)

def SantosSantosPachecoOptimized(Z=50):
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
    # SantosSantosPacheco()
    SantosSantosPachecoOptimized()
