"""
@author Konrad Cybulski
@since 14/08/2016
@modified 14/08/2016
"""
import numpy as np
import SimulationInstance
import SimulationInstanceVectorized
import SimulationInstanceComms

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
    generations = 3*np.power(10,3)

    mutation_rate = np.power(10*Z, -1)

    execution_error = 0.08
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_probability = 0.2
    randomseed = 1
    socialnorm = SJ
    cost = 1
    benefit = 5

    SimulationInstance.run_instance(runs, generations, Z,
                                    mutation_rate, execution_error, reputation_assignment_error, private_assessment_error,
                                    reputation_update_probability, randomseed, socialnorm,
                                    cost, benefit)


def santos_santos_pacheco_optimized(Z=50):
    runs = 1
    generations = 3*np.power(10,4)

    mutation_rate = np.power(10*Z, -1)

    execution_error = 0.08
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_probability = 0.2
    randomseed = np.random.randint(999999)
    socialnorm = SJ
    cost = 1
    benefit = 5
    SimulationInstanceVectorized.run_instance(runs, generations, Z,
                                              mutation_rate, execution_error, reputation_assignment_error,
                                              private_assessment_error, reputation_update_probability,
                                              randomseed, socialnorm,
                                              cost, benefit)

def santos_santos_pacheco_comms(Z=50, rep_spread_rate= 0.8):
    runs = 10
    generations = 3*np.power(10,3)

    rep_spread_rate = np.power(float(1/Z), float(3/(2*(Z+1))))

    mutation_rate = np.power(10*Z, -1)

    execution_error = 0.08
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_probability = 0.2
    randomseed = 1
    socialnorm = SJ
    cost = 1
    benefit = 5
    SimulationInstanceComms.run_instance(runs, generations, Z,
                                              mutation_rate, execution_error, reputation_assignment_error,
                                              private_assessment_error, reputation_update_probability,
                                              randomseed, socialnorm,
                                              cost, benefit, rep_spread_rate)

if __name__ == '__main__':
    # santos_santos_pacheco()
    santos_santos_pacheco_optimized()
    # santos_santos_pacheco_comms()
