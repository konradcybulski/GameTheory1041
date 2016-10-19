import numpy as np
import SimulationInstance
import SimulationInstanceVectorized

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

"""
Cooperation Index functions
"""

def simulation(runs, generations, Z, socialnorm):
    """
    :param Z: population size
    :return: cooperation index of simulation
    """

    mutation_rate = float(np.power(float(10*Z), float(-1)))

    execution_error = 0.08
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_probability = 1
    cost = 1
    benefit = 5

    return SimulationInstance.simulate(runs, generations, Z,
                                       mutation_rate, execution_error,
                                       reputation_assignment_error, private_assessment_error,
                                       reputation_update_probability, socialnorm,
                                       cost, benefit)


def simulation_optimized(runs, generations, Z, socialnorm):
    """

    :param runs:
    :param Z: population size
    :param socialnorm:
    :return: cooperation index
    """

    mutation_rate = float(np.power(float(10*Z), float(-1)))

    execution_error = 0.08
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_probability = 1
    cost = 1
    benefit = 5
    return SimulationInstanceVectorized.run_instance(runs, generations, Z,
                                                     mutation_rate, execution_error, reputation_assignment_error,
                                                     private_assessment_error, reputation_update_probability,
                                                     socialnorm, cost, benefit)
