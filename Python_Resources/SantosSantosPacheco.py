import numpy as np
import time
import multiprocessing
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
    :return:
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
    randomseed = np.random.randint(999999)
    cost = 1
    benefit = 5
    return SimulationInstanceVectorized.run_instance(runs, generations, Z,
                                              mutation_rate, execution_error, reputation_assignment_error,
                                              private_assessment_error, reputation_update_probability,
                                              randomseed, socialnorm,
                                              cost, benefit)

"""
Generation Information
"""

def simulation_optimized_generation_information(runs, generations, Z, socialnorm,
                                    generation_data_save_wait, generation_data_save_filename):
    """

    :param runs:
    :param Z: population size
    :param socialnorm:
    :return: cooperation index
    """

    mutation_rate = float(np.power(float(10*Z), float(-1)))

    execution_error = 0.01
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_probability = 0.2
    randomseed = np.random.randint(999999)
    cost = 1
    benefit = 5
    return SimulationInstanceVectorized.run_instance_generation_information(runs, generations, Z,
                                              mutation_rate, execution_error, reputation_assignment_error,
                                              private_assessment_error, reputation_update_probability,
                                              randomseed, socialnorm,
                                              cost, benefit,
                                              generation_data_save_wait, generation_data_save_filename)

def simulation_generation_information(runs, generations, Z, socialnorm,
                                    generation_data_save_wait, generation_data_save_filename):
    """

    :param runs:
    :param Z: population size
    :param socialnorm:
    :return: cooperation index
    """

    mutation_rate = float(np.power(float(10*Z), float(-1)))

    execution_error = 0.01
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_probability = 0.2
    randomseed = np.random.randint(999999)
    cost = 1
    benefit = 5
    return SimulationInstance.run_instance_generation_information(runs, generations, Z,
                                              mutation_rate, execution_error, reputation_assignment_error,
                                              private_assessment_error, reputation_update_probability,
                                              randomseed, socialnorm,
                                              cost, benefit,
                                              generation_data_save_wait, generation_data_save_filename)