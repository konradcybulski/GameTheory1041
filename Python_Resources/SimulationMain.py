"""
@author Konrad Cybulski
@since 14/08/2016
@modified 14/08/2016
"""
import numpy as np
import time
import multiprocessing
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


def santos_santos_pacheco(runs, Z, socialnorm):
    """
    This class does everything
    :param Z: population size
    :return:
    """

    generations = 3*np.power(10,3)

    mutation_rate = np.power(10*Z, -1)

    execution_error = 0.08
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_probability = 0.2
    randomseed = np.random.randint(999999)
    cost = 1
    benefit = 5

    SimulationInstance.run_instance(runs, generations, Z,
                                    mutation_rate, execution_error, reputation_assignment_error, private_assessment_error,
                                    reputation_update_probability, randomseed, socialnorm,
                                    cost, benefit)


def santos_santos_pacheco_optimized(runs, Z, socialnorm):
    """

    :param runs:
    :param Z: population size
    :param socialnorm:
    :return: cooperation index
    """
    generations = 3*np.power(10,5)

    mutation_rate = float(np.power(float(10*Z), float(-1)))

    execution_error = 0.08
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_probability = 0.2
    randomseed = np.random.randint(999999)
    cost = 1
    benefit = 5
    return SimulationInstanceVectorized.run_instance(runs, generations, Z,
                                              mutation_rate, execution_error, reputation_assignment_error,
                                              private_assessment_error, reputation_update_probability,
                                              randomseed, socialnorm,
                                              cost, benefit)

def santos_santos_pacheco_comms(runs, Z, socialnorm):
    generations = 3*np.power(10,5)

    rep_spread_rate = np.power(float(1/Z), float(3/(2*(Z+1))))

    mutation_rate = float(np.power(float(10*Z), float(-1)))

    execution_error = 0.08
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_probability = 0.2
    randomseed = np.random.randint(999999)
    cost = 1
    benefit = 5
    return SimulationInstanceComms.run_instance(runs, generations, Z,
                                              mutation_rate, execution_error, reputation_assignment_error,
                                              private_assessment_error, reputation_update_probability,
                                              randomseed, socialnorm,
                                              cost, benefit, rep_spread_rate)

def ssp_parallel(runs, Z, socialnorm):
    num_threads = multiprocessing.cpu_count()
    runs_per_thread = int(np.ceil(float(runs) / float(num_threads)))
    pool = multiprocessing.Pool(num_threads)
    results = [pool.apply_async(santos_santos_pacheco_optimized,
                                args=(runs_per_thread, Z, socialnorm)) for i in range(num_threads)]
    cooperation_index = float(0)
    for result in results:
        cooperation_index_i = result.get()
        cooperation_index += float(cooperation_index_i)
    cooperation_index /= float(num_threads)
    return cooperation_index


def ssp_tofile(filename, population_size, socialnorm, socialnorm_string):
    start_sim = time.clock()
    coop_index = ssp_parallel(104, population_size, socialnorm)
    end_sim = time.clock()
    out_string = socialnorm_string + ',' + str(population_size) + ',' + str(coop_index) + ',\n'
    file_out = open(filename, 'a')
    file_out.write(out_string)
    file_out.close()
    print("Z: " + str(population_size) + ', Cooperation Index: ' + str(coop_index) +
          ', completed in ' + str(end_sim - start_sim) + " seconds.")
    pass

if __name__ == '__main__':
    # santos_santos_pacheco()
    start = time.clock()
    ssp_tofile("SSP_results_SternJudging.csv", 10, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 20, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 30, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 40, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 50, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 60, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 70, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 80, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 90, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 100, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 110, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 120, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 130, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 140, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 150, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 175, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 200, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 225, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 250, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 275, SJ, "Stern Judging")
    ssp_tofile("SSP_results_SternJudging.csv", 300, SJ, "Stern Judging")
    end = time.clock()
    print("Simulation completed in " + str(end - start))
    # santos_santos_pacheco_comms()
