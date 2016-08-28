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
    :param Z: population size
    :return:
    """

    generations = 3*np.power(10,4)

    mutation_rate = float(np.power(float(10*Z), float(-1)))

    execution_error = 0.08
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_probability = 0.2
    randomseed = np.random.randint(999999)
    cost = 1
    benefit = 5

    return SimulationInstance.run_instance(runs, generations, Z,
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
    generations = 3*np.power(10, 5)

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
    """
    result is in the form:
            [cooperation_index_avg,
            cooperation_index_min,
            cooperation_index_max,
            cooperation_index_zero_proportion,
            cooperation_index_without_zeros]
    """
    cooperation_index_average = float(0)
    cooperation_index_min = float(1)
    cooperation_index_max = float(0)
    cooperation_index_zero_proportion = float(0)
    cooperation_index_without_zeros = float(0)
    for result in results:
        cooperation_index_values_i = result.get()
        cooperation_index_average += float(cooperation_index_values_i[0])
        cooperation_index_min = min(cooperation_index_min, cooperation_index_values_i[1])
        cooperation_index_max = max(cooperation_index_max, cooperation_index_values_i[2])
        cooperation_index_zero_proportion += float(cooperation_index_values_i[3])
        cooperation_index_without_zeros += float(cooperation_index_values_i[4])
    cooperation_index_average /= float(num_threads)
    cooperation_index_zero_proportion /= float(num_threads)
    cooperation_index_without_zeros /= float(num_threads)
    return [cooperation_index_average,
            cooperation_index_min,
            cooperation_index_max,
            cooperation_index_zero_proportion,
            cooperation_index_without_zeros]


def ssp_tofile(filename, population_size, socialnorm):
    start_sim = time.clock()
    coop_index_values = ssp_parallel(32, population_size, socialnorm)
    """
    result is in the form:
        [cooperation_index_avg,
         cooperation_index_min,
         cooperation_index_max,
         cooperation_index_zero_proportion,
         cooperation_index_without_zeros]
    """
    end_sim = time.clock()
    out_string = str(population_size) + ',' +\
                 str(coop_index_values[0]) + ',' +\
                 str(coop_index_values[1]) + ',' +\
                 str(coop_index_values[2]) + ',' +\
                 str(coop_index_values[3]) + ',' +\
                 str(coop_index_values[4]) + ',\n'
    file_out = open(filename, 'a')
    file_out.write(out_string)
    file_out.close()
    print("Z: " + str(population_size) +
          ", Cooperation Index: " + str(coop_index_values[0]) +
          ", Min: " + str(coop_index_values[1]) +
          ", Max: " + str(coop_index_values[2]) +
          ", Zero proportion: " + str(coop_index_values[3]) +
          ", CoopIndx without zeros: " +
          str(coop_index_values[4]))
    pass

if __name__ == '__main__':
    # santos_santos_pacheco()
    start = time.clock()
    ssp_tofile("SSP_results_SternJudging.csv", 5, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 5, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 5, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 5, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 5, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 5, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 10, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 10, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 10, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 10, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 10, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 10, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 20, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 20, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 20, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 20, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 20, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 20, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 30, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 30, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 30, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 30, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 30, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 30, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 40, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 40, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 40, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 40, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 40, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 40, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 50, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 50, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 50, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 50, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 50, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 50, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 60, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 60, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 60, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 60, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 60, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 60, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 70, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 70, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 70, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 70, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 70, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 70, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 80, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 80, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 80, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 80, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 80, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 80, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 90, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 90, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 90, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 90, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 90, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 90, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 100, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 100, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 100, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 100, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 100, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 100, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 110, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 110, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 110, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 110, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 110, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 110, SS)

    ssp_tofile("SSP_results_SternJudging.csv", 120, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 120, SJ)
    ssp_tofile("SSP_results_SternJudging.csv", 120, SJ)
    ssp_tofile("SSP_results_SimpleStanding.csv", 120, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 120, SS)
    ssp_tofile("SSP_results_SimpleStanding.csv", 120, SS)
    end = time.clock()
    print("Simulation completed in " + str(end - start))
    # santos_santos_pacheco_comms()
