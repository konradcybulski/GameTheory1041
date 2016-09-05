"""
@author Konrad Cybulski
@since 14/08/2016
@modified 14/08/2016
"""
import numpy as np
import time
import multiprocessing
import SantosSantosPacheco


def ssp_parallel_generation_information(Z, generations, socialnorm, data_delay, socialnorm_name):
    num_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_threads)
    results = [pool.apply_async(SantosSantosPacheco.simulation_optimized_generation_information,
                                args=(1, generations, Z, socialnorm, data_delay,
                                "SSP_" + socialnorm_name + "_Z" +
                                      str(Z) + "_" + str(i+1) + ".csv")) for i in range(num_threads)]
    for result in results:
        result.get()


def ssp_parallel(runs, generations, Z, socialnorm):
    num_threads = multiprocessing.cpu_count()
    runs_per_thread = int(np.ceil(float(runs) / float(num_threads)))
    pool = multiprocessing.Pool(num_threads)
    results = [pool.apply_async(SantosSantosPacheco.simulation_optimized,
                                args=(runs_per_thread, generations, Z, socialnorm)) for i in range(num_threads)]
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
        print(cooperation_index_values_i[0])
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
    coop_index_values = ssp_parallel(104, 3*np.power(10, 5), population_size, socialnorm)
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
    start = time.clock()
    coop_index_values = ssp_parallel(8, 3*np.power(10, 4), 12, [[0, 0], [0, 1]])
    print("Z: " + str(12) +
          ", Cooperation Index: " + str(coop_index_values[0]) +
          ", Min: " + str(coop_index_values[1]) +
          ", Max: " + str(coop_index_values[2]) +
          ", Zero proportion: " + str(coop_index_values[3]) +
          ", CoopIndx without zeros: " +
          str(coop_index_values[4]))
    coop_index_values = ssp_parallel(8, 3*np.power(10, 4), 12, [[1, 1], [0, 1]])
    print("Z: " + str(12) +
          ", Cooperation Index: " + str(coop_index_values[0]) +
          ", Min: " + str(coop_index_values[1]) +
          ", Max: " + str(coop_index_values[2]) +
          ", Zero proportion: " + str(coop_index_values[3]) +
          ", CoopIndx without zeros: " +
          str(coop_index_values[4]))
    # ssp_tofile("SSP_results_SternJudging.csv", 5, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 5, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 10, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 10, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 20, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 20, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 30, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 30, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 40, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 40, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 50, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 50, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 60, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 60, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 70, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 70, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 80, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 80, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 90, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 90, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 100, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 100, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 110, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 110, SantosSantosPacheco.SS)
    #
    # ssp_tofile("SSP_results_SternJudging.csv", 120, SantosSantosPacheco.SJ)
    # ssp_tofile("SSP_results_SimpleStanding.csv", 120, SantosSantosPacheco.SS)
    end = time.clock()
    print("Simulation completed in " + str(end - start))
