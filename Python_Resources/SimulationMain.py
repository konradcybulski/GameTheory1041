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
    cooperation_index_average = float(0)
    for i in range(runs_per_thread):
        results = [pool.apply_async(SantosSantosPacheco.simulation,
                                    args=(runs_per_thread, generations, Z, socialnorm)) for i in range(num_threads)]
        """
        result is the cooperation index
        """
        for result in results:
            cooperation_index_values_i = result.get()
            cooperation_index_average += float(cooperation_index_values_i)
            print(cooperation_index_values_i)
    cooperation_index_average /= float(num_threads*runs_per_thread)
    return cooperation_index_average


def ssp_tofile(filename, population_size, socialnorm):
    start_sim = time.clock()
    coop_index = ssp_parallel(104, 3*np.power(10, 5), population_size, socialnorm)
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
                 str(coop_index) + ',\n'
    file_out = open(filename, 'a')
    file_out.write(out_string)
    file_out.close()
    print("Z: " + str(population_size) +
          ", Cooperation Index: " + str(coop_index))


if __name__ == '__main__':
    start = time.clock()
    coop_index = ssp_parallel(16, 3*np.power(10, 5), 12, [[1, 0], [0, 1]])
    print("Z: " + str(12) +
          ", Cooperation Index: " + str(coop_index))
    end = time.clock()
    print("Simulation completed in " + str(end - start))

    print("_____________________________________\n")
    start = time.clock()
    coop_index = ssp_parallel(16, 3*np.power(10, 5), 50, [[1, 0], [0, 1]])
    print("Z: " + str(50) +
          ", Cooperation Index: " + str(coop_index))
    end = time.clock()
    print("Simulation completed in " + str(end - start))
