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
    results = [pool.apply_async(SantosSantosPacheco.simulation,
                                args=(runs_per_thread, generations, Z, socialnorm)) for i in range(num_threads)]
    """
    result is the cooperation index
    """
    cooperation_index_average = float(0)
    for result in results:
        cooperation_index_values_i = result.get()
        cooperation_index_average += float(cooperation_index_values_i)
        print(cooperation_index_values_i)
    cooperation_index_average /= float(num_threads)
    return cooperation_index_average


def ssp_tofile(filename, population_size, socialnorm, theoretical_index):
    start_sim = time.clock()
    coop_index = ssp_parallel(8, 3*np.power(10, 6), population_size, socialnorm)
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
                 str(socialnorm) + ',' +\
                 str(theoretical_index) + ',' +\
                 str(coop_index) + ',\n'
    file_out = open(filename, 'a')
    file_out.write(out_string)
    file_out.close()
    print("Z: " + str(population_size) +
          ", Cooperation Index: " + str(coop_index))


if __name__ == '__main__':
    start = time.clock()
    # GC, GD, BC, BD ==> GC, BC, GD, BD
    # Rule 0: [0, 0, 0, 0]: 0.007790514478896741
    # Rule 1: [0, 0, 0, 1]: 0.4627661993687701
    # Rule 2: [0, 0, 1, 0]: 0.06161330969353635
    # Rule 3: [0, 0, 1, 1]: 0.0005849304509235106
    # Rule 4: [0, 1, 0, 0]: 0.0043600195271226265
    # Rule 5: [0, 1, 0, 1]: 0.07095791170280523
    # Rule 6: [0, 1, 1, 0]: 0.05197578035307538
    # Rule 7: [0, 1, 1, 1]: 0.06161330969353021
    # Rule 8: [1, 0, 0, 0]: 0.04719576688988282
    # Rule 9: [1, 0, 0, 1]: 0.816033606679371
    # Rule 10: [1, 0, 1, 0]: 0.07095791170280075
    # Rule 11: [1, 0, 1, 1]: 0.4627661993697064
    # Rule 12: [1, 1, 0, 0]: 0.000584930450923339
    # Rule 13: [1, 1, 0, 1]: 0.047195766889946984
    # Rule 14: [1, 1, 1, 0]: 0.004360019527122341
    # Rule 15: [1, 1, 1, 1]: 0.0077905144788752686
    institutions = [
        [[[1, 0], [0, 1]], 0.816033606679371],
        [[[0, 0], [0, 1]], 0.4627661993687701],
        [[[0, 0], [0, 0]], 0.007790514478896741]
    ]
    for institution_info in institutions:
        for _ in range(1):
            ssp_tofile("Data/SSP_results_Z12_alt.csv", 12, institution_info[0], institution_info[1])
    end = time.clock()
    print("Simulation completed in " + str(end - start))
