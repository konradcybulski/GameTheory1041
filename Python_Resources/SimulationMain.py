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
    cooperation_indexes = []
    for result in results:
        cooperation_index_values_i = result.get()
        cooperation_indexes.append(float(cooperation_index_values_i))
        print(cooperation_index_values_i)
    return cooperation_indexes


def ssp_tofile(filename, population_size, socialnorm, theoretical_index):
    start_sim = time.clock()
    coop_indexes = ssp_parallel(8, 3*np.power(10, 5), population_size, socialnorm)
    """
    result is in the form:
        [cooperation_index_avg,
         cooperation_index_min,
         cooperation_index_max,
         cooperation_index_zero_proportion,
         cooperation_index_without_zeros]
    """
    end_sim = time.clock()
    file_out = open(filename, 'a')
    for coop_index in coop_indexes:
        out_string = str(population_size) + ',' +\
                     str(socialnorm) + ',' +\
                     str(theoretical_index) + ',' +\
                     str(coop_index) + ',\n'
        file_out.write(out_string)
    file_out.close()
    print("Z: " + str(population_size) +
          ", Cooperation Index: " + str(np.average(coop_indexes)))


if __name__ == '__main__':
    start = time.clock()
    # GC, GD, BC, BD ==> GC, BC, GD, BD
    institutions = [
        [[0, 0], [0, 0]],
        # [[0, 0], [0, 1]],
        # [[0, 1], [0, 0]],
        # [[0, 1], [0, 1]],
        # [[0, 0], [1, 0]],
        # [[0, 0], [1, 1]],
        # [[0, 1], [1, 0]],
        # [[0, 1], [1, 1]],
        # [[1, 0], [0, 0]],
        # [[1, 0], [0, 1]],
        # [[1, 1], [0, 0]],
        # [[1, 1], [0, 1]],
        # [[1, 0], [1, 0]],
        # [[1, 0], [1, 1]],
        # [[1, 1], [1, 0]],
        # [[1, 1], [1, 1]]
    ]
    theoretical_Z12 = [
        0.007790514478896741,
        # 0.4627661993687701,
        # 0.06161330969353635,
        # 0.0005849304509235106,
        # 0.0043600195271226265,
        # 0.07095791170280523,
        # 0.05197578035307538,
        # 0.06161330969353021,
        # 0.04719576688988282,
        # 0.816033606679371,
        # 0.07095791170280075,
        # 0.4627661993697064,
        # 0.000584930450923339,
        # 0.047195766889946984,
        # 0.004360019527122341,
        # 0.007790514478875268
    ]
    institutions_Z50 = [
        0.004885658144033963,
        # 0.4923273034325223,
        # 0.006870340397749519,
        # 9.031340098879267e-12,
        # 1.578455107243566e-05,
        # 0.006825503964836176,
        # 5.535122147388484e-05,
        # 0.006870340397746998,
        # 0.07083386985149893,
        # 0.8224949855474589,
        # 0.006825503964834948,
        # 0.49232730343821446,
        # 9.031340098549926e-12,
        # 0.07083386985154855,
        # 1.5784551072303234e-05,
        # 0.004885658144036075
    ]
    for i in range(len(institutions)):
        for _ in range(1):
            ssp_tofile("Data/SSP_results_Z50.csv", 50, institutions[i], institutions_Z50[i])
    end = time.clock()
    print("Simulation completed in " + str(end - start))
