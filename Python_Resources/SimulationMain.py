"""
@author Konrad Cybulski
@since 14/08/2016
@modified 14/08/2016
"""
import numpy as np
import time
import multiprocessing
import SantosSantosPacheco

def ssp_parallel(runs, generations, Z, socialnorm):
    num_threads = multiprocessing.cpu_count()
    runs_per_thread = int(np.ceil(float(runs) / float(num_threads)))
    pool = multiprocessing.Pool(num_threads)
    cooperation_index_average = float(0)
    for i in range(runs_per_thread):
        results = [pool.apply_async(SantosSantosPacheco.simulation,
                                    args=(runs_per_thread, generations, Z, socialnorm)) for _ in range(num_threads)]
        """
        Result is the cooperation index
        """
        for result in results:
            cooperation_index_values_i = result.get()
            cooperation_index_average += float(cooperation_index_values_i)
            print(cooperation_index_values_i)
    cooperation_index_average /= float(num_threads*runs_per_thread)
    return cooperation_index_average


def ssp_tofile(filename, population_size, socialnorm):
    coop_index = ssp_parallel(104, 3*np.power(10, 5), population_size, socialnorm)
    """
    result is in the form:
        [cooperation_index_avg,
         cooperation_index_min,
         cooperation_index_max,
         cooperation_index_zero_proportion,
         cooperation_index_without_zeros]
    """
    out_string = str(population_size) + ',' +\
                 str(coop_index) + ',\n'
    file_out = open(filename, 'a')
    file_out.write(out_string)
    file_out.close()
    print("Z: " + str(population_size) +
          ", Cooperation Index: " + str(coop_index))


if __name__ == '__main__':
    start = time.clock()

    SJ = [[1, 0], [0, 1]]
    SS = [[1, 1], [0, 1]]
    ZERO = [[0, 0], [0, 0]]
    IS = [[1, 1], [0, 0]]

    constant_assessment_folder = "Data/ConstantAssessmentError/"

    simulation_data = [

        # Z = 5
        [constant_assessment_folder + "Z5_Data.txt", 5, SJ],
        [constant_assessment_folder + "Z5_Data.txt", 5, SS],
        [constant_assessment_folder + "Z5_Data.txt", 5, ZERO],
        [constant_assessment_folder + "Z5_Data.txt", 5, IS],

        # Z = 12
        [constant_assessment_folder + "Z12_Data.txt", 12, SJ],
        [constant_assessment_folder + "Z12_Data.txt", 12, SS],
        [constant_assessment_folder + "Z12_Data.txt", 12, ZERO],
        [constant_assessment_folder + "Z12_Data.txt", 12, IS],

        # Z = 25
        [constant_assessment_folder + "Z25_Data.txt", 25, SJ],
        [constant_assessment_folder + "Z25_Data.txt", 25, SS],
        [constant_assessment_folder + "Z25_Data.txt", 25, ZERO],
        [constant_assessment_folder + "Z25_Data.txt", 25, IS],

    ]

    for data in simulation_data:
        ssp_tofile(data[0], data[1], data[2])
    end = time.clock()
    print("\nSimulation completed in " + str(end - start))