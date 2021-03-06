import numpy as np
import Cython_resources.build.simulation_instance as simulation_instance
import time
import multiprocessing


def simulate_tofile(filename, runs_per_thread, generations, Z, socialnorm):

    runs = runs_per_thread
    population_size = Z
    mutation_rate = float(np.power(float(10 * Z), float(-1)))
    execution_error = 0.08
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_prob = 1.0
    cost = 1
    benefit = 5
    num_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_threads)
    cooperation_index_average = float(0)
    results = [pool.apply_async(simulation_instance.run_instance,
                                args=(runs, generations, population_size, mutation_rate,
                                      execution_error, reputation_assignment_error,
                                      private_assessment_error, reputation_update_prob,
                                      np.array(socialnorm), cost, benefit)) for _ in range(num_threads)]
    """
    result is the [cooperation index, Z, socialnorm]
    """
    for result in results:
        cooperation_index_values_i = result.get()
        cooperation_index_average += float(cooperation_index_values_i[0])
        out_string = str(cooperation_index_values_i[0]) + ", Z: " + \
            str(cooperation_index_values_i[1]) + " of " + str(np.ndarray.tolist(cooperation_index_values_i[2]))
        file_out = open(filename, 'a')
        file_out.write(out_string + "\n")
        file_out.close()
        print(out_string)
    cooperation_index_average /= float(num_threads)
    pool.close()
    return cooperation_index_average

if __name__ == '__main__':
    start = time.clock()

    SJ = [[1, 0], [0, 1]]
    SS = [[1, 1], [0, 1]]
    ZERO = [[0, 0], [0, 0]]
    IS = [[1, 1], [0, 0]]

    constant_assessment_folder = "Data/ConstantAssessmentError/"

    run_number = 100
    generation_number = 3 * np.power(10, 5)
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
        simulate_tofile(data[0], run_number, generation_number, data[1], data[2])
    end = time.clock()
    print("\nSimulation completed in " + str(end - start))
