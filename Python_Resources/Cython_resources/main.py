import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
import simulation_instance
from InstanceVariables import InstanceVariables
import time
import multiprocessing

def simulate(runs, generations, population_size, mutation_rate,
             execution_error, reputation_assignment_error,
             private_assessment_error, reputation_update_prob,
             socialnorm, cost, benefit):
    result = simulation_instance.run_instance(runs, generations, population_size, mutation_rate,
                                              execution_error, reputation_assignment_error,
                                              private_assessment_error, reputation_update_prob,
                                              socialnorm, cost, benefit)
    return result

def simulate_parallel(runs, generations, population_size, mutation_rate,
             execution_error, reputation_assignment_error,
             private_assessment_error, reputation_update_prob,
             socialnorm, cost, benefit):
    num_threads = multiprocessing.cpu_count()
    runs_per_thread = int(np.ceil(float(runs) / float(num_threads)))
    pool = multiprocessing.Pool(num_threads)
    cooperation_index_average = float(0)
    for i in range(runs_per_thread):
        results = [pool.apply_async(simulation_instance.run_instance,
                                    args=(runs, generations, population_size, mutation_rate,
                                          execution_error, reputation_assignment_error,
                                          private_assessment_error, reputation_update_prob,
                                          socialnorm, cost, benefit)) for _ in range(num_threads)]
        """
        result is the cooperation index
        """
        for result in results:
            cooperation_index_values_i = result.get()
            cooperation_index_average += float(cooperation_index_values_i)
            print(cooperation_index_values_i)
    cooperation_index_average /= float(num_threads * runs_per_thread)
    return cooperation_index_average


def simulate_data(filename, runs_per_thread, generations, Z, socialnorm):

    runs = runs_per_thread
    population_size = Z
    mutation_rate = float(np.power(float(10 * population_size), float(-1)))
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
    return cooperation_index_average

if __name__ == '__main__':
    start = time.clock()
    # runs = 1
    # generations = 3*np.power(10, 3)
    # population_size = 12
    # mutation_rate = float(np.power(float(10*population_size), float(-1)))
    # execution_error = 0.08
    # reputation_assignment_error = 0.01
    # private_assessment_error = 0.01
    # reputation_update_prob = 1
    # socialnorm = np.array([[1, 0], [0, 1]])
    # cost = 1
    # benefit = 5
    #
    # result = simulate(runs, generations, population_size, mutation_rate,
    #                            execution_error, reputation_assignment_error,
    #                            private_assessment_error, reputation_update_prob,
    #                            socialnorm, cost, benefit)
    # print("Coop_: " + str(result))

    # Generate Z = 12
    SJ = [[1, 0], [0, 1]]
    SS = [[1, 1], [0, 1]]
    ZERO = [[0, 0], [0, 0]]
    IS = [[1, 1], [0, 0]]

    run_number = 1
    generation_number = 3 * np.power(10, 4)
    simulation_data = [
        ["Z5_Data.txt", 5, SS],
        ["Z5_Data.txt", 5, SJ],
        ["Z5_Data.txt", 5, ZERO],
        ["Z5_Data.txt", 5, IS],

        ["Z12_Data.txt", 12, SS],
        ["Z12_Data.txt", 12, SJ],
        ["Z12_Data.txt", 12, ZERO],
        ["Z12_Data.txt", 12, IS],

        ["Z25_Data.txt", 25, SS],
        ["Z25_Data.txt", 25, SJ],
        ["Z25_Data.txt", 25, ZERO],
        ["Z25_Data.txt", 25, IS],

        ["Z50_Data.txt", 50, SS],
        ["Z50_Data.txt", 50, SJ],
        ["Z50_Data.txt", 50, ZERO],
        ["Z50_Data.txt", 50, IS],
    ]

    for data in simulation_data:
        simulate_data(data[0], run_number, generation_number, data[1], data[2])
    end = time.clock()
    print("\nSimulation completed in " + str(end - start))
