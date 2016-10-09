import numpy as np
import Cython_resources.build.simulation_instance as simulation_instance
import Cython_resources.build.simulation_instance_comms as simulation_instance_comms
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
             private_assessment_error, reputation_update_prob, reputation_spread_rate,
             socialnorm, cost, benefit):
    num_threads = multiprocessing.cpu_count()
    runs_per_thread = int(np.ceil(float(runs) / float(num_threads)))
    pool = multiprocessing.Pool(num_threads)
    cooperation_index_average = float(0)
    results = [pool.apply_async(simulation_instance_comms.run_instance,
                                args=(runs, generations, population_size, mutation_rate,
                                      execution_error, reputation_assignment_error,
                                      private_assessment_error, reputation_update_prob, reputation_spread_rate,
                                      socialnorm, cost, benefit)) for _ in range(num_threads)]
    """
    result is the cooperation index
    """
    for result in results:
        cooperation_index_values_i = result.get()
        cooperation_index_average += float(cooperation_index_values_i[0])
        print(cooperation_index_values_i)
    cooperation_index_average /= float(num_threads * runs_per_thread)
    return cooperation_index_average


def simulate_data(filename, runs_per_thread, generations, Z, socialnorm, spread):

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
    results = [pool.apply_async(simulation_instance_comms.run_instance,
                                args=(runs, generations, population_size, mutation_rate,
                                      execution_error, reputation_assignment_error,
                                      private_assessment_error, reputation_update_prob, spread,
                                      np.array(socialnorm), cost, benefit)) for _ in range(num_threads)]
    """
    result is the [cooperation index, Z, socialnorm]
    """
    for result in results:
        cooperation_index_values_i = result.get()
        cooperation_index_average += float(cooperation_index_values_i[0])
        out_string = str(cooperation_index_values_i[0]) + ", Z: " + \
            str(cooperation_index_values_i[1]) + " of " + str(np.ndarray.tolist(cooperation_index_values_i[2])) +\
            ", Spread: " + str(spread)
        file_out = open(filename, 'a')
        file_out.write(out_string + "\n")
        file_out.close()
        print(out_string)
    cooperation_index_average /= float(num_threads)
    # pool.join()
    # pool.terminate()
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
    # reputation_update_prob = 1.0
    # reputation_spread_rate = 1.0
    # socialnorm = np.array([[1, 0], [0, 1]])
    # cost = 1
    # benefit = 5
    #
    # result = simulate_parallel(runs, generations, population_size, mutation_rate,
    #                             execution_error, reputation_assignment_error,
    #                             private_assessment_error, reputation_update_prob,
    #                             reputation_spread_rate,
    #                             socialnorm, cost, benefit)
    # print("Coop_: " + str(result))

    # # Generate Z = 12
    SJ = [[1, 0], [0, 1]]
    SS = [[1, 1], [0, 1]]
    ZERO = [[0, 0], [0, 0]]
    IS = [[1, 1], [0, 0]]

    run_number = 1
    generation_number = 3 * np.power(10, 5)
    simulation_data = [

        ["Data/Spread05.txt", 25, SJ, 0.5],
        ["Data/Spread1.txt", 25, SJ, 1.0],
        ["Data/Spread175.txt", 25, SJ, 1.75],
        ["Data/Spread3.txt", 25, SJ, 3.0],
        ["Data/Spread6.txt", 25, SJ, 6.0],

        ["Data/Spread05.txt", 50, SJ, 0.5],
        ["Data/Spread1.txt", 50, SJ, 1.0],
        ["Data/Spread175.txt", 50, SJ, 1.75],
        ["Data/Spread3.txt", 50, SJ, 3.0],
        ["Data/Spread6.txt", 50, SJ, 6.0],

        ["Data/Spread05.txt", 70, SJ, 0.5],
        ["Data/Spread1.txt", 70, SJ, 1.0],
        ["Data/Spread175.txt", 70, SJ, 1.75],
        ["Data/Spread3.txt", 70, SJ, 3.0],
        ["Data/Spread6.txt", 70, SJ, 6.0],

        ["Data/Spread05.txt", 90, SJ, 0.5],
        ["Data/Spread1.txt", 90, SJ, 1.0],
        ["Data/Spread175.txt", 90, SJ, 1.75],
        ["Data/Spread3.txt", 90, SJ, 3.0],
        ["Data/Spread6.txt", 90, SJ, 6.0],

        ["Data/Spread05.txt", 110, SJ, 0.5],
        ["Data/Spread1.txt", 110, SJ, 1.0],
        ["Data/Spread175.txt", 110, SJ, 1.75],
        ["Data/Spread3.txt", 110, SJ, 3.0],
        ["Data/Spread6.txt", 110, SJ, 6.0],

        # Simple Standing
        ["Data/Spread05.txt", 25, SS, 0.5],
        ["Data/Spread1.txt", 25, SS, 1.0],
        ["Data/Spread175.txt", 25, SS, 1.75],
        ["Data/Spread3.txt", 25, SS, 3.0],
        ["Data/Spread6.txt", 25, SS, 6.0],

        ["Data/Spread05.txt", 50, SS, 0.5],
        ["Data/Spread1.txt", 50, SS, 1.0],
        ["Data/Spread175.txt", 50, SS, 1.75],
        ["Data/Spread3.txt", 50, SS, 3.0],
        ["Data/Spread6.txt", 50, SS, 6.0],

        ["Data/Spread05.txt", 70, SS, 0.5],
        ["Data/Spread1.txt", 70, SS, 1.0],
        ["Data/Spread175.txt", 70, SS, 1.75],
        ["Data/Spread3.txt", 70, SS, 3.0],
        ["Data/Spread6.txt", 70, SS, 6.0],

        ["Data/Spread05.txt", 90, SS, 0.5],
        ["Data/Spread1.txt", 90, SS, 1.0],
        ["Data/Spread175.txt", 90, SS, 1.75],
        ["Data/Spread3.txt", 90, SS, 3.0],
        ["Data/Spread6.txt", 90, SS, 6.0],

        ["Data/Spread05.txt", 110, SS, 0.5],
        ["Data/Spread1.txt", 110, SS, 1.0],
        ["Data/Spread175.txt", 110, SS, 1.75],
        ["Data/Spread3.txt", 110, SS, 3.0],
        ["Data/Spread6.txt", 110, SS, 6.0],

    ]

    for data in simulation_data:
        simulate_data(data[0], run_number, generation_number, data[1], data[2], data[3])
    end = time.clock()
    print("\nSimulation completed in " + str(end - start))
