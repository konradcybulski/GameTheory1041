import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
import simulation_instance
from InstanceVariables import InstanceVariables

def fitness_function_pure(x, y, variables):
    Z = variables.population_size

    fitness_x = 0
    fitness_y = 0

    probabilities_x = np.ones(Z, dtype=np.float) / float(Z - 1)
    probabilities_x[x] = 0
    probabilities_y = np.ones(Z, dtype=np.float) / float(Z - 1)
    probabilities_y[y] = 0
    return 1

if __name__ == "__main__":
    runs = 1
    generations = 3*np.power(10, 5)
    population_size = 50
    mutation_rate = float(np.power(float(10*population_size), float(-1)))
    execution_error = 0.08
    reputation_assignment_error = 0.01
    private_assessment_error = 0.01
    reputation_update_prob = 1
    socialnorm = np.array([[1, 0], [0, 1]])
    cost = 1
    benefit = 5
    result = simulation_instance.run_instance(runs, generations, population_size, mutation_rate,
                                  execution_error, reputation_assignment_error,
                                  private_assessment_error, reputation_update_prob,
                                  socialnorm, cost, benefit)
    print(result)