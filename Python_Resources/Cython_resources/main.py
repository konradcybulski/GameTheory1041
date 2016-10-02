import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
import simulation_instance
from InstanceVariables import InstanceVariables

def simulate(runs, generations, population_size, mutation_rate,
             execution_error, reputation_assignment_error,
             private_assessment_error, reputation_update_prob,
             socialnorm, cost, benefit):
    result = simulation_instance.run_instance(runs, generations, population_size, mutation_rate,
                                              execution_error, reputation_assignment_error,
                                              private_assessment_error, reputation_update_prob,
                                              socialnorm, cost, benefit)
    return result

if __name__ == "__main__":
    runs = 1
    generations = 3*np.power(10, 3)
    population_size = 12
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