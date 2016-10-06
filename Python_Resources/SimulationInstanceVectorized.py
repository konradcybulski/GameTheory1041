"""
@author Konrad Cybulski
@since 14/08/2016
@modified 16/08/2016
Code written utilising pseudocode provided
 in 'Social norms of cooperation in small-scale
 societies' by Santos, Santos, Pacheco
"""
import numpy as np
import time
import multiprocessing
from collections import Counter

from InstanceVariables import InstanceVariables

"""
    Static Variables
"""

#              B  G
strategies = np.array([[0, 0],  # AllD
                      [1, 0],  # pDisc
                      [0, 1],  # Disc
                      [1, 1]])  # AllC


def fitness_function(x, y_array, variables):
    """
    :param x: the index of agent-x in population
    :param y_array: the array of indices of agent-y's in population
    :return: the fitness of x after
    """
    # Action of X:
    arr_len = y_array.size

    p_rep = variables.reputation[y_array]

    x_strategy = strategies[variables.population[x]]
    xactionbad = x_strategy[0]
    xactiongood = x_strategy[1]

    """
        Action of X with errors
    """
    cx = xactiongood*p_rep + xactionbad*(1 - p_rep)

    # Adjust for assessment error
    assessment_error_changes = np.random.binomial(arr_len, variables.assessment_error)
    mask_assessment_error = np.random.randint(arr_len, size=assessment_error_changes)
    cx[mask_assessment_error] = (xactiongood*(1 - p_rep[mask_assessment_error]) +
                                 xactionbad*p_rep[mask_assessment_error])

    # Adjust for execution error
    elements_to_change_execution_error = np.random.binomial(arr_len, variables.execution_error)
    mask_execution_error = np.random.randint(arr_len, size=elements_to_change_execution_error)
    cx[mask_execution_error] = 0

    """
        Update Reputation of X with errors
    """
    reputation_x_vector = np.insert(variables.socialnorm[(1 - cx, 1 - p_rep)], 0, variables.reputation[x])

    # Reputation update rate:
    elements_to_change_reputation_update_rate = np.random.binomial(arr_len, (float(1) - variables.reputation_update_rate))
    mask_reputation_update_rate = np.random.randint(1, arr_len,
                                                    size=elements_to_change_reputation_update_rate)
    reputation_x_vector[mask_reputation_update_rate] = reputation_x_vector[mask_reputation_update_rate - 1]

    # Reputation assignment error:
    elements_to_change_reputation_assignment_error = np.random.binomial(arr_len, variables.reputation_assignment_error)
    mask_reputation_assignment_error = np.random.randint(
        1, arr_len, size=elements_to_change_reputation_assignment_error)
    reputation_x_vector[mask_reputation_assignment_error] = 1 - reputation_x_vector[mask_reputation_assignment_error]

    variables.reputation[x] = reputation_x_vector[len(reputation_x_vector)-1]
    mask = np.ones(reputation_x_vector.shape, dtype=bool)
    mask[arr_len] = False
    reputation_x_vector = reputation_x_vector[mask]

    # Action of Y:
    pstratindex = variables.population[y_array]
    pstrategy = strategies[pstratindex]
    pactionbad = pstrategy[:, 0]
    pactiongood = pstrategy[:, 1]

    """
        Action of Y with errors
    """
    cy = reputation_x_vector*pactiongood + (1 - reputation_x_vector)*pactionbad
    # Adjust for assessment error
    elements_to_change_assessment_error = np.random.binomial(arr_len, variables.assessment_error)
    mask_assessment_error = np.random.randint(arr_len, size=elements_to_change_assessment_error)
    cy[mask_assessment_error] = pactiongood[mask_assessment_error]*(1 - reputation_x_vector[mask_assessment_error]) +\
        pactionbad[mask_assessment_error]*reputation_x_vector[mask_assessment_error]
    # # Adjust for execution error
    elements_to_change_execution_error = np.random.binomial(arr_len, variables.execution_error)
    mask_execution_error = np.random.randint(arr_len, size=elements_to_change_execution_error)
    cy[mask_execution_error] = 0

    """
        Update Reputation of Y with errors
    """
    reputation_y_vector = variables.socialnorm[(1 - cy, 1 - reputation_x_vector)]
    # Reputation update rate:
    elements_to_change_reputation_update_rate = np.random.binomial(arr_len, (float(1) - variables.reputation_update_rate))
    mask_reputation_update_rate = np.random.randint(arr_len, size=elements_to_change_reputation_update_rate)
    reputation_y_vector[mask_reputation_update_rate] = variables.reputation[y_array[mask_reputation_update_rate]]
    # Reputation assignment error:
    elements_to_change_reputation_assignment_error = np.random.binomial(arr_len, variables.reputation_assignment_error)
    mask_reputation_assignment_error = np.random.randint(
        arr_len, size=elements_to_change_reputation_assignment_error)
    reputation_y_vector[mask_reputation_assignment_error] = 1 - reputation_y_vector[mask_reputation_assignment_error]
    variables.reputation[y_array] = reputation_y_vector

    # Track cooperation
    coops_y = np.sum(cy)
    coops_x = np.sum(cx)
    if variables.track_cooperation:
        cur_cooperation_index = float(float(coops_y + coops_x)/float(2 * arr_len))
        variables.increment_coop_index(cur_cooperation_index)

    return float((variables.benefit * coops_y) - (variables.cost * coops_x))


def simulate(variables):
    for r in range(0, variables.runs):

        # Initialise random population
        variables.population = np.random.randint(4, size=variables.population_size)  # equivalent to U(0, 3)
        variables.reputation = np.random.randint(2, size=variables.population_size)  # equivalent to U(0, 1)

        for t in range(0, variables.generations):
            # Check if after transient period
            if t > variables.generations // 10:
                variables.track_cooperation = True

            mutation_probabilities = np.random.rand(variables.population_size) < variables.mutation_rate
            for i in range(variables.population_size):

                fitness_a = 0.0
                fitness_b = 0.0

                agent_one = np.random.randint(variables.population_size)

                # Random mutation probability
                if mutation_probabilities[i]:
                    variables.population[agent_one] = np.random.randint(4)

                # Make sure B != A
                agent_two = np.random.randint(variables.population_size)
                while agent_two == agent_one:
                    agent_two = np.random.randint(variables.population_size)

                # Creating tournament arrays
                probabilities_a = np.ones(variables.population_size, dtype=np.float) / float(variables.population_size - 1)
                probabilities_a[agent_one] = 0.0
                probabilities_b = np.ones(variables.population_size, dtype=np.float) / float(variables.population_size - 1)
                probabilities_b[agent_two] = 0.0

                for _ in range(4):
                    tournament_sample_a = np.random.choice(variables.population_size, size=(variables.population_size - 1)//2, p=probabilities_a,
                                                           replace=False)
                    tournament_sample_b = np.random.choice(variables.population_size, size=(variables.population_size - 1)//2, p=probabilities_b,
                                                           replace=False)
                    fitness_a += fitness_function(agent_one, tournament_sample_a, variables)
                    fitness_b += fitness_function(agent_two, tournament_sample_b, variables)

                fitness_a /= float(2.0 * variables.population_size)
                fitness_b /= float(2.0 * variables.population_size)

                if np.random.random() < np.power(1.0 + np.exp(float(1)*float(fitness_a - fitness_b)), -1.0):
                    variables.population[agent_one] = variables.population[agent_two]


def run_instance(NumRuns, NumGenerations, PopulationSize, MutationRate,
                 ExecutionError, ReputationAssignmentError,
                 PrivateAssessmentError, ReputationUpdateProbability,
                 RandomSeed, SocialNormMatrix, CostValue, BenefitValue):
    """
    :return: an array in the form:
            [cooperation_index_avg,
            cooperation_index_min,
            cooperation_index_max,
            cooperation_index_zero_proportion,
            cooperation_index_without_zeros]
    """
    variables = InstanceVariables(NumRuns, NumGenerations, PopulationSize, MutationRate,
                                  ExecutionError, ReputationAssignmentError,
                                  PrivateAssessmentError, ReputationUpdateProbability,
                                  SocialNormMatrix, CostValue, BenefitValue)

    simulate(variables)
    result = variables.get_average_coop_index()
    return result

if __name__ == "__main__":
    start = time.clock()
    num_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_threads)
    cooperation_index_avg = 0.0
    results = [pool.apply_async(run_instance,
                                args=(1, 3*np.power(10, 4), 12, np.power(10*12, -1.0),
                                      0.08, 0.01, 0.01, 1.0,
                                      1, [[1, 0], [0, 1]], 1, 5)) for _ in range(num_threads)]
    """
    result is the cooperation index
    """
    for result in results:
        cooperation_index_values_i = result.get()
        cooperation_index_avg += float(cooperation_index_values_i)
        print(cooperation_index_values_i)
    cooperation_index_avg /= float(num_threads)
    end = time.clock()
    print(cooperation_index_avg)
    print("Simulation completed in " + str(end - start) + " seconds.")
