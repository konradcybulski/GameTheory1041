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

from InstanceVariables import InstanceVariables

"""
    Static Variables
"""

#              B  G
strategies = np.array([[0, 0],  # AllD
                      [1, 0],  # pDisc
                      [0, 1],  # Disc
                      [1, 1]])  # AllC


def payoff_function(x, y_array, variables):
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


def fitness_function(agent_x, agent_y, variables):
    Z = variables.population_size

    fitness_x = 0.0
    fitness_y = 0.0

    # Creating tournament arrays
    probabilities_x = np.ones(variables.population_size, dtype=np.float) / float(variables.population_size - 1)
    probabilities_x[agent_x] = 0.0
    probabilities_y = np.ones(variables.population_size, dtype=np.float) / float(variables.population_size - 1)
    probabilities_y[agent_y] = 0.0

    for _ in range(4):
        tournament_sample_x = np.random.choice(variables.population_size, size=(variables.population_size - 1) // 2,
                                               p=probabilities_x,
                                               replace=False)
        tournament_sample_y = np.random.choice(variables.population_size, size=(variables.population_size - 1) // 2,
                                               p=probabilities_y,
                                               replace=False)

        fitness_x += payoff_function(agent_x, tournament_sample_x, variables)
        fitness_y += payoff_function(agent_y, tournament_sample_y, variables)

    fitness_x /= float(2.0 * variables.population_size)
    fitness_y /= float(2.0 * variables.population_size)
    return [fitness_x, fitness_y]


def simulate(runs, generations, population_size, mutation_rate,
             execution_error, reputation_assignment_error,
             private_assessment_error, reputation_update_prob,
             socialnorm, cost, benefit):
    variables = InstanceVariables(runs, generations, population_size, mutation_rate,
                                  execution_error, reputation_assignment_error,
                                  private_assessment_error, reputation_update_prob,
                                  socialnorm, cost, benefit)
    Z = variables.population_size
    for r in range(0, variables.runs):

        # Initialise random population
        variables.population = np.random.randint(4, size=variables.population_size)  # equivalent to U(0, 3)
        variables.reputation = np.random.randint(2, size=variables.population_size)  # equivalent to U(0, 1)

        for t in range(0, variables.generations):
            # Check if after transient period
            if t > variables.generations // 10:
                variables.track_cooperation = True

            mutation_probabilities = np.random.rand(variables.population_size) < variables.mutation_rate
            agent_pairs = [np.random.choice(Z, size=2, replace=False) for _ in range(Z)]
            for i in range(variables.population_size):

                agent_one = agent_pairs[i][0]
                agent_two = agent_pairs[i][1]

                # Random mutation probability
                if mutation_probabilities[i]:
                    variables.population[agent_one] = np.random.randint(4)

                fitness = fitness_function(agent_one, agent_two, variables)
                fitness_a = fitness[0]
                fitness_b = fitness[1]

                if np.random.random() < np.power(1.0 + np.exp(float(1)*float(fitness_a - fitness_b)), -1.0):
                    variables.population[agent_one] = variables.population[agent_two]
    result = variables.get_average_coop_index()
    return result
