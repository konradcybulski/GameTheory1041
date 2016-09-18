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
from collections import Counter
from InstanceVariables import InstanceVariables

"""
    Static Variables
"""

#              B  G
strategies = [[0, 0],  # AllD
              [1, 0],  # pDisc
              [0, 1],  # Disc
              [1, 1]]  # AllC

# Data saving variables
generation_data_save_wait = -1
generation_data_save_filename = ""

def save_generation_data(gen_num, fitness_a, fitness_b, strat_a, strat_b, variables):
    counter = Counter(variables.population)
    alld_count = counter[0]
    pdisc_count = counter[1]
    disc_count = counter[2]
    allc_count = counter[3]
    out_string = str(gen_num) + "," +\
        str(alld_count) + "," +\
        str(pdisc_count) + "," +\
        str(disc_count) + "," +\
        str(allc_count) + "," +\
        str(float(alld_count)/float(variables.population_size)) + "," +\
        str(float(pdisc_count)/float(variables.population_size)) + "," +\
        str(float(disc_count)/float(variables.population_size)) + "," +\
        str(float(allc_count)/float(variables.population_size)) + "," +\
        str(fitness_a) + "," +\
        str(fitness_b) + "," +\
        str(strat_a) + "," +\
        str(strat_b) + "," +\
        ",\n"
    file_out = open(generation_data_save_filename, 'a')
    file_out.write(out_string)
    file_out.close()


def fitness_function(x, y, variables):

    """
    :param x: the index of agent-x in population
    :param y: the index of agent-y in population
    :return: the fitness of x after
    """
    # Action of X:
    xstrategy = strategies[variables.population[x]]
    if np.random.random() < variables.assessment_error:
        if np.random.random() < variables.execution_error and xstrategy[1 - variables.reputation[y]]:
            cx = 1 - xstrategy[1 - variables.reputation[y]]
        else:
            cx = xstrategy[1 - variables.reputation[y]]
    else:
        if np.random.random() < variables.execution_error and xstrategy[variables.reputation[y]]:
            cx = 1 - xstrategy[variables.reputation[y]]
        else:
            cx = xstrategy[variables.reputation[y]]
    # Action of Y:
    ystrategy = strategies[variables.population[y]]
    if np.random.random() < variables.assessment_error:
        if np.random.random() < variables.execution_error and ystrategy[1 - variables.reputation[x]]:
            cy = 1 - ystrategy[1 - variables.reputation[x]]
        else:
            cy = ystrategy[1 - variables.reputation[x]]
    else:
        if np.random.random() < variables.execution_error and ystrategy[variables.reputation[x]]:
            cy = 1 - ystrategy[variables.reputation[x]]
        else:
            cy = ystrategy[variables.reputation[x]]
    rx = variables.reputation[x]
    ry = variables.reputation[y]
    # Update Reputation of X:
    if np.random.random() < variables.reputation_update_rate:
        if np.random.random() < variables.reputation_assignment_error:
            rx = 1 - variables.socialnorm[1 - cx][1 - variables.reputation[y]]
        else:
            rx = variables.socialnorm[1 - cx][1 - variables.reputation[y]]
    # Update Reputation of Y:
    if np.random.random() < variables.reputation_update_rate:
        if np.random.random() < variables.reputation_assignment_error:
            ry = 1 - variables.socialnorm[1 - cy][1 - variables.reputation[x]]
        else:
            ry = variables.socialnorm[1 - cy][1 - variables.reputation[x]]
    variables.reputation[x] = rx
    variables.reputation[y] = ry
    # Track cooperation
    cur_cooperation_index = float(float(cy + cx)/float(2))
    variables.increment_coop_index(cur_cooperation_index)
    return (variables.benefit * cy) - (variables.cost * cx)


def simulate(runs, generations, population_size, mutation_rate,
                 execution_error, reputation_assignment_error,
                 private_assessment_error, reputation_update_prob,
                 socialnorm, cost, benefit):
    variables = InstanceVariables(runs, generations, population_size, mutation_rate,
                                  execution_error, reputation_assignment_error,
                                  private_assessment_error, reputation_update_prob,
                                  socialnorm, cost, benefit)
    # Simulation begins
    Z = variables.population_size
    for r in range(0, variables.runs):

        # Initialise random population
        variables.population = np.random.randint(4, size=Z)  # equivalent to U(0, 3)
        variables.reputation = np.random.randint(2, size=Z)  # equivalent to U(0, 1)

        for t in range(0, variables.generations):

            mutation_probs = np.random.rand(Z) < variables.mutation_rate
            agent_pairs = [np.random.choice(Z, size=2, replace=False)]*Z
            for i in range(Z):

                agent_one = agent_pairs[i][0]

                # Random mutation probability
                if mutation_probs[i]:
                    variables.population[agent_one] = np.random.randint(4)

                # Make sure B != A
                agent_two = agent_pairs[i][1]

                fitness_a = 0
                fitness_b = 0

                # Creating tournament arrays
                probabilities_a = np.ones(Z, dtype=np.float) / float(Z - 1)
                probabilities_a[agent_one] = 0
                probabilities_b = np.ones(Z, dtype=np.float) / float(Z - 1)
                probabilities_b[agent_two] = 0

                tournament_sample_a = np.random.choice(Z, size=2 * Z, p=probabilities_a,
                                                       replace=True)
                tournament_sample_b = np.random.choice(Z, size=2 * Z, p=probabilities_b,
                                                       replace=True)

                for c in range(0, 2 * Z):
                    agent_three = tournament_sample_a[c]
                    # Update Fitness of A and Reputation of A & C
                    fitness_a += fitness_function(agent_one, agent_three, variables)

                    agent_three = tournament_sample_b[c]
                    # Update Fitness of B and Reputation of B & C
                    fitness_b += fitness_function(agent_two, agent_three, variables)
                fitness_a /= (2 * Z)
                fitness_b /= (2 * Z)

                if generation_data_save_filename != "":
                    if t % generation_data_save_wait == 0:
                        save_generation_data(t, fitness_a, fitness_b, variables.population[agent_one],
                                             variables.population[agent_two], variables)

                if np.random.random() < np.power(1 + np.exp(fitness_a - fitness_b), -1):
                    variables.population[agent_one] = variables.population[agent_two]
    # Simulation ends
    # Return cooperation index.
    coop_index = variables.get_average_coop_index()
    return coop_index
