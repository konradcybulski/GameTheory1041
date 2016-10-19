"""
@author Konrad Cybulski
@since 14/08/2016
@modified 16/08/2016
Code written utilising pseudocode provided
 in 'Social norms of cooperation in small-scale
 societies' by Santos, Santos, Pacheco
"""
import numpy as np
from InstanceVariables import InstanceVariables

"""
    Static Variables
"""

#              B  G
strategies = np.array([[0, 0],  # AllD
              [1, 0],  # pDisc
              [0, 1],  # Disc
              [1, 1]])  # AllC

def payoff_function(x, y, variables):
    """
    :param x: the index of agent-x in population
    :param y: the index of agent-y in population
    :param variables: the class containing simulation instance variables.
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


def fitness_function(agent_x, agent_y, variables):
    Z = variables.population_size

    fitness_x = 0
    fitness_y = 0

    # Creating tournament arrays
    probabilities_x = np.ones(Z, dtype=np.float) / float(Z - 1)
    probabilities_x[agent_x] = 0
    probabilities_y = np.ones(Z, dtype=np.float) / float(Z - 1)
    probabilities_y[agent_y] = 0

    tournament_sample_x = np.random.choice(Z, size=2 * Z, p=probabilities_x,
                                           replace=True)
    tournament_sample_y = np.random.choice(Z, size=2 * Z, p=probabilities_y,
                                           replace=True)

    for c in range(0, 2 * Z):
        agent_three = tournament_sample_x[c]
        # Update Fitness of A and Reputation of A & C
        fitness_x += payoff_function(agent_x, agent_three, variables)

        agent_three = tournament_sample_y[c]
        # Update Fitness of B and Reputation of B & C
        fitness_y += payoff_function(agent_y, agent_three, variables)
    fitness_x /= (2 * Z)
    fitness_y /= (2 * Z)
    return [fitness_x, fitness_y]


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
            agent_pairs = [np.random.choice(Z, size=2, replace=False) for _ in range(Z)]
            for i in range(Z):

                agent_one = agent_pairs[i][0]
                agent_two = agent_pairs[i][1]

                # Random mutation probability
                if mutation_probs[i]:
                    variables.population[agent_one] = np.random.randint(4)

                # Calculate fitness of agents
                fitness = fitness_function(agent_one, agent_two, variables)
                fitness_a = fitness[0]
                fitness_b = fitness[1]

                if np.random.random() < np.power(1 + np.exp(fitness_a - fitness_b), -1):
                    variables.population[agent_one] = variables.population[agent_two]

    # Simulation ends
    # Return cooperation index.
    coop_index = variables.get_average_coop_index()
    return coop_index
