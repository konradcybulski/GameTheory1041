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

"""
    Static Variables
"""

#              B  G
strategies = [[0, 0],  # AllD
              [1, 0],  # pDisc
              [0, 1],  # Disc
              [1, 1]]  # AllC

"""
    Simulation Variables
"""
runs = 0  # number of runs
generations = 0  # number of generations
population_size = 0  # population size
population = []  # vector of all individual strategies
# population[k] : strategy of individual k
# population[k] = [0,0], [1,0], [0,1] or [1,1]
reputation = []  # vector of all individual public reputations
# reputation[k] : public reputation of individual k
# reputation[k] = 0 or 1
mutation_probability = 0  # mutation probability
execution_error = 0  # execution error probability
reputation_assignment_error = 0  # reputation assignment error probability
assessment_error = 0  # private assessment error probability
reputation_update_rate = 0  # reputation update probability
randomseed = 0  # seed used to generate randomness
socialnorm = [[1, 0], [0, 1]]  # matrix determining the reputation dynamic with
# regard to the action taken and the reputation
# of the other agent
cost = 1  # cost defining the payoff matrix cost
benefit = 5  # benefit defined as the payoff matrix benefit

### Tracking Variables
cooperation_index_sum = 0
cooperation_index_min = 1
cooperation_index_max = 0
cooperation_index_average = 0
cooperation_index_zeros = 0

# Data saving variables
generation_data_save_wait = -1
generation_data_save_filename = ""

def save_generation_data(gen_num, fitness_a, fitness_b, strat_a, strat_b):
    counter = Counter(population)
    alld_count = counter[0]
    pdisc_count = counter[1]
    disc_count = counter[2]
    allc_count = counter[3]
    out_string = str(gen_num) + "," +\
        str(alld_count) + "," +\
        str(pdisc_count) + "," +\
        str(disc_count) + "," +\
        str(allc_count) + "," +\
        str(float(alld_count)/float(population_size)) + "," +\
        str(float(pdisc_count)/float(population_size)) + "," +\
        str(float(disc_count)/float(population_size)) + "," +\
        str(float(allc_count)/float(population_size)) + "," +\
        str(fitness_a) + "," +\
        str(fitness_b) + "," +\
        str(strat_a) + "," +\
        str(strat_b) + "," +\
        ",\n"
    file_out = open(generation_data_save_filename, 'a')
    file_out.write(out_string)
    file_out.close()


def fitness_function(x, y):
    """
    :param x: the index of agent-x in population
    :param y: the index of agent-y in population
    :return: the fitness of x after
    """
    # Action of X:
    xstrategy = strategies[population[x]]
    if np.random.random() < assessment_error:
        if np.random.random() < execution_error and xstrategy[1 - reputation[y]]:
            cx = 1 - xstrategy[1 - reputation[y]]
        else:
            cx = xstrategy[1 - reputation[y]]
    else:
        if np.random.random() < execution_error and xstrategy[reputation[y]]:
            cx = 1 - xstrategy[reputation[y]]
        else:
            cx = xstrategy[reputation[y]]
    # Action of Y:
    ystrategy = strategies[population[y]]
    if np.random.random() < assessment_error:
        if np.random.random() < execution_error and ystrategy[1 - reputation[x]]:
            cy = 1 - ystrategy[1 - reputation[x]]
        else:
            cy = ystrategy[1 - reputation[x]]
    else:
        if np.random.random() < execution_error and ystrategy[reputation[x]]:
            cy = 1 - ystrategy[reputation[x]]
        else:
            cy = ystrategy[reputation[x]]

    # Update Reputation of X:
    if np.random.random() < reputation_update_rate:
        if np.random.random() < reputation_assignment_error:
            reputation[x] = 1 - socialnorm[1 - cx][1 - reputation[y]]
        else:
            reputation[x] = socialnorm[1 - cx][1 - reputation[y]]
    # Update Reputation of Y:
    if np.random.random() < reputation_update_rate:
        if np.random.random() < reputation_assignment_error:
            reputation[y] = 1 - socialnorm[1 - cy][1 - reputation[x]]
        else:
            reputation[y] = socialnorm[1 - cy][1 - reputation[x]]
    # Track cooperation
    global cooperation_index_sum
    global cooperation_index_min
    global cooperation_index_max
    global cooperation_index_zeros
    coops_x = 1 if cx == 1 else 0
    coops_y = 1 if cy == 1 else 0
    cur_cooperation_index = float(float(coops_y + coops_x)/float(2))
    cooperation_index_sum += cur_cooperation_index
    cooperation_index_min = min(cooperation_index_min, cur_cooperation_index)
    cooperation_index_max = max(cooperation_index_max, cur_cooperation_index)
    if cur_cooperation_index < float(np.power(float(10), float(-5))):
        cooperation_index_zeros += 1
    return (benefit * cy) - (cost * cx)


def simulate():
    for r in range(0, runs):

        # Initialise random population
        global population
        global reputation
        population = np.random.randint(4, size=population_size)  # equivalent to U(0, 3)
        reputation = np.random.randint(2, size=population_size)  # equivalent to U(0, 1)

        for t in range(0, generations):

            mutation_probs = np.random.rand(population_size) < mutation_probability
            for i in range(population_size):

                agent_one = np.random.randint(population_size)

                # Random mutation probability
                if mutation_probs[i]:
                    population[agent_one] = np.random.randint(4)

                # Make sure B != A
                agent_two = np.random.randint(population_size)
                while agent_two == agent_one:
                    agent_two = np.random.randint(population_size)

                fitness_a = 0
                fitness_b = 0

                # Creating tournament arrays
                probabilities_a = np.ones(population_size, dtype=np.float) / float(population_size - 1)
                probabilities_a[agent_one] = 0
                probabilities_b = np.ones(population_size, dtype=np.float) / float(population_size - 1)
                probabilities_b[agent_two] = 0

                tournament_sample_a = np.random.choice(population_size, size=2 * population_size, p=probabilities_a,
                                                       replace=True)
                tournament_sample_b = np.random.choice(population_size, size=2 * population_size, p=probabilities_b,
                                                       replace=True)

                for c in range(0, 2 * population_size):
                    agent_three = tournament_sample_a[c]
                    # Update Fitness of A and Reputation of A & C
                    fitness_a += fitness_function(agent_one, agent_three)

                    agent_three = tournament_sample_b[c]
                    # Update Fitness of B and Reputation of B & C
                    fitness_b += fitness_function(agent_two, agent_three)
                fitness_a /= (2 * population_size)
                fitness_b /= (2 * population_size)
                if generation_data_save_filename != "":
                    if t % generation_data_save_wait == 0:
                        save_generation_data(t, fitness_a, fitness_b, population[agent_one], population[agent_two])

                if np.random.random() < np.power(1 + np.exp(fitness_a - fitness_b), -1):
                    population[agent_one] = population[agent_two]
    global cooperation_index_sum
    global cooperation_index_average
    cooperation_index_average = float(cooperation_index_sum)/float(runs*generations*4*population_size*population_size)

def run_instance_generation_information(NumRuns, NumGenerations, PopulationSize, MutationRate,
                 ExecutionError, ReputationAssignmentError,
                 PrivateAssessmentError, ReputationUpdateProbability,
                 RandomSeed, SocialNormMatrix, CostValue, BenefitValue,
                 GenerationDataSaveWait, GenerationDataSaveFilename):
    global generation_data_save_wait
    global generation_data_save_filename
    generation_data_save_wait = GenerationDataSaveWait
    generation_data_save_filename = GenerationDataSaveFilename
    file_out = open(generation_data_save_filename, 'w+')
    file_out.write("Generation Number,AllD Count,pDisc Count,Disc Count,AllC Count," +
                   "AllD Ratio,pDisc Ratio,Disc Ratio,AllC Ratio,\n")
    file_out.close()
    output = run_instance(NumRuns, NumGenerations, PopulationSize, MutationRate,
                 ExecutionError, ReputationAssignmentError,
                 PrivateAssessmentError, ReputationUpdateProbability,
                 RandomSeed, SocialNormMatrix, CostValue, BenefitValue)
    final_string = str(population_size) + ',' +\
                 str(output[0]) + ',' +\
                 str(output[1]) + ',' +\
                 str(output[2]) + ',' +\
                 str(output[3]) + ',' +\
                 str(output[4]) + ',\n'
    file_out = open(generation_data_save_filename, 'a')
    file_out.write("---,---,---,---,---,---,---,---,---,\n")
    file_out.write(final_string)
    file_out.close()


def run_instance(NumRuns, NumGenerations, PopulationSize, MutationRate,
                 ExecutionError, ReputationAssignmentError,
                 PrivateAssessmentError, ReputationUpdateProbability,
                 RandomSeed, SocialNormMatrix, CostValue, BenefitValue):
    global runs
    global generations
    global population_size
    global population
    global reputation
    global mutation_probability
    global execution_error
    global reputation_assignment_error
    global assessment_error
    global reputation_update_rate
    global randomseed
    global socialnorm
    global cost
    global benefit
    runs = NumRuns
    generations = NumGenerations
    population_size = PopulationSize

    mutation_probability = MutationRate
    execution_error = ExecutionError
    reputation_assignment_error = ReputationAssignmentError
    assessment_error = PrivateAssessmentError
    reputation_update_rate = ReputationUpdateProbability
    randomseed = RandomSeed
    socialnorm = SocialNormMatrix
    cost = CostValue
    benefit = BenefitValue

    ### Reset tracking variables
    global cooperation_index_sum
    global cooperation_index_average
    global cooperation_index_min
    global cooperation_index_max
    global cooperation_index_zeros
    cooperation_index_min = 1
    cooperation_index_max = 0
    cooperation_index_zeros = 0
    cooperation_index_sum = 0
    cooperation_index_average = 0

    simulate()
    return_list = [cooperation_index_average,
                   cooperation_index_min,
                   cooperation_index_max,
                   float(cooperation_index_zeros) / float(runs * generations * 4 * population_size*population_size),
                   float(cooperation_index_sum) / float(
                       (runs * generations * 4 * population_size*population_size) - cooperation_index_zeros)]
    return return_list