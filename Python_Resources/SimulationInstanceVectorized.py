"""
@author Konrad Cybulski
@since 14/08/2016
@modified 16/08/2016
Code written utilising pseudocode provided
 in 'Social norms of cooperation in small-scale
 societies' by Santos, Santos, Pacheco
"""
import numpy as np
import numpy.ma as ma
import time
from numba import jit
from collections import Counter

"""
    Static Variables
"""

#              B  G
strategies = np.array([[0, 0],  # AllD
                      [1, 0],  # pDisc
                      [0, 1],  # Disc
                      [1, 1]])  # AllC

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
socialnorm = np.array([[1, 0], [0, 1]])  # matrix determining the reputation dynamic with
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

def save_generation_data(gen_num):
    # file_out.write("Generation Number,AllD Count,pDisc Count,Disc Count,AllD Count," +
    #                "AllD Ratio,pDisc Ratio,Disc Ratio,AllD Ratio")
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
        ",\n"
    file_out = open(generation_data_save_filename, 'a')
    file_out.write(out_string)
    file_out.close()

def fitness_function(x, y_array):
    """
    :param x: the index of agent-x in population
    :param y_array: the array of indices of agent-y's in population
    :return: the fitness of x after
    """
    # Action of X:
    arr_len = y_array.size

    p_rep = reputation[y_array]

    x_strategy = strategies[population[x]]
    xactionbad = x_strategy[0]
    xactiongood = x_strategy[1]

    """
        Action of X with errors
    """
    cx = xactiongood*p_rep + xactionbad*(1 - p_rep)

    # Adjust for assessment error
    assessment_error_changes = np.int(arr_len * assessment_error)
    mask_assessment_error = np.random.randint(arr_len, size=assessment_error_changes)
    cx[mask_assessment_error] = (xactiongood*(1 - p_rep[mask_assessment_error]) +
                                 xactionbad*p_rep[mask_assessment_error])

    # Adjust for execution error
    elements_to_change_execution_error = np.int(arr_len * execution_error)
    mask_execution_error = np.random.randint(arr_len, size=elements_to_change_execution_error)
    cx[mask_execution_error] = 0

    """
        Update Reputation of X with errors
    """
    reputation_x_vector = np.insert(socialnorm[(1 - cx, 1 - p_rep)], 0, reputation[x])

    # Reputation update rate:
    elements_to_change_reputation_update_rate = np.int(arr_len * (float(1) - reputation_update_rate))
    mask_reputation_update_rate = np.random.randint(1, arr_len,
                                                    size=elements_to_change_reputation_update_rate)
    reputation_x_vector[mask_reputation_update_rate] = reputation_x_vector[mask_reputation_update_rate - 1]

    # Reputation assignment error:
    elements_to_change_reputation_assignment_error = np.int(arr_len * reputation_assignment_error)
    mask_reputation_assignment_error = np.random.randint(
        1, arr_len, size=elements_to_change_reputation_assignment_error)
    reputation_x_vector[mask_reputation_assignment_error] = 1 - reputation_x_vector[mask_reputation_assignment_error]

    reputation[x] = reputation_x_vector[len(reputation_x_vector)-1]
    mask = np.ones(reputation_x_vector.shape, dtype=bool)
    mask[arr_len] = False
    reputation_x_vector = reputation_x_vector[mask]

    # Action of Y:
    pstratindex = population[y_array]
    pstrategy = strategies[pstratindex]
    pactionbad = pstrategy[:, 0]
    pactiongood = pstrategy[:, 1]

    """
        Action of Y with errors
    """
    cy = reputation_x_vector*pactiongood + (1 - reputation_x_vector)*pactionbad
    # Adjust for assessment error
    elements_to_change_assessment_error = np.int(arr_len * assessment_error)
    mask_assessment_error = np.random.randint(arr_len, size=elements_to_change_assessment_error)
    cy[mask_assessment_error] = pactiongood[mask_assessment_error]*(1 - reputation_x_vector[mask_assessment_error]) +\
        pactionbad[mask_assessment_error]*reputation_x_vector[mask_assessment_error]
    # # Adjust for execution error
    elements_to_change_execution_error = np.int(arr_len * execution_error)
    mask_execution_error = np.random.randint(arr_len, size=elements_to_change_execution_error)
    cy[mask_execution_error] = 0

    """
        Update Reputation of Y with errors
    """
    reputation_y_vector = socialnorm[(1 - cy, 1 - reputation_x_vector)]
    # Reputation update rate:
    elements_to_change_reputation_update_rate = np.int(arr_len * (float(1) - reputation_update_rate))
    mask_reputation_update_rate = np.random.randint(arr_len, size=elements_to_change_reputation_update_rate)
    reputation_y_vector[mask_reputation_update_rate] = reputation[y_array[mask_reputation_update_rate]]
    # Reputation assignment error:
    elements_to_change_reputation_assignment_error = np.int(arr_len * reputation_assignment_error)
    mask_reputation_assignment_error = np.random.randint(
        arr_len, size=elements_to_change_reputation_assignment_error)
    reputation_y_vector[mask_reputation_assignment_error] = 1 - reputation_y_vector[mask_reputation_assignment_error]
    reputation[y_array] = reputation_y_vector

    # Track cooperation
    global cooperation_index_sum
    global cooperation_index_min
    global cooperation_index_max
    global cooperation_index_zeros
    coops_y = np.sum(cy)
    coops_x = np.sum(cx)
    cur_cooperation_index = float(float(coops_y + coops_x)/float(2 * arr_len))
    cooperation_index_sum += cur_cooperation_index
    cooperation_index_min = min(cooperation_index_min, cur_cooperation_index)
    cooperation_index_max = max(cooperation_index_max, cur_cooperation_index)
    if cur_cooperation_index < float(np.power(float(10), float(-5))):
        cooperation_index_zeros += 1

    return (benefit * coops_y) - (cost * coops_x)


def simulate():
    for r in range(0, runs):

        # Initialise random population
        global population
        global reputation
        population = np.random.randint(4, size=population_size)  # equivalent to U(0, 3)
        reputation = np.random.randint(2, size=population_size)  # equivalent to U(0, 1)

        for t in range(0, generations):
            if generation_data_save_filename != "":
                if t % generation_data_save_wait == 0:
                    save_generation_data(t)

            agent_one = np.random.randint(population_size)

            # Random mutation probability
            if np.random.random() < mutation_probability:
                population[agent_one] = np.random.randint(4)

            # Make sure B != A
            agent_two = np.random.randint(population_size)
            while agent_two == agent_one:
                agent_two = np.random.randint(population_size)

            # Creating tournament arrays
            # Agent One:
            tournament_sample_a = np.random.randint(population_size, size=2*population_size)
            while (tournament_sample_a == agent_one).any():
                tournament_sample_a[tournament_sample_a == agent_one] =\
                    np.random.randint(population_size, size=np.sum(tournament_sample_a == agent_one))
            fitness_a = fitness_function(agent_one, tournament_sample_a)

            # Agent Two:
            tournament_sample_b = np.random.randint(population_size, size=2*population_size)
            while (tournament_sample_b == agent_two).any():
                tournament_sample_b[tournament_sample_b == agent_two] =\
                    np.random.randint(population_size, size=np.sum(tournament_sample_b == agent_two))
            fitness_b = fitness_function(agent_two, tournament_sample_b)

            fitness_a /= (2 * population_size)
            fitness_b /= (2 * population_size)
            if np.random.random() < np.power(1 + np.exp(fitness_a - fitness_b), -1):
                population[agent_one] = population[agent_two]
    global cooperation_index_sum
    global cooperation_index_average
    cooperation_index_average = float(cooperation_index_sum)/float(runs*generations*2)


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
    """
    :return: an array in the form:
            [cooperation_index_avg,
            cooperation_index_min,
            cooperation_index_max,
            cooperation_index_zero_proportion,
            cooperation_index_without_zeros]
    """
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
    socialnorm = np.array(SocialNormMatrix)
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

    # start = time.clock()
    # print("Simulation beginning...")

    simulate()
    # end = time.clock()
    # print("Simulation completed in " + str(end - start))
    return_list = [cooperation_index_average,
                   cooperation_index_min,
                   cooperation_index_max,
                   float(cooperation_index_zeros) / float(runs * generations * 2),
                   float(cooperation_index_sum) / float((runs * generations * 2) - cooperation_index_zeros)]
    return return_list # float(cooperation_index_average)
