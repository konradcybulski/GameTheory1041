from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport exp
import cython
import numpy as np
cimport numpy as np
DINT = np.int
DDOUBLE = np.double
ctypedef np.int_t DINT_t
ctypedef np.double_t DDOUBLE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int[:] choice_weighted(int length, int size, double[:] weights):
    cdef int[:] arr = np.zeros(size, dtype=DINT)
    cdef int arr_i
    cdef int idx, i
    cdef double cs, random
    cdef randoms = np.random.random(size=size)
    for arr_i in range(size):
        cs = 0.0
        i = 0
        while cs < randoms[arr_i] and i < length:
            cs += weights[i]
            i += 1
        arr[arr_i] = i - 1
    return arr


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef payoff_function(int x, int y, InstanceVariables variables):
    # Action of X:
    """
    [np.int_t,
    ndim=2,
    negative_indices=False,
    mode='c']
    """
    cdef int[:] xstrategy = variables.strategies[variables.population[x]]
    cdef int cx, cy, rx, ry
    cdef double cur_cooperation_index
    cdef double[8] rands
    for i in range(8):
        rands[i] = rand()*1.0/RAND_MAX

    if rands[0] < variables.private_assessment_error:
        if rands[1] < variables.execution_error and xstrategy[1 - variables.reputation[y]]:
            cx = 1 - xstrategy[1 - variables.reputation[y]]
        else:
            cx = xstrategy[1 - variables.reputation[y]]
    else:
        if rands[1] < variables.execution_error and xstrategy[variables.reputation[y]]:
            cx = 1 - xstrategy[variables.reputation[y]]
        else:
            cx = xstrategy[variables.reputation[y]]

    # Action of Y:
    cdef int[:] ystrategy = variables.strategies[variables.population[y]]
    if rands[2] < variables.private_assessment_error:
        if rands[3] < variables.execution_error and ystrategy[1 - variables.reputation[x]]:
            cy = 1 - ystrategy[1 - variables.reputation[x]]
        else:
            cy = ystrategy[1 - variables.reputation[x]]
    else:
        if rands[3] < variables.execution_error and ystrategy[variables.reputation[x]]:
            cy = 1 - ystrategy[variables.reputation[x]]
        else:
            cy = ystrategy[variables.reputation[x]]
    rx = variables.reputation[x]
    ry = variables.reputation[y]
    # Update Reputation of X:
    if rands[4] < variables.reputation_update_rate:
        if rands[5] < variables.reputation_assignment_error:
            rx = 1 - variables.socialnorm[1 - cx][1 - variables.reputation[y]]
        else:
            rx = variables.socialnorm[1 - cx][1 - variables.reputation[y]]

    # Update Reputation of Y:
    if rands[6] < variables.reputation_update_rate:
        if rands[7] < variables.reputation_assignment_error:
            ry = 1 - variables.socialnorm[1 - cy][1 - variables.reputation[x]]
        else:
            ry = variables.socialnorm[1 - cy][1 - variables.reputation[x]]

    variables.reputation[x] = rx
    variables.reputation[y] = ry
    # Track cooperation
    if variables.track_coop:
        cur_cooperation_index = float(float(cy + cx)/2.0)
        variables.increment_coop_index(cur_cooperation_index)
    return float((variables.benefit * cy) - (variables.cost * cx))

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef double[:] fitness_function(int x, int y, InstanceVariables variables):
    cdef int Z = variables.population_size

    cdef double fitness_x = 0.0
    cdef double fitness_y = 0.0
    cdef int t_size = 2 * Z
    cdef int agent_z, c

    cdef double[:] probabilities_x = np.ones(Z, dtype=DDOUBLE) / float(Z - 1)
    probabilities_x[x] = 0
    cdef double[:] probabilities_y = np.ones(Z, dtype=DDOUBLE) / float(Z - 1)
    probabilities_y[y] = 0

    cdef int[:] t_arr_x = choice_weighted(Z, size=2 * Z, weights=probabilities_x)
    cdef int[:] t_arr_y = choice_weighted(Z, size=2 * Z, weights=probabilities_y)
    for c in range(0, 2 * Z):
        agent_z = t_arr_x[c]
        fitness_x += payoff_function(x, agent_z, variables)
        agent_z = t_arr_y[c]
        fitness_y += payoff_function(y, agent_z, variables)

    fitness_x /= float(2 * Z)
    fitness_y /= float(2 * Z)
    cdef double[2] return_arr
    return_arr[:] = [fitness_x, fitness_y]
    return return_arr


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef double simulate(int runs, int generations, int population_size, double mutation_rate,
                 double execution_error, double reputation_assignment_error,
                 double private_assessment_error, double reputation_update_rate,
                 np.ndarray[DINT_t, ndim=2] socialnorm, int cost, int benefit):
    cdef InstanceVariables variables = InstanceVariables(runs, generations, population_size, mutation_rate,
                                  execution_error, reputation_assignment_error,
                                  private_assessment_error, reputation_update_rate,
                                  socialnorm, cost, benefit)
    cdef int r, g, i
    cdef int Z = variables.population_size
    cdef np.ndarray[DINT_t, ndim=1, negative_indices=False, mode='c'] mutation_probs
    cdef np.ndarray[DINT_t, ndim=2, negative_indices=False, mode='c'] agent_pairs
    cdef double fitness_a, fitness_b, random
    cdef double[:] fitness
    cdef int agent_one, agent_two

    for r in range(0, variables.runs):
        variables.population = np.random.randint(4, size=Z)  # equivalent to U(0, 3)
        variables.reputation = np.random.randint(2, size=Z)  # equivalent to U(0, 1)
        for g in range(0, variables.generations):
            if g > variables.generations // 10:
                variables.track_coop = 1
            mutation_probs = np.array([1 if (rand()*1.0/RAND_MAX) < variables.mutation_rate else 0 for _ in range(Z)])
            agent_pairs = np.array([np.random.choice(Z, size=2, replace=False) for _ in range(Z)])
            for i in range(Z):

                agent_one = agent_pairs[i][0]
                agent_two = agent_pairs[i][1]

                # Random mutation probability
                if mutation_probs[i]:
                    variables.population[agent_one] = int(rand()*4.0/RAND_MAX) #np.random.randint(4)

                # Calculate fitness of agents
                fitness = fitness_function(agent_one, agent_two, variables)
                fitness_a = fitness[0]
                fitness_b = fitness[1]

                if (rand()*1.0/RAND_MAX) < pow(1 + exp(fitness_a - fitness_b), -1):
                    variables.population[agent_one] = variables.population[agent_two]
    cdef double result = variables.get_average_coop_index()
    return result

def run_instance(int runs, int generations, int population_size, double mutation_rate,
                 double execution_error, double reputation_assignment_error,
                 double private_assessment_error, double reputation_update_rate,
                 np.ndarray[DINT_t, ndim=2] socialnorm, int cost, int benefit):
    cdef double result = simulate(1, generations, population_size, mutation_rate,
                                  execution_error, reputation_assignment_error,
                                  private_assessment_error, reputation_update_rate,
                                  socialnorm, cost, benefit)
    return [result, population_size, socialnorm]

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef class InstanceVariables:
    cdef public int runs
    cdef public int generations
    cdef public int population_size
    cdef public double mutation_rate,
    cdef public double execution_error
    cdef public double reputation_assignment_error
    cdef public double private_assessment_error
    cdef public double reputation_update_rate
    cdef public int[:,:] socialnorm
    cdef public int cost
    cdef public int benefit

    cdef public int[:,:] strategies

    cdef public int[:] population
    cdef public int[:] reputation

    cdef double coop_index_sum
    cdef double interaction_count
    cdef int track_coop

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    def __cinit__(self, int runs, int generations, int population_size, double mutation_rate,
                 double execution_error, double reputation_assignment_error,
                 double private_assessment_error, double reputation_update_rate,
                 np.ndarray[DINT_t, ndim=2] socialnorm, int cost, int benefit):
        self.runs = runs
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.execution_error = execution_error
        self.reputation_assignment_error = reputation_assignment_error
        self.private_assessment_error = private_assessment_error
        self.reputation_update_rate = reputation_update_rate
        self.socialnorm = socialnorm  # matrix determining the reputation dynamic with
        # regard to the action taken and the reputation
        # of the other agent
        self.cost = cost  # cost defining the payoff matrix cost
        self.benefit = benefit  # benefit defined as the payoff matrix benefit

        # Population and reputation arrays
        self.population = np.zeros(population_size, dtype=DINT)  # vector of all individual strategies
        # population[k] : strategy of individual k
        # population[k] = 0, 1, 2 or 3
        self.reputation = np.zeros(population_size, dtype=DINT)  # vector of all individual public reputations
        # reputation[k] : public reputation of individual k
        # reputation[k] = 0 or 1
        self.strategies = np.array([[0, 0],
                                    [1, 0],
                                    [0, 1],
                                    [1, 1]])

        # Cooperation Tracking
        self.coop_index_sum = 0.0
        self.interaction_count = 0.0
        self.track_coop = 0

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef void increment_coop_index(self, double coop_index):
        self.coop_index_sum += coop_index
        self.interaction_count += 1.0

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    @cython.wraparound(False)
    cdef double get_average_coop_index(self):
        return self.coop_index_sum/self.interaction_count
