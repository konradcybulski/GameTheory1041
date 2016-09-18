"""
@author Konrad Cybulski
@since 14/09/2016
@modified 14/09/2016
"""
import numpy as np

class InstanceVariables:
    def __init__(self, runs, generations, population_size, mutation_rate,
                 execution_error, reputation_assignment_error,
                 private_assessment_error, reputation_update_rate,
                 socialnorm, cost, benefit):
        self.runs = runs
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.execution_error = execution_error
        self.reputation_assignment_error = reputation_assignment_error
        self.assessment_error = private_assessment_error
        self.reputation_update_rate = reputation_update_rate
        self.socialnorm = socialnorm  # matrix determining the reputation dynamic with
        # regard to the action taken and the reputation
        # of the other agent
        self.cost = cost  # cost defining the payoff matrix cost
        self.benefit = benefit  # benefit defined as the payoff matrix benefit

        # Population and reputation arrays
        self.population = np.zeros(population_size, dtype=int)  # vector of all individual strategies
        # population[k] : strategy of individual k
        # population[k] = 0, 1, 2 or 3
        self.reputation = np.zeros(population_size, dtype=int)  # vector of all individual public reputations
        # reputation[k] : public reputation of individual k
        # reputation[k] = 0 or 1

        # Cooperation Tracking
        self.coop_index_sum = float(0)
        self.interaction_count = float(0)

    def increment_coop_index(self, coop_index):
        self.coop_index_sum += float(coop_index)
        self.interaction_count += 1

    def get_average_coop_index(self):
        return float(self.coop_index_sum)/float(self.interaction_count)


"""
    Test class
"""
if __name__ == "__main__":
    variables = InstanceVariables(1, 1, 12, 0.001,
                                  0.08, 0.01,
                                  0.01, 1,
                                  [[0, 0], [0, 1]], 1, 5)
    print("Benefit: " + str(variables.benefit))
    variables.benefit += 5
    print("Benefit: " + str(variables.benefit))
    variables.increment_coop_index(0.5)
    variables.increment_coop_index(1)
    print(variables.get_average_coop_index())