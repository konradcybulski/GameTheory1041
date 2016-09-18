import unittest
from InstanceVariables import InstanceVariables
import SimulationInstance
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_fitness_function_action(self):
        runs = 1
        generations = 1
        population_size = 2
        mutation_rate = 0.0
        execution_error = 0.0
        reputation_assignment_error = 0.0
        private_assessment_error = 0.0
        reputation_update_prob = 1.0
        socialnorm = [[1, 0], [0, 1]]
        cost = 1
        benefit = 5
        variables = InstanceVariables(runs, generations, population_size, mutation_rate,
                                           execution_error, reputation_assignment_error,
                                           private_assessment_error, reputation_update_prob,
                                           socialnorm, cost, benefit)

        # pop , rep, payoff_expected, strategy string of A, strategy string of B
        test_cases = [
                    # Test AllC and AllD
                      [[3, 3], [0, 0], benefit - cost, "AllC", "AllC"],
                      [[0, 3], [0, 0], benefit, "AllD", "AllC"],
                      [[3, 0], [0, 0], -cost, "AllC", "AllD"],
                      [[0, 0], [0, 0], 0, "AllD", "AllD"],

                    # Test Disc
                      [[0, 2], [1, 0], benefit, "AllD_G", "Disc"],
                      [[0, 2], [0, 0], 0, "AllD_B", "Disc"],
                      [[3, 2], [1, 0], benefit - cost, "AllC_G", "Disc"],
                      [[3, 2], [0, 0], -cost, "AllC_B", "Disc"],

                      [[1, 2], [1, 1], benefit, "pDisc_G", "Disc_G"],
                      [[1, 2], [0, 1], 0, "pDisc_B", "Disc_G"],
                      [[1, 2], [1, 0], benefit - cost, "pDisc_G", "Disc_B"],
                      [[1, 2], [0, 0], -cost, "pDisc_B", "Disc_B"],

                      [[2, 2], [1, 1], benefit - cost, "Disc_G", "Disc_G"],
                      [[2, 2], [1, 0], benefit, "Disc_G", "Disc_B"],
                      [[2, 2], [0, 1], -cost, "Disc_B", "Disc_G"],
                      [[2, 2], [0, 0], 0, "Disc_B", "Disc_B"],

                    # Test pDisc
                      [[0, 1], [1, 0], 0, "AllD_G", "pDisc"],
                      [[0, 1], [0, 0], benefit, "AllD_B", "pDisc"],
                      [[3, 1], [1, 0], -cost, "AllC_G", "pDisc"],
                      [[3, 1], [0, 0], benefit - cost, "AllC_B", "pDisc"],

                      [[1, 1], [1, 1], 0, "pDisc_G", "pDisc_G"],
                      [[1, 1], [1, 0], -cost, "pDisc_G", "pDisc_B"],
                      [[1, 1], [0, 1], benefit, "pDisc_B", "pDisc_G"],
                      [[1, 1], [0, 0], benefit - cost, "pDisc_B", "pDisc_B"],

                      [[2, 1], [1, 1], -cost, "Disc_G", "pDisc_G"],
                      [[2, 1], [0, 1], benefit - cost, "Disc_B", "pDisc_G"],
                      [[2, 1], [1, 0], 0, "Disc_G", "pDisc_B"],
                      [[2, 1], [0, 0], benefit, "Disc_B", "pDisc_B"],
                      ]

        for test_case in test_cases:
            variables.population = test_case[0]
            variables.reputation = test_case[1]
            payoff = SimulationInstance.fitness_function(0, 1, variables)
            self.assertEqual(payoff, test_case[2], "Payoff for " + test_case[3] + " playing against " + test_case[4] +
                             " should be " + str(test_case[2]) + " but is " + str(payoff))

    def test_fitness_function_reputation_sj(self):
        runs = 1
        generations = 1
        population_size = 2
        mutation_rate = 0.0
        execution_error = 0.0
        reputation_assignment_error = 0.0
        private_assessment_error = 0.0
        reputation_update_prob = 1.0
        socialnorm = [[1, 0], [0, 1]]
        cost = 1
        benefit = 5
        variables = InstanceVariables(runs, generations, population_size, mutation_rate,
                                      execution_error, reputation_assignment_error,
                                      private_assessment_error, reputation_update_prob,
                                      socialnorm, cost, benefit)

        # pop , rep, expected rep, strategy string of A, strategy string of B
        test_cases = [
                    # Test AllC and AllD
                      [[3, 3], [1, 1], [1, 1], "AllC_G", "AllC_G"],
                      [[3, 3], [0, 1], [1, 0], "AllC_B", "AllC_G"],
                      [[3, 3], [1, 0], [0, 1], "AllC_G", "AllC_B"],
                      [[3, 3], [0, 0], [0, 0], "AllC_B", "AllC_B"],

                      [[3, 0], [1, 1], [1, 0], "AllC_G", "AllD_G"],
                      [[3, 0], [0, 1], [1, 1], "AllC_B", "AllD_G"],
                      [[3, 0], [1, 0], [0, 0], "AllC_G", "AllD_B"],
                      [[3, 0], [0, 0], [0, 1], "AllC_B", "AllD_B"],

                      [[0, 0], [1, 1], [0, 0], "AllD_G", "AllD_G"],
                      [[0, 0], [0, 1], [0, 1], "AllD_B", "AllD_G"],
                      [[0, 0], [1, 0], [1, 0], "AllD_G", "AllD_B"],
                      [[0, 0], [0, 0], [1, 1], "AllD_B", "AllD_B"],

                    # Test Disc
                      [[0, 2], [1, 1], [0, 1], "AllD_G", "Disc_G"],
                      [[0, 2], [0, 1], [0, 1], "AllD_B", "Disc_G"],
                      [[0, 2], [1, 0], [1, 1], "AllD_G", "Disc_B"],
                      [[0, 2], [0, 0], [1, 1], "AllD_B", "Disc_B"],

                      [[3, 2], [1, 1], [1, 1], "AllC_G", "Disc_G"],
                      [[3, 2], [0, 1], [1, 1], "AllC_B", "Disc_G"],
                      [[3, 2], [1, 0], [0, 1], "AllC_G", "Disc_B"],
                      [[3, 2], [0, 0], [0, 1], "AllC_B", "Disc_B"],

                      [[2, 2], [1, 1], [1, 1], "Disc_G", "Disc_G"],
                      [[2, 2], [0, 1], [1, 1], "Disc_B", "Disc_G"],
                      [[2, 2], [1, 0], [1, 1], "Disc_G", "Disc_B"],
                      [[2, 2], [0, 0], [1, 1], "Disc_B", "Disc_B"],

                      [[1, 2], [1, 1], [0, 1], "pDisc_G", "Disc_G"],
                      [[1, 2], [0, 1], [0, 1], "pDisc_B", "Disc_G"],
                      [[1, 2], [1, 0], [0, 1], "pDisc_G", "Disc_B"],
                      [[1, 2], [0, 0], [0, 1], "pDisc_B", "Disc_B"],

                    # Test pDisc
                      [[0, 1], [1, 1], [0, 0], "AllD_G", "pDisc_G"],
                      [[0, 1], [0, 1], [0, 0], "AllD_B", "pDisc_G"],
                      [[0, 1], [1, 0], [1, 0], "AllD_G", "pDisc_B"],
                      [[0, 1], [0, 0], [1, 0], "AllD_B", "pDisc_B"],

                      [[3, 1], [1, 1], [1, 0], "AllC_G", "pDisc_G"],
                      [[3, 1], [0, 1], [1, 0], "AllC_B", "pDisc_G"],
                      [[3, 1], [1, 0], [0, 0], "AllC_G", "pDisc_B"],
                      [[3, 1], [0, 0], [0, 0], "AllC_B", "pDisc_B"],

                      [[2, 1], [1, 1], [1, 0], "Disc_G", "pDisc_G"],
                      [[2, 1], [0, 1], [1, 0], "Disc_B", "pDisc_G"],
                      [[2, 1], [1, 0], [1, 0], "Disc_G", "pDisc_B"],
                      [[2, 1], [0, 0], [1, 0], "Disc_B", "pDisc_B"],

                      [[1, 1], [1, 1], [0, 0], "pDisc_G", "pDisc_G"],
                      [[1, 1], [0, 1], [0, 0], "pDisc_B", "pDisc_G"],
                      [[1, 1], [1, 0], [0, 0], "pDisc_G", "pDisc_B"],
                      [[1, 1], [0, 0], [0, 0], "pDisc_B", "pDisc_B"],
                      ]

        for test_case in test_cases:
            variables.population = test_case[0]
            variables.reputation = test_case[1]
            payoff = SimulationInstance.fitness_function(0, 1, variables)
            new_rep = variables.reputation
            self.assertEqual(new_rep, test_case[2], "Reputations for " + test_case[3] + " playing against " + test_case[4] +
                             ", with social norm + " + str(variables.socialnorm) + " should be " +
                             str(test_case[2]) + " but is " + str(new_rep))

    def test_fitness_function_reputation_ss(self):
        runs = 1
        generations = 1
        population_size = 2
        mutation_rate = 0.0
        execution_error = 0.0
        reputation_assignment_error = 0.0
        private_assessment_error = 0.0
        reputation_update_prob = 1.0
        socialnorm = [[1, 1], [0, 1]]
        cost = 1
        benefit = 5
        variables = InstanceVariables(runs, generations, population_size, mutation_rate,
                                      execution_error, reputation_assignment_error,
                                      private_assessment_error, reputation_update_prob,
                                      socialnorm, cost, benefit)

        # pop , rep, expected rep, strategy string of A, strategy string of B
        test_cases = [
                    # Test AllC and AllD
                      [[3, 3], [1, 1], [1, 1], "AllC_G", "AllC_G"],
                      [[3, 3], [0, 1], [1, 1], "AllC_B", "AllC_G"],
                      [[3, 3], [1, 0], [1, 1], "AllC_G", "AllC_B"],
                      [[3, 3], [0, 0], [1, 1], "AllC_B", "AllC_B"],

                      [[3, 0], [1, 1], [1, 0], "AllC_G", "AllD_G"],
                      [[3, 0], [0, 1], [1, 1], "AllC_B", "AllD_G"],
                      [[3, 0], [1, 0], [1, 0], "AllC_G", "AllD_B"],
                      [[3, 0], [0, 0], [1, 1], "AllC_B", "AllD_B"],

                      [[0, 0], [1, 1], [0, 0], "AllD_G", "AllD_G"],
                      [[0, 0], [0, 1], [0, 1], "AllD_B", "AllD_G"],
                      [[0, 0], [1, 0], [1, 0], "AllD_G", "AllD_B"],
                      [[0, 0], [0, 0], [1, 1], "AllD_B", "AllD_B"],

                    # Test Disc
                      [[0, 2], [1, 1], [0, 1], "AllD_G", "Disc_G"],
                      [[0, 2], [0, 1], [0, 1], "AllD_B", "Disc_G"],
                      [[0, 2], [1, 0], [1, 1], "AllD_G", "Disc_B"],
                      [[0, 2], [0, 0], [1, 1], "AllD_B", "Disc_B"],

                      [[3, 2], [1, 1], [1, 1], "AllC_G", "Disc_G"],
                      [[3, 2], [0, 1], [1, 1], "AllC_B", "Disc_G"],
                      [[3, 2], [1, 0], [1, 1], "AllC_G", "Disc_B"],
                      [[3, 2], [0, 0], [1, 1], "AllC_B", "Disc_B"],

                      [[2, 2], [1, 1], [1, 1], "Disc_G", "Disc_G"],
                      [[2, 2], [0, 1], [1, 1], "Disc_B", "Disc_G"],
                      [[2, 2], [1, 0], [1, 1], "Disc_G", "Disc_B"],
                      [[2, 2], [0, 0], [1, 1], "Disc_B", "Disc_B"],

                      [[1, 2], [1, 1], [0, 1], "pDisc_G", "Disc_G"],
                      [[1, 2], [0, 1], [0, 1], "pDisc_B", "Disc_G"],
                      [[1, 2], [1, 0], [1, 1], "pDisc_G", "Disc_B"],
                      [[1, 2], [0, 0], [1, 1], "pDisc_B", "Disc_B"],

                    # Test pDisc
                      [[0, 1], [1, 1], [0, 0], "AllD_G", "pDisc_G"],
                      [[0, 1], [0, 1], [0, 1], "AllD_B", "pDisc_G"],
                      [[0, 1], [1, 0], [1, 0], "AllD_G", "pDisc_B"],
                      [[0, 1], [0, 0], [1, 1], "AllD_B", "pDisc_B"],

                      [[3, 1], [1, 1], [1, 0], "AllC_G", "pDisc_G"],
                      [[3, 1], [0, 1], [1, 1], "AllC_B", "pDisc_G"],
                      [[3, 1], [1, 0], [1, 0], "AllC_G", "pDisc_B"],
                      [[3, 1], [0, 0], [1, 1], "AllC_B", "pDisc_B"],

                      [[2, 1], [1, 1], [1, 0], "Disc_G", "pDisc_G"],
                      [[2, 1], [0, 1], [1, 1], "Disc_B", "pDisc_G"],
                      [[2, 1], [1, 0], [1, 0], "Disc_G", "pDisc_B"],
                      [[2, 1], [0, 0], [1, 1], "Disc_B", "pDisc_B"],

                      [[1, 1], [1, 1], [0, 0], "pDisc_G", "pDisc_G"],
                      [[1, 1], [0, 1], [0, 1], "pDisc_B", "pDisc_G"],
                      [[1, 1], [1, 0], [1, 0], "pDisc_G", "pDisc_B"],
                      [[1, 1], [0, 0], [1, 1], "pDisc_B", "pDisc_B"],
                      ]

        for test_case in test_cases:
            variables.population = test_case[0]
            variables.reputation = test_case[1]
            payoff = SimulationInstance.fitness_function(0, 1, variables)
            new_rep = variables.reputation
            self.assertEqual(new_rep, test_case[2], "Reputations for " + test_case[3] + " playing against " + test_case[4] +
                             ", with social norm + " + str(variables.socialnorm) + " should be " +
                             str(test_case[2]) + " but is " + str(new_rep))


if __name__ == '__main__':
    unittest.main()
