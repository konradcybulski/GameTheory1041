/* 
 * File:   SimInstance.h
 * Author: kocyb_000
 *
 * Created on 16 August 2016, 11:00 PM
 */

#ifndef SIMINSTANCE_H
#define SIMINSTANCE_H

#include "vector"
#include "stdio.h"
using namespace std;

class SimInstance {
public:
    SimInstance(int Runs, int Gens,
        float MutationRate,
        float ExecutionError,
        float ReputationAssignmentError,
        float PrivateAssessmentError,
        float ReputationUpdateProbability,
        int RandomSeed, vector<vector<int>> SocialNormMatrix,
        int CostValue, int BenefitValue);
    void RunInstance(int Z);
	void RunInstanceParallel(int Z);
protected:
    float RFloat();
    int U(int a, int b);
	int FitnessFunction(int x, int y);
	int ReputationFunction(vector<vector<int>> socialnorm_matrix,
		int action_x, int rep_y);
	vector<int> InteractZ(int a, int b, int iterations, int Z);
	void Add();
};

#endif /* SIMINSTANCE_H */

