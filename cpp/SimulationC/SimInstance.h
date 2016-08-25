/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

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
protected:
    float RFloat();
    int U(int a, int b);
    int FitnessFunction(int x, int y);
    int ReputationFunction(vector<vector<int>> socialnorm_matrix, 
        int action_x, int rep_y);
    
    vector<vector<int>> Strategies;
    int runs;
    int gens;
    vector<int> P;
    vector<int> D;
    float mu;
    float epsilon;
    float alpha;
    float Xerror;
    float tau;
    int RandomSeed;
    vector<vector<int>> socialnorm;
    int cost;
    int benefit;
    int CooperationCount;
    int InteractionCount;

};

#endif /* SIMINSTANCE_H */

