/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   SimInstanceParallel.h
 * Author: kocyb_000
 *
 * Created on 17 August 2016, 2:23 PM
 */

#ifndef SIMINSTANCEPARALLEL_H
#define SIMINSTANCEPARALLEL_H

#include "vector"
#include "SimInstance.h"
#include "stdio.h"
using namespace std;

class SimInstanceParallel : public SimInstance {
public:
    SimInstanceParallel(int Runs, int Gens,
        float MutationRate,
        float ExecutionError,
        float ReputationAssignmentError,
        float PrivateAssessmentError,
        float ReputationUpdateProbability,
        int RandomSeed, vector<vector<int>> SocialNormMatrix,
        int CostValue, int BenefitValue);
protected:
    void RunInstance(int Z);
    void InteractZ(int a, int b, int iterations,
            int Z, vector<int> fitness_vec);
    void Test(void *threadid);
};

#endif /* SIMINSTANCEPARALLEL_H */

