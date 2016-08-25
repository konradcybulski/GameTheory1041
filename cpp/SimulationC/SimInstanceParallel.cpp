/* 
 * File:   SimInstance.cpp
 * Author: kocyb_000
 * 
 * Created on 16 August 2016, 11:00 PM
 */

#include <cmath>
#include "SimInstance/SimInstance.h"
#include "SimInstance/SimInstanceParallel.h"
#include "stdlib.h"
#include "stdio.h"
#include "random"
#include <iostream>
#include "string"
#include "math.h"
#include "vector"
#include <thread>
#include "future"
#include "cstdio"

using namespace std;

SimInstanceParallel::SimInstanceParallel(int Runs, int Gens,
        float MutationRate,
        float ExecutionError,
        float ReputationAssignmentError,
        float PrivateAssessmentError,
        float ReputationUpdateProbability,
        int RandomSeed, vector<vector<int>> SocialNormMatrix,
        int CostValue, int BenefitValue) : SimInstance(Runs, Gens,
        MutationRate,
        ExecutionError,
        ReputationAssignmentError,
        PrivateAssessmentError,
        ReputationUpdateProbability,
        RandomSeed, SocialNormMatrix,
        CostValue, BenefitValue){}

void SimInstanceParallel::RunInstance(int Z){
    // Multiprocessing:
    int num_threads = 2;//thread.hardware_concurrency();
    
    for(int r=0; r<runs; r++){
        vector<int> Ptemp(Z);
        vector<int> Dtemp(Z);
        for(int i=0; i<Z; i++){
            Ptemp[i] = U(0, 3);
            Dtemp[i] = U(0, 1);
        }
        P = Ptemp;
        D = Dtemp;
        for(int t=0; t<gens; t++){
            // Update Progress:
            if(t % (gens / 100) == 0){
                cout << "Simulation at ";;//<< float(t/gens)*float(100) << std::endl;
                cout << t << "out of " << gens << endl;
            }

            int a = U(0, Z-1);
            if(RFloat() < mu)
                P[a] = U(0, 3);
            int b;
            do{
                b = U(0, Z-1);
            }while(b == a);

            vector<int> fitness_vec(2);
            fitness_vec[0] = 0;
            fitness_vec[1] = 0;
            int chunk_N = (2*Z)/num_threads;
            for(int i=0; i<num_threads; i++){
                //, a, b, chunk_N, Z, fitness_vec);
                
            }
            int Fa = fitness_vec[0];
            int Fb = fitness_vec[1];
            Fa /= 2*Z;
            Fb /= 2*Z;
            if(RFloat() < powf(1 + expf((float)(Fa - Fb)), -1))
                P[a] = P[b];
        }
    }
    cout << "Cooperation Index: " <<
            float(CooperationCount)/float(InteractionCount) << endl;
}

void SimInstanceParallel::InteractZ(int a, int b, int iterations, int Z,
                                vector<int> fitness_vec){
    for(int i=0; i<iterations; i++){
        int c;
        do{
            c = U(0, Z-1);
        }while(c == a);
        fitness_vec[0] += FitnessFunction(a, c);
        do{
            c = (0, Z-1);
        }while(c == b);
        fitness_vec[0] += FitnessFunction(b, c);
    }
}

void SimInstanceParallel::Test(void *threadid){
    long tid;
    tid = (long)threadid;
    cout << "Hello World! Thread ID, " << tid << endl;
    pthread_exit(NULL);
}
