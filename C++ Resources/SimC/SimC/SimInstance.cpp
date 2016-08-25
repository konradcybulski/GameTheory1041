/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   SimInstance.cpp
 * Author: kocyb_000
 * 
 * Created on 16 August 2016, 11:00 PM
 */

#include <cmath>

#include "SimInstance.h"
#include "stdlib.h"
#include "stdio.h"
#include "random"
#include "iostream"
#include "string"
#include "math.h"
#include "vector"
#include "thread"
#include "future"

using namespace std;

// Static Vars
vector<vector<int>> Strategies = {{0,0}, {0,1}, {1,0}, {1,1}};

// Class Variables
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

// Tracking Variables:
int CooperationCount;
int InteractionCount;

float SimInstance::RFloat(){ return (float)rand() / (float)RAND_MAX; }
int SimInstance::U(int a, int b){
    return a + rand() % (b - a + 1);
}

SimInstance::SimInstance(int Runs, int Gens,
        float MutationRate,
        float ExecutionError,
        float ReputationAssignmentError,
        float PrivateAssessmentError,
        float ReputationUpdateProbability,
        int RandomSeed, vector<vector<int>> SocialNormMatrix,
        int CostValue, int BenefitValue) {
    runs = Runs;
    gens = Gens;
    mu = MutationRate;
    epsilon = ExecutionError;
    alpha = ReputationAssignmentError;
    Xerror = PrivateAssessmentError;
    tau = ReputationUpdateProbability;
    srand( RandomSeed );
    socialnorm = SocialNormMatrix;
    cost = CostValue;
    benefit = BenefitValue;

    CooperationCount = 0;
    InteractionCount = 0;
}

void SimInstance::RunInstance(int Z){
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
            if(t % (gens / 5) == 0){
                cout << "Simulation at ";;//<< float(t/gens)*float(100) << std::endl;
                cout << t << " out of " << gens << endl;
            }

            int a = U(0, Z-1);
            if(RFloat() < mu)
                P[a] = U(0, 3);
            int b;
            do{
                b = U(0, Z-1);
            }while(b == a);

            int Fa = 0;
            int Fb = 0;
            for(int i=0; i<2*Z; i++){
                int c;
                do{
                    c = U(0, Z-1);
                }while(c == a);
                Fa += FitnessFunction(a, c);

                do{
                    c = U(0, Z-1);
                }while(c == b);
				Fb += FitnessFunction(b, c);
            }
            Fa /= 2*Z;
            Fb /= 2*Z;
            if(RFloat() < powf(1 + expf((float)(Fa - Fb)), -1))
                P[a] = P[b];
        }
    }
    cout << "Cooperation Index: " <<
            float(CooperationCount)/float(InteractionCount) << endl;
}

int SimInstance::FitnessFunction(int x, int y){
    vector<int>& XStrategy = Strategies[P[x]];
    vector<int>& YStrategy = Strategies[P[y]];
    int Cx;
    int Cy;
    if(RFloat() < Xerror){
        if (RFloat() < epsilon && XStrategy[1 - D[y]] == 1){
            Cx = 1 - XStrategy[1 - D[y]];
        }else{
            Cx = XStrategy[1 - D[y]];
        }
    }else{
        if (RFloat() < epsilon && XStrategy[D[y]] == 1){
            Cx = 1 - XStrategy[D[y]];
        }else{
            Cx = XStrategy[D[y]];
        }
    }
    if (RFloat() < Xerror){
        if (RFloat() < epsilon && YStrategy[1 - D[x]] == 1){
            Cy = 1 - YStrategy[1 - D[x]];
        }else{
            Cy = YStrategy[1 - D[x]];
        }
    }else{
        if (RFloat() < epsilon && YStrategy[D[x]] == 1){
            Cy = 1 - YStrategy[D[x]];
        }else{
            Cy = YStrategy[D[x]];
        }
    }
    // Update Reputation
    // X
    if (RFloat() < tau){
        if (RFloat() < alpha){
            D[x] = 1 - ReputationFunction(socialnorm, Cx, D[y]);
        }else{
            D[x] = ReputationFunction(socialnorm, Cx, D[y]);
        }
    }
    // Y
    if (RFloat() < tau){
        if (RFloat() < alpha){
            D[y] = 1 - ReputationFunction(socialnorm, Cy, D[x]);
        }else{
            D[y] = ReputationFunction(socialnorm, Cy, D[x]);
        }
    }
    // Track cooperation
	InteractionCount += 2;
	CooperationCount += Cx == 1 ? 1 : 0;
	CooperationCount += Cy == 1 ? 1 : 0;
	//vector<int> return_vec = { (benefit * Cy) - (cost * Cx), coop_count, 2 };
	return (benefit * Cy) - (cost * Cx);
}

int SimInstance::ReputationFunction(vector<vector<int>> socialnorm_matrix, int action_x, int rep_y){
    return socialnorm_matrix[1 - action_x][1 - rep_y];
}

void SimInstance::RunInstanceParallel(int Z){
	int num_threads = std::thread::hardware_concurrency(); //std::thread.hardware_concurrency();

	for (int r = 0; r<runs; r++) {
		vector<int> Ptemp(Z);
		vector<int> Dtemp(Z);
		for (int i = 0; i<Z; i++) {
			Ptemp[i] = U(0, 3);
			Dtemp[i] = U(0, 1);
		}
		P = Ptemp;
		D = Dtemp;
		for (int t = 0; t<gens; t++) {
			// Update Progress:
			if (t % (gens / 5) == 0) {
				cout << "Simulation at ";;//<< float(t/gens)*float(100) << std::endl;
				cout << t << " out of " << gens << endl;
			}

			int a = U(0, Z - 1);
			if (RFloat() < mu)
				P[a] = U(0, 3);
			int b;
			do {
				b = U(0, Z - 1);
			} while (b == a);

			int Fa = 0;
			int Fb = 0;
			int chunksize = 2*Z / num_threads;

			
			vector<future<vector<int>>> futures(num_threads);

			for (int i = 0; i < num_threads; i++) {
				// std::thread(&SimInstance::Add, this); 
				futures[i] = std::async(&SimInstance::InteractZ, this, a, b, chunksize, Z); ///(&SimInstance::InteractZ, a, b, chunksize, Z); //std::async(InteractZ, a, b, chunksize, Z);
			}

			for (int i = 0; i < num_threads; i++) {
				vector<int> result = futures[i].get(); //.join();
				Fa += result[0];
				Fb += result[1];
			}
			Fa /= 2 * Z;
			Fb /= 2 * Z;
			if (RFloat() < powf(1 + expf((float)(Fa - Fb)), -1))
				P[a] = P[b];
		}
	}
	cout << "Cooperation Index: " <<
		float(CooperationCount) / float(InteractionCount) << endl;
}

// return is of type vector<int> {Fa, Fb, CoopCount, InteractionCount}
vector<int> SimInstance::InteractZ(int a, int b, int iterations, int Z) {
	vector<int> output_vec = { 0, 0 };
	for (int i = 0; i<iterations; i++) {
		int c;
		do {
			c = U(0, Z - 1);
		} while (c == a);
		output_vec[0] += FitnessFunction(a, c);

		do {
			c = U(0, Z - 1);
		} while (c == b);
		output_vec[1] += FitnessFunction(b, c);
	}
	return output_vec;
}

void SimInstance::Add() {
	int a = 0;
}