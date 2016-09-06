/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   SimMain.cpp
 * Author: kocyb_000
 *
 * Created on 17 August 2016, 1:34 PM
 */

#include <cstdlib>
#include <math.h>
#include "stdio.h"
#include "SimInstance.h"
#include <vector>
#include <chrono>
#include "iostream"
#include <thread>

using namespace std;
void SantosSantosPacheco(int Z);

vector<vector<int>> SJ = {{1, 0},
                {0, 1}};

/*
 * 
 */
int main(int argc, char** argv) {
    SantosSantosPacheco(30);
    return 0;
}

void SantosSantosPacheco(int Z){
    int Runs = 1;
    int Generations = 3 * powf(10, 3);

    float mu = powf(10*Z, -1);
    float epsilon = 0.08;
    float alpha = 0.01;
    float Xerror = 0.01;
    float tau = 0.2;
    int randomseed = 4;
	vector<vector<int>> socialnorm = { {0, 0}, {0, 1} }; //SJ;
    int cost = 1;
    int benefit = 5;
    
    SimInstance instance(Runs, Generations,
            mu,
            epsilon,
            alpha,
            Xerror,
            tau,
            randomseed,
            socialnorm,
            cost,
            benefit);
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	instance.RunInstance(Z);
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	auto duration1 = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
	auto average_time = duration1;

	t1 = std::chrono::high_resolution_clock::now();
	instance.RunInstance(Z);
	t2 = std::chrono::high_resolution_clock::now();
	duration1 = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
	average_time += duration1;

	t1 = std::chrono::high_resolution_clock::now();
	instance.RunInstance(Z);
	t2 = std::chrono::high_resolution_clock::now();
	duration1 = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
	average_time += duration1;

	cout << "Single Thread: " << average_time/float(3) << " seconds" << endl;
	std::getchar();
}
