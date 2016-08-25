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
#include "SimInstance/SimInstance.h"

using namespace std;
void SantosSantosPacheco(int Z);

vector<vector<int>> SJ = {{1, 0},
                {0, 1}};

/*
 * 
 */
int main(int argc, char** argv) {
    SantosSantosPacheco(50);
    return 0;
}

void SantosSantosPacheco(int Z){
    int Runs = 1;
    int Generations = 3 * powf(10, 5);

    float mu = powf(10*Z, -1);
    float epsilon = 0.08;
    float alpha = 0.01;
    float Xerror = 0.01;
    float tau = 0.2;
    int randomseed = 1;
    vector<vector<int>> socialnorm = SJ;
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
    instance.RunInstance(Z);
}
