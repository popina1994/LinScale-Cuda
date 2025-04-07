#ifndef _LINSCALE_COMPUTE_H_
#define _LINSCALE_COMPUTE_H_

#include <iostream>
#include <fstream>
#include <string>
#include "matrix_mkl.h"
#include "matrix_cuda_general.h"

template <typename T>
double computeMeanSquaredError(const T* pA, const T* pB, int numRows)
{
    double* diff = new double[numRows];
    double* squared = new double[numRows];
      // Compute element-wise difference: diff = a - b
    vdSub(numRows, pA, pB, diff);

    // Square each element: squared = diff^2
    vdMul(numRows, diff, diff, squared);

    // Compute sum of squared differences
    double sum_sq = cblas_dasum(numRows, squared, 1);

    // Compute MSE
    double mse = sum_sq / numRows;
    delete [] diff;
    delete [] squared;

    return mse;
}


void readCSV(const std::string& fileName, double *h_mat)
{

    std::ifstream file(fileName);

    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
    }

    std::string line;
    int idx = 0;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        // Split each line by comma (',') and store values in a vector
        while (getline(ss, value, ',')) {
            h_mat[idx++] = std::stoi(value);
        }
    }
}


#endif