#ifndef _LINSCALE_COMPUTE_H_
#define _LINSCALE_COMPUTE_H_

#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <mkl_lapacke.h>
#include <mkl_cblas.h>
#include <mkl_vml.h>
#include "types.h"
#include "matrix.h"
#include "matrix_mkl.h"

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
template <typename T>
int computeFigaro(const T* h_mat1, const T* h_mat2, T* h_matR, int numRows1, int numCols1, int numRows2, int numCols2,
    const std::string& fileName, int compute);
template <typename T, MajorOrder majorOrder>
int computeGeneral(const T* h_A, T* h_matR, int numRows, int numCols, const std::string& fileName, int compute);

#endif