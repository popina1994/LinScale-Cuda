#ifndef _LINSCALE_COMPUTE_H_
#define _LINSCALE_COMPUTE_H_

#include <iostream>
#include <fstream>
#include <string>
#include "matrix.h"
#include "matrix_cuda_general.h"


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