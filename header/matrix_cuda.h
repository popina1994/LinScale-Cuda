#ifndef _LINSCALE_CUDA_H_
#define _LINSCALE_CUDA_H_

#include "matrix.h"
#include <string>

template <typename T>
int computeFigaro(const MatrixDRow& mat1, const MatrixDRow& mat2,
    Matrix<T, MajorOrder::COL_MAJOR>& matR, const std::string& fileName, int compute);
template <typename T, MajorOrder majorOrder>
int computeGeneral(const Matrix<T, majorOrder>& matA,
    Matrix<T, MajorOrder::COL_MAJOR>& matR, const std::string& fileName, int compute);

#endif