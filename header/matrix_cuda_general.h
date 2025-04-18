#ifndef _LINSCALE_MATRIX_GENERAL_CUDA_H_
#define _LINSCALE_MATRIX_GENERAL_CUDA_H_

#include "matrix.h"
#include <string>

extern std::map<std::string, double> mapMemoryTrack;

template <typename T>
int computeFigaro(const MatrixRow<T>& mat1, const MatrixRow<T>& mat2,
    MatrixCol<T>& matR, MatrixCol<T>& matQ,  MatrixCol<T>& matU,
    MatrixCol<T>& matSigma, MatrixCol<T>& matV, ComputeDecomp decompType);
template <typename T, MajorOrder majorOrder>
int computeGeneral(const Matrix<T, majorOrder>& matA,
    MatrixCol<T>& matR, MatrixCol<T>& matQ, MatrixCol<T>& matU,
    MatrixCol<T>& matSigma, MatrixCol<T>& matV, ComputeDecomp decompType);

template <typename T, MajorOrder majorOrder>
int solveLLSNormalEquationUsingR(const Matrix<T, majorOrder>& matA,
    const Matrix<T, majorOrder>& matR,
    const Matrix<T, majorOrder>& vectB,
    Matrix<T, majorOrder>& vectX);

template <typename T, MajorOrder majorOrder>
int solveLLSQRDecomp(const Matrix<T, majorOrder>& matA,
    const Matrix<T, majorOrder>& vectB,
    Matrix<T, majorOrder>& vectX);

#endif