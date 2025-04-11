#ifndef _LINSCALE_MATRIX_MKL_H_
#define _LINSCALE_MATRIX_MKL_H_

#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <mkl.h>
#include <mkl_lapacke.h>
#include <mkl_cblas.h>
#include <mkl_vml.h>
#include "matrix.h"

template <typename T, MajorOrder order>
void concatenateMatrices(const Matrix<T, order>& mat1, const Matrix<T, order>& mat2,
    Matrix<T, order>& matOut,
    bool horizontal = true)
{
    matOut = Matrix<T, order> {mat1.getNumRows() + mat2.getNumRows(), mat1.getNumCols()};
    int numRowsOut = matOut.getNumRows();
    int numColsOut = matOut.getNumCols();
    for (int rowIdx = 0; rowIdx < mat1.getNumRows(); rowIdx++)
    {
        for (int colIdx = 0; colIdx < numColsOut; colIdx++)
        {
            matOut(rowIdx, colIdx) = mat1(rowIdx, colIdx);
        }
    }

    for (int rowIdx = 0; rowIdx < mat2.getNumRows(); rowIdx++)
    {
        for (int colIdx = 0; colIdx < numColsOut; colIdx++)
        {
            matOut(rowIdx + mat1.getNumRows(), colIdx) = mat2(rowIdx, colIdx);
        }
    }
}

template<typename T, MajorOrder majorOrderA, MajorOrder majorOrderV>
Matrix<T, majorOrderA> computeMatrixVector(const Matrix<T, majorOrderA>& matA,
    const Matrix<T, majorOrderV>& vect, int numRows, int numCols,
    bool transpose = false)
{
    T alpha = 1.0;
    T beta = 0.0;
    CBLAS_TRANSPOSE aTran = (transpose ?  CblasTrans : CblasNoTrans);

    int rowsOut = (transpose ? numCols : numRows);
    Matrix<T, majorOrderA> matOut{rowsOut, 1};

    if constexpr (MajorOrder::ROW_MAJOR == majorOrderA)
    {
        cblas_dgemv(CblasRowMajor, aTran, numRows, numCols, alpha, matA.getDataC(), numCols, vect.getDataC(), 1, beta, matOut.getData(), 1);
    }
    else
    {
        cblas_dgemv(CblasColMajor, aTran, numRows, numCols, alpha, matA.getDataC(), numRows, vect.getDataC(), 1, beta, matOut.getData(), 1);
    }
    return matOut;
}


template<typename T, MajorOrder majorOrder>
void computeMatrixMatrix(T* pMat1, T* pMat2, T*& pOutMat, int numRows1, int numCols1,
    int numCols2, bool transpose = false)
{
    double alpha = 1.0, beta = 0.0;
    CBLAS_TRANSPOSE aTran = transpose ?  CblasTrans : CblasNoTrans;
    int cntOut = transpose ? numCols1 * numCols2 : numRows1 * numCols2;

    pOutMat = new T[cntOut];
    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        cblas_dgemm(CblasRowMajor, aTran, CblasNoTrans,
                    numRows1, numCols1, numCols2, alpha, pMat1, numCols1, pMat2, numCols2, beta, pOutMat, numCols2);
    }
    else
    {
        cblas_dgemm(CblasColMajor, aTran, CblasNoTrans,
                    numRows1, numCols1, numCols2, alpha, pMat1, numRows1, pMat2, numCols1, beta, pOutMat, numRows1);
    }
}

template<typename T, MajorOrder majorOrder>
void selfTransposeMatrixMultiplication(const T* pMat, T*& pOutMat, int numRows, int numCols)
{
    pOutMat = new T[numCols * numCols];
    double alpha = 1.0, beta = 0.0;
    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    numCols, numCols, numRows, alpha, pMat, numCols, pMat, numCols, beta,
                     pOutMat, numCols);
    }
    else
    {
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
            numCols, numCols, numRows, alpha, pMat, numRows, pMat, numRows, beta,
                     pOutMat, numCols);
    }
}

template<typename T, MajorOrder majorOrder>
void selfMatrixTransposeMultiplication(const Matrix<T, majorOrder>& matA,
    Matrix<T, majorOrder>& matOut)
{
    double alpha = 1.0, beta = 0.0;
    int numRows = matA.getNumRows();
    int numCols = matA.getNumCols();

    matOut = std::move(Matrix<T, majorOrder>(numCols, numCols));
    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    numCols, numCols, numRows, alpha, matA.getDataC(),
                        numCols, matA.getDataC(), numCols, beta,
                     matOut.getData(), numCols);
    }
    else
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
            numCols, numCols, numRows, alpha, matA.getDataC(), numRows, matA.getDataC(), numRows, beta,
                     matOut.getData(), numCols);
    }
}

template<typename T, MajorOrder majorOrder>
void computeInverse(Matrix<T, majorOrder>& matA, int numRows, int numCols, bool upperTriangular = true)
{
    if (upperTriangular)
    {
        if constexpr (MajorOrder::COL_MAJOR == majorOrder)
        {
            LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', numRows, matA.getData(), numRows);
        }
        else
        {
            LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', numRows, matA.getData(), numCols);
        }
    }
    else
    {
        int N = 3;  // Matrix size
        int LDA = 3, info;
        int *ipiv = new int [numCols];  // Pivot indices
        if constexpr (MajorOrder::ROW_MAJOR == majorOrder)
        {
            // Step 1: Perform LU decomposition
            LAPACKE_dgetrf(LAPACK_ROW_MAJOR, numRows, numRows, matA.getData(), numCols, ipiv);

            // Step 2: Compute inverse using LU factorization
            LAPACKE_dgetri(LAPACK_ROW_MAJOR, numRows, matA.getData(), numCols, ipiv);
        }
        else
        {
            // Step 1: Perform LU decomposition
            LAPACKE_dgetrf(LAPACK_COL_MAJOR, numRows, numRows, matA.getData(), numCols, ipiv);

            // Step 2: Compute inverse using LU factorization
            LAPACKE_dgetri(LAPACK_COL_MAJOR, numRows, matA.getData(), numCols, ipiv);
        }
        delete [] ipiv;
    }

}

// Ax = b --- pMatA * pOutVect = pVectB,
// = A^T * A * x = A^T * b
// x = (A^T * A)^ inv * A^T * b
// A^T * A = R^T * R
template<typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> solveLLSNormalEquation(const Matrix<T, majorOrder>& matA,
    const Matrix<T, majorOrder>& matR, const Matrix<T, majorOrder>& vectB,
    int numRows, int numCols, const std::string& fileName)
{
    Matrix<T, majorOrder> matRCopy{numCols, numCols};
    Matrix<T, majorOrder> outMat{numCols, numCols};
    copyMatrix<T, majorOrder>(matR.getDataC(), matRCopy.getData(), numCols, numCols, numCols, numCols, true);
    computeInverse<double, majorOrder>(matRCopy, numCols, numCols, true);
    selfMatrixTransposeMultiplication(matRCopy, outMat);

    auto tempVect = computeMatrixVector(matA, vectB, numRows, numCols, true);
    auto vectXOut = computeMatrixVector(outMat, tempVect, numCols, numCols, false);

    return vectXOut;
}

template<typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> solveLLSMKL(const Matrix<T, majorOrder>& matA,
    const Matrix<T, majorOrder>& matR, const Matrix<T, majorOrder>& vectB,
    int numRows, int numCols, const std::string& fileName)
{
    Matrix<T, majorOrder> matCopy{matA.getNumRows(), matA.getNumCols()};
    Matrix<T, majorOrder> vectXOut{vectB.getNumRows(), vectB.getNumCols()};

    copyMatrix<T, majorOrder>(vectB.getDataC(), vectXOut.getData(), vectB.getNumRows(), vectB.getNumCols(), vectB.getNumRows(), vectB.getNumCols(), false);
    copyMatrix<T, majorOrder>(matA.getDataC(), matCopy.getData(), matA.getNumRows(), matA.getNumCols(), matA.getNumRows(), matA.getNumCols(), false);

    if constexpr (majorOrder == MajorOrder::COL_MAJOR)
    {
        LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', matA.getNumRows(), matA.getNumCols(), 1,
            matCopy.getData(), matA.getNumRows(), vectXOut.getData(), vectXOut.getNumRows());
    }
    else
    {
        LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', matA.getNumRows(), matA.getNumCols(), 1,
            matCopy.getData(), matA.getNumCols(), vectXOut.getData(), 1);
    }
    return vectXOut;
}

template<typename T, MajorOrder order>
void fillValues(Matrix<T, order>& mat, T elem)
{
    if constexpr (order == MajorOrder::ROW_MAJOR)
    {
        for (int rowIdx = 0; rowIdx < mat.getNumRows(); rowIdx++)
        {
            for (int colIdx = 0; colIdx < mat.getNumCols(); colIdx++)
            {
                mat(rowIdx, colIdx) = elem;
            }
        }
    }
    else
    {
        for (int colIdx = 0; colIdx < mat.getNumCols(); colIdx++)
        {
            for (int rowIdx = 0; rowIdx < mat.getNumRows(); rowIdx++)
            {
                mat(rowIdx, colIdx) = elem;
            }
        }
    }
}


#endif