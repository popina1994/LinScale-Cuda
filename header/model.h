#ifndef _LINSCALE_MODEL_H_
#define _LINSCALE_MODEL_H_

#include "compute.h"

void evaluateTrain(const MatrixDRow& mat1, const MatrixDRow& mat2,
    const MatrixDCol& matJoin, MatrixDCol& matCUDAR, MatrixDCol& matFigR,
    MatrixDCol& matFigQ, MatrixDCol& matCUDAQ,
    const std::string& fileName, ComputeDecomp decompType)
{
    /*********** TRAINING ***********************/
    //  printMatrix<double, MajorOrder::COL_MAJOR>(matJoin, matJoin.getNumRows(), fileName + "Join.csv", false);
    computeGeneral<double, MajorOrder::COL_MAJOR>(matJoin, matCUDAR, matCUDAQ,
        fileName, decompType);
    std::cout << "ORTHOGONALITY " << matCUDAQ.computeOrthogonality() << std::endl;
    // printMatrix<double, MajorOrder::COL_MAJOR>(matCUDAR, matCUDAR.getNumCols(), fileName + "CUDA.csv", false);

    computeFigaro<double>(mat1, mat2, matFigR, matFigQ, fileName, decompType);
    // printMatrix<double, MajorOrder::COL_MAJOR>(matFigR, matFigR.getNumCols(), fileName + "LinScale.csv", false);
}

void computeVectors(const MatrixDCol& matJoin, const MatrixDCol& matCUDAR,
    const MatrixDCol& matFigR, const MatrixDCol& vectBTrain,
    MatrixDCol& vectXCompMKL, MatrixDCol& vectXCompFig)
{
    // vectXCompMKL = matJoin.solveLLSNormalEquationUsingR(matCUDAR, vectBTrain);
    vectXCompMKL = matJoin.solveLLSNormalEquations(vectBTrain);
    // vectXCompMKL = matJoin.solveLLSQRDecomp(vectBTrain);
    vectXCompFig = matJoin.solveLLSNormalEquationUsingR(matFigR, vectBTrain);
}

void evaluateTest(int numRows, int numCols,
    const MatrixDCol& vectX, MatrixDCol& vectXCompMKL,
    MatrixDCol& vectXCompFig, int seed)
{
    auto matRandTest = MatrixDCol::generateRandom(numRows, numCols, seed + 22);

    auto outVectBTest = matRandTest.computeMatrixVector(vectX, false);
    auto matUniformAdd = MatrixDCol::generateRandom(
        matRandTest.getNumRows(), matRandTest.getNumCols(), seed + 37);
    auto matUniformCopy = matUniformAdd.divValue(1e10);
    auto outVectBTestVariance = outVectBTest.add(matUniformCopy);
    // outVectBTestVariance = std::move(outVectBTest);

    auto outVectBTestCompMKL = matRandTest.computeMatrixVector(vectXCompMKL, false);
    auto outVectBTestCompFig = matRandTest.computeMatrixVector(vectXCompFig, false);
    double cudaError = computeMeanSquaredError(outVectBTestCompMKL.getDataC(), outVectBTestVariance.getDataC(), numRows);
    double figError = computeMeanSquaredError(outVectBTestCompFig.getDataC(), outVectBTestVariance.getDataC(), numRows);

    std::cout << "CUDA MSE " << cudaError << std::endl;
    std::cout << "Figaro MSE " << figError << std::endl;
    std::cout << "MKL is " << figError / cudaError  << " more accurate" << std::endl;
 }

#endif