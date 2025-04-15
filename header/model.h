#ifndef _LINSCALE_MODEL_H_
#define _LINSCALE_MODEL_H_

#include "compute.h"

void evaluateTrain(const MatrixDRow& mat1, const MatrixDRow& mat2,
    const MatrixDCol& matJoin, MatrixDCol& matCUDAR, MatrixDCol& matFigR,
    MatrixDCol& matFigQ, MatrixDCol& matCUDAQ,
    ComputeDecomp decompType, bool checkAccuracy)
{
    /*********** TRAINING ***********************/
    computeGeneral<double, MajorOrder::COL_MAJOR>(matJoin, matCUDAR, matCUDAQ, decompType);

    computeFigaro<double>(mat1, mat2, matFigR, matFigQ, decompType);
    if (decompType == ComputeDecomp::Q_AND_R and checkAccuracy)
    {
        double cudaQOrt = matCUDAQ.computeOrthogonality();
        std::cout << "ORTHOGONALITY Cuda " << matCUDAQ.computeOrthogonality() << std::endl;
        double figQOrt = matFigQ.computeOrthogonality();
        std::cout << "ORTHOGONALITY LinScale " << matFigQ.computeOrthogonality() << std::endl;
        std::cout << "LinScale is"  << (double)cudaQOrt / figQOrt << " times more orthogonal" << std::endl;
    }
}

void computeVectors(const MatrixDCol& matJoin, const MatrixDCol& matCUDAR,
    const MatrixDCol& matFigR, const MatrixDCol& vectBTrain,
    MatrixDCol& vectXCompMKL, MatrixDCol& vectXCompFig)
{
    // vectXCompMKL = matJoin.solveLLSNormalEquationUsingR(matCUDAR, vectBTrain);
    solveLLSNormalEquationUsingR(matJoin, matCUDAR, vectBTrain, vectXCompMKL);
    solveLLSNormalEquationUsingR(matJoin, matFigR, vectBTrain, vectXCompFig);
    // vectXCompMKL = matJoin.solveLLSNormalEquations(vectBTrain);
    // vectXCompMKL = matJoin.solveLLSQRDecomp(vectBTrain);
    // vectXCompFig = matJoin.solveLLSNormalEquationUsingR(matFigR, vectBTrain);
}

void evaluateTest(int numRows, int numCols,
    const MatrixDCol& vectX, MatrixDCol& vectXCompMKL,
    MatrixDCol& vectXCompFig, int seed)
{
    auto matRandTest = MatrixDCol::generateRandom<RandomDistribution::UNIFORM>(numRows, numCols,
     seed + 22);

    auto outVectBTest = matRandTest.computeMatrixVector(vectX, false);
    auto matUniformAdd = MatrixDCol::generateRandom<RandomDistribution::UNIFORM>(
        matRandTest.getNumRows(), matRandTest.getNumCols(),
        seed + 37);
    auto matUniformCopy = matUniformAdd.divValue(1e10);
    // auto outVectBTestVariance = outVectBTest.add(matUniformCopy);
    auto outVectBTestVariance = std::move(outVectBTest);

    auto outVectBTestCompMKL = matRandTest.computeMatrixVector(vectXCompMKL, false);
    auto outVectBTestCompFig = matRandTest.computeMatrixVector(vectXCompFig, false);
    double cudaError = computeMeanSquaredError(outVectBTestCompMKL.getDataC(), outVectBTestVariance.getDataC(), numRows);
    double figError = computeMeanSquaredError(outVectBTestCompFig.getDataC(), outVectBTestVariance.getDataC(), numRows);

    std::cout << "CUDA MSE " << cudaError << std::endl;
    std::cout << "LinScale MSE " << figError << std::endl;
    std::cout << "LinSCale is " << cudaError / figError   << " times more accurate" << std::endl;
 }

#endif