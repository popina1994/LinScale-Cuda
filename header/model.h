#ifndef _LINSCALE_MODEL_H_
#define _LINSCALE_MODEL_H_

#include "compute.h"

void evaluateTrain(const MatrixDRow& mat1, const MatrixDRow& mat2,
    const MatrixDCol& matJoin, MatrixDCol& matR, MatrixDCol& matQ,
    MatrixDCol& matU, MatrixDCol& matSigma, MatrixDCol& matV,
    bool isFigaro, ComputeDecomp decompType, bool checkAccuracy)
{
    /*********** TRAINING ***********************/
    if (not isFigaro)
    {
        computeGeneral<double, MajorOrder::COL_MAJOR>(matJoin, matR, matQ, matU,
         matSigma, matV, decompType);
    }
    else
    {
        computeFigaro<double>(mat1, mat2, matR, matQ, matU,
         matSigma, matV, decompType);
    }

    if (decompType == ComputeDecomp::Q_AND_R and checkAccuracy)
    {
        double cudaQOrt = matQ.computeOrthogonality();
        std::string typeStr = isFigaro ? "LinScale" : "Cuda";
        std::cout << "ORTHOGONALITY " + typeStr << matQ.computeOrthogonality() << std::endl;
    }
}

void computeXVectorsLeastSquares(const MatrixDCol& matJoin, const MatrixDCol& matCUDAR,
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

void evaluateTestLeastSquares(int numRows, int numCols,
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
    double cudaError = outVectBTestCompMKL.computeMeanSquaredError(outVectBTestVariance);
    double figError = outVectBTestCompFig.computeMeanSquaredError(outVectBTestVariance);

    std::cout << "CUDA MSE " << cudaError << std::endl;
    std::cout << "LinScale MSE " << figError << std::endl;
    std::cout << "LinSCale is " << cudaError / figError   << " times more accurate" << std::endl;
 }

#endif