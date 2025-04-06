#ifndef _LINSCALE_MODEL_H_
#define _LINSCALE_MODEL_H_

#include "compute.h"

void evaluateTrain(const MatrixDRow& mat1, const MatrixDRow& mat2,
    const MatrixDCol& matCartProdTrain, MatrixDCol& matCUDAR, MatrixDCol& matFigR,
    const std::string& fileName, ComputeDecomp decompType)
{
    /*********** TRAINING ***********************/
    computeGeneral<double, MajorOrder::COL_MAJOR>(matCartProdTrain, matCUDAR,
         fileName, decompType);
    // printMatrix<double, MajorOrder::COL_MAJOR>(matCUDAR.getData(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), fileName + "CUDA.csv", false);

    computeFigaro<double>(mat1, mat2, matFigR, fileName, decompType);
    // printMatrix<double, MajorOrder::COL_MAJOR>(matFigR.getData(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), fileName + "LinScale.csv", false);
}

void computeVectors(const MatrixDCol& matCartProd, const MatrixDCol& matCUDAR,
    const MatrixDCol& matFigR, const MatrixDCol& vectBTrain,
    MatrixDCol& vectXCompMKL, MatrixDCol& vectXCompFig, int seed)
{
    vectXCompMKL = solveLLSMKL(matCartProd, matCUDAR, vectBTrain, matCartProd.getNumRows(), matCartProd.getNumCols(),  "results/Cuda"+ std::to_string(seed));
    vectXCompFig = solveLLSNormalEquation(matCartProd, matFigR, vectBTrain, matCartProd.getNumRows(), matCartProd.getNumCols(), "results/LinScale" + std::to_string(seed) );
}

void evaluateTest(int numRows1, int numCols1, int numRows2, int numCols2,
    const MatrixDRow& vectX, MatrixDCol& vectXCompMKL,
    MatrixDCol& vectXCompFig, int seed)
{
    auto mat1Test = generateRandom<double>(numRows1, numCols1, 22);
    auto mat2Test = generateRandom<double>(numRows2, numCols2, 31);

    MatrixDCol matCartProdTest{1, 1};
    generateCartesianProduct<double, MajorOrder::ROW_MAJOR, MajorOrder::COL_MAJOR>(
            mat1Test, mat2Test,  matCartProdTest);
    auto outVectBTest = computeMatrixVector<double, MajorOrder::COL_MAJOR>(matCartProdTest, vectX, numRows1 * numRows2, numCols1 + numCols2, false);
    auto matUniformAdd = generateRandom<double>(matCartProdTest.getNumRows(),
        matCartProdTest.getNumCols(), 37);
    decltype(matUniformAdd) matUniformCopy {matUniformAdd.getNumRows(), matUniformAdd.getNumCols()};
    divValue(matUniformAdd, 1e64, matUniformCopy);
    MatrixDCol outVectBTestVariance{1, 1};
    addVectors(outVectBTest, matUniformCopy, outVectBTestVariance);

    auto outVectBTestCompMKL = computeMatrixVector(matCartProdTest, vectXCompMKL, numRows1 * numRows2, numCols1 + numCols2, false);
    auto outVectBTestCompFig = computeMatrixVector(matCartProdTest, vectXCompFig, numRows1 * numRows2, numCols1 + numCols2, false);
    double cudaError = computeMeanSquaredError(outVectBTestCompMKL.getDataC(), outVectBTestVariance.getDataC(), numRows1 * numRows2);
    double figError = computeMeanSquaredError(outVectBTestCompFig.getDataC(), outVectBTestVariance.getDataC(), numRows1 * numRows2);

    std::cout << "CUDA MSE " << cudaError << std::endl;
    std::cout << "Figaro MSE " << figError << std::endl;
}

#endif