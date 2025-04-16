#include "matrix.h"
#include <random>
#include <mkl.h>
#include <mkl_lapacke.h>
#include <mkl_cblas.h>
#include <mkl_vml.h>

template <typename T, MajorOrder majorOrder>
constexpr static CBLAS_LAYOUT getCBlasMajorOrder(const Matrix<T, majorOrder>& mat)
{
    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        return CblasRowMajor;
    }
    else
    {
        return CblasColMajor;
    }
}

template <typename T, MajorOrder majorOrder>
constexpr static int getLaPackMajorOrder(const Matrix<T, majorOrder>& mat)
{
    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        return LAPACK_ROW_MAJOR;
    }
    else
    {
        return LAPACK_COL_MAJOR;
    }
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::subtract(const Matrix<T, majorOrder>& mat) const
{
    auto matOut = Matrix<T, majorOrder>{mat.getNumRows(), mat.getNumCols()};

    vdSub(getNumElements(), getDataC(), mat.getDataC(), matOut.getData());

    return matOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::add(const Matrix<T, majorOrder>& mat) const
{
     auto matOut = Matrix<T, majorOrder>{mat.getNumRows(), mat.getNumCols()};

    vdAdd(getNumElements(), getDataC(), mat.getDataC(), matOut.getData());

    return matOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::divide(const Matrix<T, majorOrder>& mat) const
{
    auto matOut = Matrix<T, majorOrder>{mat.getNumRows(), mat.getNumCols()};

    vdDiv(getNumElements(), getDataC(), mat.getDataC(), matOut.getData());

    return matOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::elementWiseMultiply(const Matrix<T, majorOrder>& mat) const
{
    auto matOut = Matrix<T, majorOrder>{mat.getNumRows(), mat.getNumCols()};

    vdMul(getNumElements(), getDataC(), mat.getDataC(), matOut.getData());

    return matOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::computeMatrixVector(const Matrix<T, majorOrder>& vectV, bool transpose) const
{
    T alpha = 1.0;
    T beta = 0.0;
    auto aTran = (transpose ?  CblasTrans : CblasNoTrans);
    auto rowsOut = (transpose ? getNumCols() : getNumRows());
    Matrix<T, majorOrder> matOut{rowsOut, 1};
    const auto& matA = *this;
    cblas_dgemv(getCBlasMajorOrder(matA), aTran, getNumRows(), getNumCols(), alpha,
        getDataC(), getLeadingDimension(), vectV.getDataC(), 1, beta, matOut.getData(), 1);

    return matOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::computeMatrixMatrix(
    const Matrix<T, majorOrder>& matB) const
{
    double alpha = 1.0, beta = 0.0;
    Matrix<T, majorOrder> matOut{getNumRows(), matB.getNumCols()};
    auto& matA = *this;

    cblas_dgemm(getCBlasMajorOrder(*this), CblasNoTrans, CblasNoTrans,
            getNumRows(), matB.getNumCols(), getNumCols(), alpha, getDataC(),
                getLeadingDimension(), matB.getDataC(), matB.getLeadingDimension(),
                beta, matOut.getData(), matOut.getLeadingDimension());

    return matOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::selfMatrixTransposeMultiplication(void) const
{
    double alpha = 1.0, beta = 0.0;
    const auto& matA = *this;

    Matrix<T, majorOrder> matOut{getNumCols(), getNumCols()};
    cblas_dgemm(getCBlasMajorOrder(matA), CblasNoTrans, CblasTrans,
                getNumRows(), getNumRows(), getNumCols(), alpha, getDataC(),
                getLeadingDimension(), getDataC(), getLeadingDimension(), beta,
                matOut.getData(), getNumCols());

    return matOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::selfTransposeMatrixMultiplication(void) const
{
    double alpha = 1.0, beta = 0.0;
    const auto& matA = *this;

    Matrix<T, majorOrder> matOut{getNumCols(), getNumCols()};
    cblas_dgemm(getCBlasMajorOrder(matA), CblasTrans, CblasNoTrans,
                getNumCols(), getNumCols(), getNumRows(), alpha, getDataC(),
                getLeadingDimension(), getDataC(), getLeadingDimension(), beta,
                matOut.getData(), matOut.getLeadingDimension());

    return matOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::computeInverse(bool isUpperTriangular) const
{
    Matrix<T, majorOrder> matCopy(getDataC(), getNumRows(), getNumCols());
    if (isUpperTriangular)
    {
        LAPACKE_dtrtri(getLaPackMajorOrder(matCopy), 'U', 'N',
            matCopy.getNumRows(), matCopy.getData(), matCopy.getLeadingDimension());
    }
    else
    {
        int N = 3;
        int LDA = 3, info;
        int *ipiv = new int [getNumCols()];

        LAPACKE_dgetrf(getLaPackMajorOrder(matCopy), matCopy.getNumRows(),
            matCopy.getNumRows(), matCopy.getData(), matCopy.getLeadingDimension(), ipiv);

        LAPACKE_dgetri(getLaPackMajorOrder(matCopy), matCopy.getNumRows(), matCopy.getData(),
                matCopy.getLeadingDimension(), ipiv);

        delete [] ipiv;
    }
    return matCopy;
}


// Ax = b --- pMatA * pOutVect = pVectB,
// = A^T * A * x = A^T * b
// x = (A^T * A)^ inv * A^T * b
// A^T * A = R^T * R
template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::solveLLSNormalEquations(
    const Matrix<T, majorOrder>& vectB) const
{
    const auto& matA = *this;
    auto matSMTT = matA.selfTransposeMatrixMultiplication();
    auto matSMTTInv = matSMTT.computeInverse(false);

    auto tempVect = this->computeMatrixVector(vectB, true);
    auto vectXOut = matSMTTInv.computeMatrixVector(tempVect, false);

    return vectXOut;
}

// Ax = b --- pMatA * pOutVect = pVectB,
// = A^T * A * x = A^T * b
// x = (A^T * A)^ inv * A^T * b
// (A^T * A)^ inv = R^inv * R^T^inv
template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::solveLLSNormalEquationUsingR(
    const Matrix<T, majorOrder>& matR, const Matrix<T, majorOrder>& vectB) const
{
    auto matRInv = matR.computeInverse(true);
    auto outMat = matRInv.selfMatrixTransposeMultiplication();

    auto tempVect = this->computeMatrixVector(vectB, true);
    auto vectXOut = outMat.computeMatrixVector(tempVect, false);

    return vectXOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::solveLLSQRDecomp(
    const Matrix<T, majorOrder>& vectB) const
{
    Matrix<T, majorOrder> matA(getDataC(), getNumRows(), getNumCols());
    Matrix<T, majorOrder> vectX{vectB.getDataC(), vectB.getNumRows(), vectB.getNumCols()};

    LAPACKE_dgels(getLaPackMajorOrder(matA), 'N', matA.getNumRows(), matA.getNumCols(), 1,
        matA.getData(), matA.getLeadingDimension(), vectX.getData(), vectB.getLeadingDimension());

    return vectX;
}



template <typename T, MajorOrder majorOrder>
T Matrix<T, majorOrder>::computeFrobeniusNorm(void) const
{
    const auto& matA = *this;
    auto laPackMajorOrder = getLaPackMajorOrder(matA);
    T frobNorm;
    if constexpr (std::is_same<T, double>::value)
    {
        frobNorm = LAPACKE_dlange(laPackMajorOrder, 'F', getNumRows(), getNumCols(),
        getDataC(), getLeadingDimension());
    }
    else
    {
        frobNorm = LAPACKE_slange(laPackMajorOrder, 'F', getNumRows(), getNumCols(),
        getDataC(), getLeadingDimension());
    }
    return frobNorm;
}

template <typename T, MajorOrder majorOrder>
template <RandomDistribution distrType>
auto Matrix<T, majorOrder>::drawDistribution(
        T first, T second)
{
    if constexpr (distrType == RandomDistribution::UNIFORM)
    {
        return std::uniform_real_distribution<T>(first, second);
    }
    else if constexpr (distrType == RandomDistribution::NORMAL)
    {
        return std::normal_distribution<T>(first, second);
    }
}

template <typename T, MajorOrder majorOrder>
template <RandomDistribution distrType>
Matrix<T, majorOrder>
Matrix<T, majorOrder>::generateRandom(int numRows, int numCols,
        int seed, T start, T end)
{
    std::mt19937 gen(seed);
    auto dist = drawDistribution<distrType>(start, end);
    auto matA = std::move(Matrix<T, majorOrder>{numRows, numCols});
    if constexpr (MajorOrder::COL_MAJOR == majorOrder)
    {
        for (int colIdx = 0; colIdx < numCols; colIdx++)
        {
            for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
            {
                matA(rowIdx, colIdx) = dist(gen);
            }
        }
    }
    else
    {
        for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
        {
            for (int colIdx = 0; colIdx < numCols; colIdx++)
            {
                matA(rowIdx, colIdx) = dist(gen);
            }
        }
    }
    return std::move(matA);
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::zero(int numRows, int numCols)
{
    auto matA = std::move(Matrix<T, majorOrder>{numRows, numCols});
    for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
    {
        for (int colIdx = 0; colIdx < numCols; colIdx++)
        {
            matA(rowIdx, colIdx) = 0;
        }
    }
    return std::move(matA);
}


template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::identity(int numRows)
{
    auto matA = zero(numRows, numRows);
    for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
    {
        matA(rowIdx, rowIdx) = 1;
    }
    return std::move(matA);
}

template <typename T, MajorOrder majorOrder>
T Matrix<T, majorOrder>::computeOrthogonality(void) const
{
    auto eye = identity(getNumCols());
    T frobNorm = 0.0;
    auto eyeComp = selfTransposeMatrixMultiplication();
    auto diff = eye.subtract(eyeComp);
    auto relError = diff.computeFrobeniusNorm() / eye.computeFrobeniusNorm();

    return relError;
}

template <typename T, MajorOrder majorOrder>
double Matrix<T, majorOrder>::computeMeanSquaredError(const Matrix<T, majorOrder>& mat2) const
{
    auto& mat1 = *this;
    std::vector<double> vDiff(mat1.getNumElements());
    std::vector<double> vSquared(vDiff.size());

      // Compute element-wise difference: diff = a - b
    vdSub(mat1.getNumElements(), getDataC(), mat2.getDataC(), vDiff.data());

    // Square each element: squared = diff^2
    vdMul(vDiff.size(), vDiff.data(), vDiff.data(), vSquared.data());

    // Compute sum of squared differences
    double sum_sq = cblas_dasum(vSquared.size(), vSquared.data(), 1);

    // Compute MSE
    double mse = sum_sq / numRows;

    return mse;
}

template class Matrix<double, MajorOrder::ROW_MAJOR>;
template class Matrix<double, MajorOrder::COL_MAJOR>;

template auto MatrixDCol::drawDistribution<RandomDistribution::UNIFORM>(double, double);
template auto MatrixDCol::drawDistribution<RandomDistribution::NORMAL>(double, double);

template MatrixDCol MatrixDCol::generateRandom<RandomDistribution::UNIFORM>(int, int, int, double, double);
template MatrixDCol MatrixDCol::generateRandom<RandomDistribution::NORMAL>(int, int, int, double, double);

