#include "matrix.h"
#include "matrix_mkl.h"
#include <random>

template <typename T, MajorOrder majorOrder>
static CBLAS_LAYOUT getCBlasMajorOrder(const Matrix<T, majorOrder>& mat)
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
static int getLaPackMajorOrder(const Matrix<T, majorOrder>& mat)
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
Matrix<T, majorOrder> Matrix<T, majorOrder>::selfMatrixTransposeMultiplication(void) const
{
    double alpha = 1.0, beta = 0.0;
    const auto& matA = *this;

    Matrix<T, majorOrder> matOut{getNumCols(), getNumCols()};
    cblas_dgemm(getCBlasMajorOrder(matA), CblasNoTrans, CblasTrans,
                getNumCols(), getNumCols(), getNumRows(), alpha, getDataC(),
                getLeadingDimension(), getDataC(), getLeadingDimension(), beta,
                matOut.getData(), getNumCols());

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
Matrix<T, majorOrder>
Matrix<T, majorOrder>::generateRandom(int numRows, int numCols, int seed,
        double start, double end)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dist(start, end);
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

template class Matrix<double, MajorOrder::ROW_MAJOR>;
template class Matrix<double, MajorOrder::COL_MAJOR>;

// template <>
// Matrix<double, MajorOrder::ROW_MAJOR>
// Matrix<double, MajorOrder::ROW_MAJOR>::generateRandom(int, int, int, double, double);

// template <>
// Matrix<double, MajorOrder::COL_MAJOR>
// Matrix<double, MajorOrder::COL_MAJOR>::generateRandom(int, int, int, double, double);

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
T Matrix<T, majorOrder>::orthogonality(void)
{
    auto eye = identity(getNumCols());
    T frobNorm = 0.0;

    return frobNorm;
}

