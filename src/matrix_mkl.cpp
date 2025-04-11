#include "matrix.h"
#include "matrix_mkl.h"
#include <random>

template <typename T, MajorOrder majorOrder>
int Matrix<T, majorOrder>::getCBlasMajorOrder(void) const
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
Matrix<T, majorOrder> Matrix<T, majorOrder>::subtract(const Matrix<T, majorOrder>& mat)
{
    auto matOut = Matrix<T, majorOrder>{mat.getNumRows(), mat.getNumCols()};

    vdSub(getNumElements(), getDataC(), mat.getDataC(), matOut.getData());

    return matOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::add(const Matrix<T, majorOrder>& mat)
{
     auto matOut = Matrix<T, majorOrder>{mat.getNumRows(), mat.getNumCols()};

    vdAdd(getNumElements(), getDataC(), mat.getDataC(), matOut.getData());

    return matOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::divide(const Matrix<T, majorOrder>& mat)
{
    auto matOut = Matrix<T, majorOrder>{mat.getNumRows(), mat.getNumCols()};

    vdDiv(getNumElements(), getDataC(), mat.getDataC(), matOut.getData());

    return matOut;
}

template <typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::elementWiseMultiply(const Matrix<T, majorOrder>& mat)
{
    auto matOut = Matrix<T, majorOrder>{mat.getNumRows(), mat.getNumCols()};

    vdMul(getNumElements(), getDataC(), mat.getDataC(), matOut.getData());

    return matOut;
}


template <typename T, MajorOrder majorOrder>
T Matrix<T, majorOrder>::computeFrobeniusNorm(void)
{
    auto cBlasMajorOrder = getCBlasMajorOrder();
    T frobNorm;
    if constexpr (std::is_same<T, double>::value)
    {
        frobNorm = LAPACKE_dlange(cBlasMajorOrder, 'F', getNumRows(), getNumCols(),
        getDataC(), getLeadingDimension());
    }
    else
    {
        frobNorm = LAPACKE_slange(cBlasMajorOrder, 'F', getNumRows(), getNumCols(),
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
        for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
        {
            for (int colIdx = 0; colIdx < numCols; colIdx++)
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

