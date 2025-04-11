#ifndef _LIN_SCALE_MATRIX_H_
#define _LIN_SCALE_MATRIX_H_

#include <iostream>
#include <fstream>
#include <cstring>

enum class MajorOrder
{
    ROW_MAJOR = 0,
    COL_MAJOR = 1
};

enum class ComputeDecomp
{
    R_ONLY = 0,
    Q_AND_R = 1,
    SIGMA_ONLY = 2,
    U_AND_S_AND_V = 3
};

#define IDX(rowIdx, colIdx, width) ((rowIdx) * (width) + (colIdx))
#define IDX_R(rowIdx, colIdx, numRows, numCols) ((rowIdx) * (numCols) + (colIdx) )
#define IDX_C(rowIdx, colIdx, numRows, numCols) ((rowIdx)  + (colIdx) * (numRows))


template<MajorOrder majorOrder>
int64_t getPosId(int64_t rowIdx, int64_t colIdx, int64_t numRows, int64_t numCols)
{
    if (MajorOrder::ROW_MAJOR == majorOrder)
    {
        return IDX_R(rowIdx, colIdx, numRows, numCols);
    }
    else
    {
        return IDX_C(rowIdx, colIdx, numRows, numCols);
    }
}


template <typename T, MajorOrder majorOrder>
void copyMatrix(const T* pArr, T*& pArrOut, int numRows, int numCols, int numRowsCopy, int numColsCopy, bool upperTriangular = false)
{
    for (int rowIdx = 0; rowIdx < numRowsCopy; rowIdx++)
    {
        for (int colIdx = 0; colIdx < numColsCopy; colIdx++)
        {
            if (upperTriangular and (rowIdx > colIdx))
            {
                pArrOut[getPosId<majorOrder>(rowIdx, colIdx, numRowsCopy,  numColsCopy)]  = 0;
            }
            else
            {
                pArrOut[getPosId<majorOrder>(rowIdx, colIdx, numRowsCopy,  numColsCopy)]  = pArr[getPosId<majorOrder>(rowIdx, colIdx, numRows,  numCols)];
            }
        }
    }
}

template <typename T, MajorOrder majorOrder>
class Matrix
{
    int numRows = 0;
    int numCols = 0;
    T* pArr = nullptr;
public:
    Matrix(int _numRows, int _numCols): numRows(_numRows), numCols(_numCols)
    {
        pArr = new T[int64_t(numRows) * int64_t(numCols)];
        // std::cout << "CREATE" << pArr << std::endl;
    }

    Matrix(const T* pArrCopy, int _numRows, int _numCols): numRows(_numRows), numCols(_numCols)
    {
        pArr = new T[int64_t(numRows) * int64_t(numCols)];
        memcpy(pArr, pArrCopy, getSize());
        // std::cout << "CREATE" << pArr << std::endl;
    }

    Matrix(const Matrix& matIn) = delete;
    Matrix& operator=(const Matrix& matIn) = delete;
    Matrix(Matrix&& matIn)
    {
        if (pArr != nullptr)
        {
            delete [] pArr;
        }
        pArr = matIn.pArr;
        // std::cout << "MOVE " << pArr << std::endl;
        numRows = matIn.numRows;
        numCols = matIn.numCols;
        matIn.pArr = nullptr;
    }

    Matrix& operator=(Matrix&& matIn)
    {
        if (pArr != nullptr)
        {
            delete [] pArr;
        }
        pArr = matIn.pArr;
        // std::cout << "ASSIGN " << pArr << std::endl;
        numRows = matIn.numRows;
        numCols = matIn.numCols;
        matIn.pArr = nullptr;
        return *this;
    }

    ~Matrix()
    {
        if (pArr != nullptr)
        {
            // std::cout << "DELETE" << pArr  << std::endl;
            delete [] pArr;
            pArr = nullptr;
        }
    }

    T& operator()(int rowIdx, int colIdx)
    {
        int64_t posId = getPosId<majorOrder>(rowIdx, colIdx, numRows, numCols);
        return pArr[posId];
    }

    const T& operator()(int rowIdx, int colIdx) const
    {
        int64_t posId = getPosId<majorOrder>(rowIdx, colIdx, numRows, numCols);
        return pArr[posId];
    }
    T*& getData()
    {
        return pArr;
    }

    const T* getDataC() const
    {
        return pArr;
    }

    int getNumRows(void) const {
        return numRows;

    }
    int getNumCols(void) const
    {
        return numCols;
    }

    int getNumElements(void) const
    {
        return numRows * numCols;
    }

    int getSize() const
    {
        return getNumElements() * sizeof(T);
    }

    int getLeadingDimension(void) const
    {
        if constexpr (majorOrder == MajorOrder::COL_MAJOR)
        {
            return getNumRows();
        }
        else
        {
            return getNumCols();
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const Matrix<T, majorOrder>& mat)
    {
        out << " Matrix: (" << mat.getNumRows() << "x" << mat.getNumCols() << ")" << std::endl;
        for (int rowIdx = 0; rowIdx < mat.getNumRows(); rowIdx++)
        {
            for (int colIdx = 0; colIdx < mat.getNumCols(); colIdx++)
            {
                out << mat(rowIdx, colIdx) << " ";
            }
            out << std::endl;
        }
        out << std::endl;

        return out;
    }

    Matrix<T, majorOrder> subMatrix(int startRowIdx, int endRowIdx,
        int startColIdx, int endColIdx) const;

    // TODO: compute condition number
    // TODO: Migrate MatrixVector
    // TODO: Migrate MatrixMatrix
    // TODO: Migrate SMTT
    // TODO: Migrate computeInverse
    // TODO: Migrate LLS classical
    // TODO: Migrate LLS normal

    T computeFrobeniusNorm(void);

    static Matrix<T, majorOrder>
    generateRandom(int numRows, int numCols, int seed,
        double start = 0.0, double end = 1.0);

    static Matrix<T, majorOrder> zero(int numRows, int numCols);

    static Matrix<T, majorOrder> identity(int numRows);

    Matrix<T, majorOrder> add(const Matrix<T, majorOrder>& mat) const;

    Matrix<T, majorOrder> subtract(const Matrix<T, majorOrder>& mat) const;

    Matrix<T, majorOrder> divide(const Matrix<T, majorOrder>& mat) const;

    Matrix<T, majorOrder> elementWiseMultiply(const Matrix<T, majorOrder>& mat) const;

    Matrix<T, majorOrder> divValue(T val) const;

    Matrix<T, majorOrder> computeMatrixVector(const Matrix<T, majorOrder>& vectV,
        bool transpose = false) const;

    Matrix<T, majorOrder> selfMatrixTransposeMultiplication(void) const;

    Matrix<T, majorOrder> computeInverse(bool isUpperTriangular) const;

    T computeFrobeniusNorm(void) const;

    T orthogonality(void);

};


using MatrixDCol = Matrix<double, MajorOrder::COL_MAJOR>;
using MatrixDRow = Matrix<double, MajorOrder::ROW_MAJOR>;
template <typename T>
using MatrixCol = Matrix<T, MajorOrder::COL_MAJOR>;
template <typename T>
using MatrixRow = Matrix<T, MajorOrder::ROW_MAJOR>;

template <typename T>
MatrixCol<T> changeLayout(const MatrixRow<T>& mat);
template <typename T, MajorOrder majorOrder>
void printMatrix(const Matrix<T, majorOrder>& matA, int numRowsCut, const std::string& fileName, bool upperTriangular = false);

/*************************** DEFINITIONS ******************/
template<typename T, MajorOrder majorOrder>
Matrix<T, majorOrder>
Matrix<T, majorOrder>::subMatrix(int startRowIdx, int endRowIdx,
        int startColIdx, int endColIdx) const
{
        // pArrOut = new T[numRowsCopy * numColsCopy];
    Matrix<T, majorOrder> matOut{endRowIdx - startRowIdx + 1,
        endColIdx - startColIdx + 1};
    const auto& matA = *this;
    for (int srcRowIdx = startRowIdx; srcRowIdx <= endRowIdx; srcRowIdx++)
    {
        for (int srcColIdx = startColIdx; srcColIdx <= endColIdx; srcColIdx++)
        {
            int outRowIdx = srcRowIdx - startRowIdx;
            int outColIdx = srcColIdx - startColIdx;
            matOut(outRowIdx, outColIdx) = matA(srcRowIdx, srcColIdx);
        }
    }

    return matOut;
}


template<typename T, MajorOrder majorOrder>
Matrix<T, majorOrder> Matrix<T, majorOrder>::divValue(T val) const
{
    const auto& mat = *this;
    auto matOut = Matrix<T, majorOrder>{mat.getNumRows(), mat.getNumCols()};
    if constexpr (majorOrder == MajorOrder::ROW_MAJOR)
    {
        for (int rowIdx = 0; rowIdx < mat.getNumRows(); rowIdx++)
        {
            for (int colIdx = 0; colIdx < mat.getNumCols(); colIdx++)
            {
                matOut(rowIdx, colIdx) = mat(rowIdx, colIdx)/ val;
            }
        }
    }
    else
    {
        for (int colIdx = 0; colIdx < mat.getNumCols(); colIdx++)
        {
            for (int rowIdx = 0; rowIdx < mat.getNumRows(); rowIdx++)
            {
                matOut(rowIdx, colIdx) = mat(rowIdx, colIdx) / val;
            }
        }
    }
    return matOut;
}

template <typename T>
MatrixCol<T> changeLayout(const MatrixRow<T>& mat)
{
    T alpha = 1.0;
    MatrixCol<T> matOut{mat.getNumRows(), mat.getNumCols()};
    // mkl_domatcopy('R', 'T', mat.getNumRows(), mat.getNumCols(),
    //     alpha, mat.getDataC(), mat.getNumCols(), matOut.getData(), matOut.getNumRows());
    for (int rowIdx = 0; rowIdx < mat.getNumRows(); rowIdx++)
    {
        for (int colIdx = 0; colIdx < mat.getNumCols(); colIdx++)
        {
            matOut(rowIdx, colIdx) = mat(rowIdx, colIdx);
        }
    }
    return matOut;
}


template <typename T, MajorOrder majorOrder>
void printMatrix(const Matrix<T, majorOrder>& matA, int numRowsCut, const std::string& fileName, bool upperTriangular)
{
    std::ofstream outFile(fileName);
    if (!outFile.is_open())
    {
        std::cerr << "WTF?  " << fileName << std::endl;
    }
    for (int rowIdx = 0; rowIdx < std::min(matA.getNumRows(), numRowsCut); rowIdx++)
    {
        for (int colIdx = 0; colIdx < matA.getNumCols(); colIdx++)
        {
            if (colIdx > 0)
            {
                outFile << ",";
            }
            if (upperTriangular and (rowIdx > colIdx))
            {
                outFile << "0";
            }
            else
            {
                outFile << matA(rowIdx, colIdx);
            }

        }
        outFile << std::endl;
    }
}

#endif