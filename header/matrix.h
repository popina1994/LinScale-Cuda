#ifndef _LIN_SCALE_MATRIX_H_
#define _LIN_SCALE_MATRIX_H_

#include <iostream>

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

template <typename T, MajorOrder order>
void printMatrix(T* pArr, int numRows, int numCols, int numRowsCut, const std::string& fileName, bool upperTriangular = false)
{
    std::ofstream outFile(fileName);
    if (!outFile.is_open())
    {
        std::cerr << "WTF?" << fileName << std::endl;
    }
    outFile << std::fixed << std::setprecision(15);
    for (int rowIdx = 0; rowIdx < std::min(numRows, numRowsCut); rowIdx++)
    {
        for (int colIdx = 0; colIdx < numCols; colIdx++)
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
                if constexpr (order == MajorOrder::ROW_MAJOR)
                {
                    outFile << pArr[IDX_R(rowIdx, colIdx, numRows, numCols)];
                }
                else
                {
                    outFile << pArr[IDX_C(rowIdx, colIdx, numRows, numCols)];
                }
            }

        }
        outFile << std::endl;
    }
}

template <typename T, MajorOrder order>
void copyMatrix(const T* pArr, T*& pArrOut, int numRows, int numCols, int numRowsCopy, int numColsCopy, bool upperTriangular = false)
{
    // pArrOut = new T[numRowsCopy * numColsCopy];
    for (int rowIdx = 0; rowIdx < numRowsCopy; rowIdx++)
    {
        for (int colIdx = 0; colIdx < numColsCopy; colIdx++)
        {
            if (upperTriangular and (rowIdx > colIdx))
            {
                pArrOut[getPosId<order>(rowIdx, colIdx, numRowsCopy,  numColsCopy)]  = 0;
            }
            else
            {
                pArrOut[getPosId<order>(rowIdx, colIdx, numRowsCopy,  numColsCopy)]  = pArr[getPosId<order>(rowIdx, colIdx, numRows,  numCols)];
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

    Matrix(T* pArrCopy, int _numRows, int _numCols): numRows(_numRows), numCols(_numCols)
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

};

using MatrixDCol = Matrix<double, MajorOrder::COL_MAJOR>;
using MatrixDRow = Matrix<double, MajorOrder::ROW_MAJOR>;
template <typename T>
using MatrixCol = Matrix<T, MajorOrder::COL_MAJOR>;
template <typename T>
using MatrixRow = Matrix<T, MajorOrder::ROW_MAJOR>;

template <typename T, MajorOrder order>
void printMatrix(const T* pArr, int numRows, int numCols, int numRowsCut, const std::string& fileName, bool upperTriangular = false)
{
    std::ofstream outFile(fileName);
    if (!outFile.is_open())
    {
        std::cerr << "WTF?" << fileName << std::endl;
    }
    for (int rowIdx = 0; rowIdx < std::min(numRows, numRowsCut); rowIdx++)
    {
        for (int colIdx = 0; colIdx < numCols; colIdx++)
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
                if constexpr (order == MajorOrder::ROW_MAJOR)
                {
                    outFile << pArr[IDX_R(rowIdx, colIdx, numRows, numCols)];
                }
                else
                {
                    outFile << pArr[IDX_C(rowIdx, colIdx, numRows, numCols)];
                }
            }

        }
        outFile << std::endl;
    }
}

template <typename T, MajorOrder order>
void printMatrix(const Matrix<T, order>& matA, int numRowsCut, const std::string& fileName, bool upperTriangular = false)
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