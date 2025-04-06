#ifndef _TYPES_H_
#define _TYPES_H_

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


#endif
