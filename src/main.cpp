
#include <iomanip>
#include <boost/program_options.hpp>
#include "compute.h"

namespace po = boost::program_options;

void evaluateTrain(const MatrixDRow& mat1, const MatrixDRow& mat2,
    double* h_pCartProdTrain, MatrixDCol& matMKLR, MatrixDCol& matFigR,
    const std::string& fileName, int compute)
{
    matMKLR = MatrixDCol{mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols()};
    /*********** TRAINING ***********************/
    computeGeneral<double, MajorOrder::COL_MAJOR>(h_pCartProdTrain, matMKLR.getData(),
        mat1.getNumRows() * mat2.getNumRows(), mat1.getNumCols() + mat2.getNumCols(), fileName, compute);
    printMatrix<double, MajorOrder::COL_MAJOR>(matMKLR.getData(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), fileName + "CUDA.csv", false);

    // Column orientation because of the current implementation of Figaro for faster processing
    matFigR = MatrixDCol{mat1.getNumRows() + mat2.getNumRows() - 1, mat1.getNumCols() + mat2.getNumCols()};
    computeFigaro<double>(mat1.getDataC(), mat2.getDataC(), matFigR.getData(), mat1.getNumRows(), mat1.getNumCols(), mat2.getNumRows(), mat2.getNumCols(),
    fileName, compute);
    printMatrix<double, MajorOrder::COL_MAJOR>(matFigR.getData(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), fileName + "LinScale.csv", false);
}


void computeVectors(const MatrixDCol& matCartProd, const MatrixDCol& matMKLR,
    const MatrixDCol& matFigR, const double* pVectBTrain,
double*& h_vectXCompMKL, double*& h_vectXCompFig, int seed)
{
  //   printMatrix<double, MajorOrder::COL_MAJOR>(matMKLR.getDataC(), matMKLR.getNumRows(),
    // matMKLR.getNumCols(), matMKLR.getNumRows(), std::to_string(seed) +"UPDATE2_MKL_R.csv", false);
      //   printMatrix<double, MajorOrder::ROW_MAJOR>(matFigR.getDataC(), matFigR.getNumRows(),
    // matFigR.getNumCols(), matFigR.getNumRows(), std::to_string(seed) +"UPDATE2_FIG_R.csv", false);
    solveLLS<double, MajorOrder::COL_MAJOR, MajorOrder::COL_MAJOR>(matCartProd.getDataC(), matMKLR.getDataC(), pVectBTrain, h_vectXCompMKL, matCartProd.getNumRows(), matCartProd.getNumCols(), std::to_string(seed) + "results/MKL");
    solveLLS<double, MajorOrder::COL_MAJOR, MajorOrder::COL_MAJOR>(matCartProd.getDataC(), matFigR.getDataC(), pVectBTrain, h_vectXCompFig, matCartProd.getNumRows(), matCartProd.getNumCols(), std::to_string(seed) + "results/LinScale");
}

template<typename T>
void evaluateTest(int numRows1, int numCols1, int numRows2, int numCols2, MatrixDRow& vectX,
    T* h_vectXCompMKL, T* h_vectXCompFig, int seed)
{
    T *pOutVectBTest;
    auto mat1Test = generateRandom<double>(numRows1, numCols1, 17);
    auto mat2Test = generateRandom<double>(numRows2, numCols2, 19);
    MatrixDCol matCartProdTest{1, 1};
    generateCartesianProduct<double, MajorOrder::ROW_MAJOR, MajorOrder::COL_MAJOR>(
            mat1Test, mat2Test,  matCartProdTest);
    computeMatrixVector<double, MajorOrder::COL_MAJOR>(matCartProdTest.getData(), vectX.getData(), pOutVectBTest, numRows1 * numRows2, numCols1 + numCols2, false);

    double* pOutVectBTestCompMKL, *pOutVectBTestCompFig;
    computeMatrixVector<double, MajorOrder::COL_MAJOR>(matCartProdTest.getData(), h_vectXCompMKL, pOutVectBTestCompMKL, numRows1 * numRows2, numCols1 + numCols2, false);
    computeMatrixVector<double, MajorOrder::COL_MAJOR>(matCartProdTest.getData(), h_vectXCompFig, pOutVectBTestCompFig, numRows1 * numRows2, numCols1 + numCols2, false);
    double mklError = computeMeanSquaredError(pOutVectBTestCompMKL, pOutVectBTest, numRows1 * numRows2);
    double figError = computeMeanSquaredError(pOutVectBTestCompFig, pOutVectBTest, numRows1 * numRows2);

    std::cout << "MKL MSE " << mklError << std::endl;
    std::cout << "Figaro MSE " << figError << std::endl;
}


void evaluate(int numRows1, int numCols1, int numRows2, int numCols2, std::string& fileName, int compute)
{
    // double *h_mat1, *h_mat2, *pArr;
    auto mat1 = generateRandom<double>(numRows1, numCols1, 0);
    auto mat2 = generateRandom<double>(numRows2, numCols2, 10);
    MatrixDCol matCartProd{1, 1};
    generateCartesianProduct<double, MajorOrder::ROW_MAJOR, MajorOrder::COL_MAJOR>(mat1, mat2, matCartProd);
    // printMatrix<double, MajorOrder::COL_MAJOR>(matCartProd.getDataC(), matCartProd.getNumRows(), matCartProd.getNumCols(), matCartProd.getNumRows(), fileName + "cartProd.csv", false);

    double* pOutVectBTrain;
    auto vectX = generateRandom<double>(1, numCols1 + numCols2, 15);
    computeMatrixVector<double, MajorOrder::COL_MAJOR>(matCartProd.getData(), vectX.getData(), pOutVectBTrain, matCartProd.getNumRows(), matCartProd.getNumCols(), false);

    MatrixDCol matMKLR{1, 1};
    MatrixDCol matFigR{1, 1};

    evaluateTrain(mat1, mat2, matCartProd.getData(), matMKLR, matFigR, fileName, compute);
    double* pVectXMKL;
    double* pVectXFig;
    computeVectors(matCartProd, matMKLR, matFigR, pOutVectBTrain, pVectXMKL, pVectXFig, -1);
    evaluateTest(numRows1, numCols1, numRows2, numCols2, vectX, pVectXMKL, pVectXFig, -1);

    // printMatrix<double, MajorOrder::ROW_MAJOR>(mat1.getData(), numRows1, numCols1, numRows1, "AS.csv", false);
    // printMatrix<double, MajorOrder::ROW_MAJOR>(mat2.getData(), numRows2, numCols2, numRows2, "BS.csv", false);

    // generateCartesianProduct<double, MajorOrder::ROW_MAJOR, MajorOrder::ROW_MAJOR>(
    //         mat1, mat2,  matCartProd);
    // computeGeneral<double, MajorOrder::ROW_MAJOR>(matCartProd.getData(), numRows1 * numRows2, numCols1 + numCols2, fileName, compute);
    // computeFigaro<double>(mat1.getData(), mat2.getData(), numRows1, numCols1, numRows2, numCols2, fileName, compute);
}

int main(int argc, char* argv[])
{
    int numRows1 = 1000, numCols1 = 4;
    int numRows2 = 2, numCols2 = 4;
    int compute = 1;
    try {
        // Define the command-line options
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Show help message")
            ("input,i", po::value<std::string>(), "Input file")
            ("m1", po::value<int>(), "Number of rows 1")
            ("m2", po::value<int>(), "Number of rows 2")
            ("n1", po::value<int>(), "Number of columns 1")
            ("n2", po::value<int>(), "Number of columns 2")
            ("compute", po::value<int>(), "Compute mode")
            ("verbose,v", "Enable verbose mode");

        // Parse the command-line arguments
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        // Handle the help flag
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        if (vm.count("m1"))
        {
            numRows1 = vm["m1"].as<int>();
        }
        if (vm.count("m2"))
        {
            numRows2 = vm["m2"].as<int>();
        }
        if (vm.count("n1"))
        {
            numCols1 = vm["n1"].as<int>();
        }
        if (vm.count("n2"))
        {
            numCols2 = vm["n2"].as<int>();
        }
	if (vm.count("compute"))
	{
		compute = vm["compute"].as<int>();
	}
        std::string fileName = "results/" + std::to_string(numRows1) + "x" + std::to_string(numCols1) + "," + std::to_string(numRows2) + "x" + std::to_string(numCols2);
        evaluate(numRows1, numCols1, numRows2, numCols2, fileName, compute);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    std::cout << "SUCCESSFULL" << std::endl;

    return 0;
}
