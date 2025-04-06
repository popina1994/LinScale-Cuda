
#include <iomanip>
#include <boost/program_options.hpp>
#include "compute.h"

namespace po = boost::program_options;

void evaluateTrain(const MatrixDRow& mat1, const MatrixDRow& mat2,
    const MatrixDCol& matCartProdTrain, MatrixDCol& matCUDAR, MatrixDCol& matFigR,
    const std::string& fileName, int compute)
{
    matCUDAR = MatrixDCol{mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols()};
    /*********** TRAINING ***********************/
    computeGeneral<double, MajorOrder::COL_MAJOR>(matCartProdTrain.getDataC(), matCUDAR.getData(),
        mat1.getNumRows() * mat2.getNumRows(), mat1.getNumCols() + mat2.getNumCols(), fileName, compute);
    // printMatrix<double, MajorOrder::COL_MAJOR>(matCUDAR.getData(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), fileName + "CUDA.csv", false);

    matFigR = MatrixDCol{mat1.getNumRows() + mat2.getNumRows() - 1, mat1.getNumCols() + mat2.getNumCols()};
    computeFigaro<double>(mat1.getDataC(), mat2.getDataC(), matFigR.getData(), mat1.getNumRows(), mat1.getNumCols(), mat2.getNumRows(), mat2.getNumCols(),
    fileName, compute);
    // printMatrix<double, MajorOrder::COL_MAJOR>(matFigR.getData(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), fileName + "LinScale.csv", false);
}

void computeVectors(const MatrixDCol& matCartProd, const MatrixDCol& matCUDAR,
    const MatrixDCol& matFigR, const MatrixDCol& vectBTrain,
    MatrixDCol& vectXCompMKL, MatrixDCol& vectXCompFig, int seed)
{
    vectXCompMKL = solveLLS(matCartProd, matCUDAR, vectBTrain, matCartProd.getNumRows(), matCartProd.getNumCols(), std::to_string(seed) + "results/Cuda");
    vectXCompFig = solveLLS(matCartProd, matFigR, vectBTrain, matCartProd.getNumRows(), matCartProd.getNumCols(), std::to_string(seed) + "results/LinScale");
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
    divValue(matUniformAdd, 100.0, matUniformCopy);

    auto outVectBTestCompMKL = computeMatrixVector(matCartProdTest, vectXCompMKL, numRows1 * numRows2, numCols1 + numCols2, false);
    auto outVectBTestCompFig = computeMatrixVector(matCartProdTest, vectXCompFig, numRows1 * numRows2, numCols1 + numCols2, false);
    double cudaError = computeMeanSquaredError(outVectBTestCompMKL.getDataC(), outVectBTest.getDataC(), numRows1 * numRows2);
    double figError = computeMeanSquaredError(outVectBTestCompFig.getDataC(), outVectBTest.getDataC(), numRows1 * numRows2);

    std::cout << "CUDA MSE " << cudaError << std::endl;
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

    auto vectX = generateRandom<double>(1, numCols1 + numCols2, 15);
    auto outVectBTrain = computeMatrixVector(matCartProd, vectX,
        matCartProd.getNumRows(), matCartProd.getNumCols(), false);

    MatrixDCol matCUDAR{1, 1};
    MatrixDCol matFigR{1, 1};

    evaluateTrain(mat1, mat2, matCartProd, matCUDAR, matFigR, fileName, compute);
    MatrixDCol matVectXMKL{1, 1};
    MatrixDCol matVectXFig{1, 1};
    computeVectors(matCartProd, matCUDAR, matFigR, outVectBTrain, matVectXMKL, matVectXFig, -1);
    evaluateTest(numRows1, numCols1, numRows2, numCols2, vectX, matVectXMKL, matVectXFig, -1);

    // printMatrix<double, MajorOrder::ROW_MAJOR>(mat1.getData(), numRows1, numCols1, numRows1, "AS.csv", false);
    // printMatrix<double, MajorOrder::ROW_MAJOR>(mat2.getData(), numRows2, numCols2, numRows2, "BS.csv", false);
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
