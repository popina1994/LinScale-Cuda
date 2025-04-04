
#include <iomanip>
#include <boost/program_options.hpp>
#include "compute.h"

namespace po = boost::program_options;

void evaluateTrain(const MatrixDRow& mat1, const MatrixDRow& mat2,
    double* h_pCartProdTrain, MatrixDRow& matMKLR, MatrixDRow& matFigR,
    const std::string& fileName, int compute)
{
    matMKLR = MatrixDRow{mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols()};
    /*********** TRAINING ***********************/
    computeGeneral<double, MajorOrder::ROW_MAJOR>(h_pCartProdTrain, matMKLR.getData(),
        mat1.getNumRows() * mat2.getNumRows(), mat1.getNumCols() + mat2.getNumCols(), fileName, compute);
    printMatrix<double, MajorOrder::ROW_MAJOR>(matMKLR.getData(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), "MKLR.csv", false);

    // Column orientation because of the current implementation of Figaro for faster processing
    matFigR = MatrixDRow{mat1.getNumRows() + mat2.getNumRows() - 1, mat1.getNumCols() + mat2.getNumCols()};
    computeFigaro<double>(mat1.getDataC(), mat2.getDataC(), matFigR.getData(), mat1.getNumRows(), mat1.getNumCols(), mat2.getNumRows(), mat2.getNumCols(),
    fileName, compute);
    printMatrix<double, MajorOrder::ROW_MAJOR>(matFigR.getData(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), mat1.getNumCols() + mat2.getNumCols(), "FigR.csv", false);
}

void evaluate(int numRows1, int numCols1, int numRows2, int numCols2, std::string& fileName, int compute)
{
    // double *h_mat1, *h_mat2, *pArr;
    auto mat1 = generateRandom<double>(numRows1, numCols1, 0);
    auto mat2 = generateRandom<double>(numRows2, numCols2, 10);
    MatrixDRow matCartProd{1, 1};
    generateCartesianProduct<double, MajorOrder::ROW_MAJOR, MajorOrder::ROW_MAJOR>(mat1, mat2, matCartProd);

    double* pOutVectBTrain;
    auto vectX = generateRandom<double>(1, numCols1 + numCols2, 15);
    computeMatrixVector<double, MajorOrder::ROW_MAJOR>(matCartProd.getData(), vectX.getData(), pOutVectBTrain, matCartProd.getNumRows(), matCartProd.getNumCols(), false);

    MatrixDRow matMKLR{1, 1};
    MatrixDRow matFigR{1, 1};
    evaluateTrain(mat1, mat2, matCartProd.getData(), matMKLR, matFigR, fileName, compute);
    // computeVectors(matCartProd, matMKLR, matFigR, pOutVectBTrain, pVectXMKL, pVectXFig, -1);
    // evaluateTest(numRows1, numCols1, numRows2, numCols2, vectX, pVectXMKL, pVectXFig, -1);

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
