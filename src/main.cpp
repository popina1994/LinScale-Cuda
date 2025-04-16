
#include <iomanip>
#include <boost/program_options.hpp>
#include "model.h"
#include "table.h"
#include <iostream>

namespace po = boost::program_options;

void evaluate(int numRows1, int numCols1, int numRows2, int numCols2, int joinValsDomainSize,
    ComputeDecomp decompType, bool checkAccuracy)
{
    auto mat1 = generateRandomJoinTable<double, MajorOrder::ROW_MAJOR>(numRows1, numCols1, 1, joinValsDomainSize, 0);
    auto mat2 = generateRandomJoinTable<double, MajorOrder::ROW_MAJOR>(numRows2, numCols2, 1, joinValsDomainSize, 10);
    mat1 = sortTable(mat1, 1);
    mat2 = sortTable(mat2, 1);
    auto matJoin = computeJoin(mat1, mat2, 1);
    auto matJoinCol = changeLayout(matJoin);

    auto vectX = MatrixDCol::generateRandom<RandomDistribution::UNIFORM>(matJoin.getNumCols(), 1, 15);
    auto outVectBTrain = matJoinCol.computeMatrixVector(vectX, false);
    auto vectNoise = MatrixDCol::generateRandom<RandomDistribution::NORMAL>(outVectBTrain.getNumRows(), 1, 22, 0.0, 1e-10);
    auto addVect = outVectBTrain.add(vectNoise);
    outVectBTrain = std::move(addVect);

    MatrixDCol matFigR{1, 1}, matFigQ{1, 1};
    MatrixDCol matCUDAR{1, 1}, matCUDAQ{1, 1};
    MatrixDCol matUCuda{1, 1}, matSigmaCuda{1, 1}, matVCuda{1, 1};
    MatrixDCol matUFig{1, 1}, matSigmaFig{1, 1}, matVFig{1, 1};

    evaluateTrain(mat1, mat2, matJoinCol, matCUDAR, matCUDAQ, matUCuda,
        matSigmaCuda, matVCuda, false, decompType, checkAccuracy);
    evaluateTrain(mat1, mat2, matJoinCol, matFigR, matFigQ, matUFig,
        matSigmaFig, matVFig, true, decompType, checkAccuracy);

    MatrixDCol matVectXMKL{1, 1}, matVectXFig{1, 1};

    if (checkAccuracy)
    {
        if (decompType == ComputeDecomp::R_ONLY or decompType == ComputeDecomp::Q_AND_R)
        {
            computeXVectorsLeastSquares(matJoinCol, matCUDAR, matFigR, outVectBTrain, matVectXMKL, matVectXFig);
            evaluateTestLeastSquares(matJoinCol.getNumRows(), matJoinCol.getNumCols(), vectX, matVectXMKL, matVectXFig, -1);
        }
        else
        {
            if (checkAccuracy)
            {
                std::cout << "MSE Sigma difference" << matSigmaCuda.computeMeanSquaredError(matSigmaFig) << std::endl;
                std::cout << matSigmaCuda << std::endl;
                std::cout << matSigmaFig << std::endl;
                // compare singular values

            }
        }
    }
}

int main(int argc, char* argv[])
{
    int numRows1 = 1000, numCols1 = 4;
    int numRows2 = 2, numCols2 = 4;
    int compute = 1;
    int joinValsDomainSize = 1;
    bool checkAccuracy = false;

    try
    {
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
            ("join_vals_domain_size", po::value<int>(), "Compute mode")
            ("check_accuracy", "Check accuracy")
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
        if (vm.count("join_vals_domain_size"))
        {
            joinValsDomainSize = vm["join_vals_domain_size"].as<int>();
        }
        if (vm.count("check_accuracy")) {
            checkAccuracy = true;
        }
        std::string fileName = "results/" + std::to_string(numRows1) + "x" + std::to_string(numCols1) + "," + std::to_string(numRows2) + "x" + std::to_string(numCols2);
        ComputeDecomp decompType;
        switch (compute) {
        case 0:
            decompType = ComputeDecomp::R_ONLY;
            break;
        case 1:
            decompType = ComputeDecomp::Q_AND_R;
            break;
        case 2:
            decompType = ComputeDecomp::SIGMA_ONLY;
            break;
        case 3:
            decompType = ComputeDecomp::U_AND_S_AND_V;
            break;
        }
        evaluate(numRows1, numCols1, numRows2, numCols2, joinValsDomainSize, decompType, checkAccuracy);
    } catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    for (const auto&[key, val]: mapMemoryTrack)
    {
        std::cout << key << " Maximally used Cuda memory "  << val << " MB" << std::endl;
    }

    return 0;
}

