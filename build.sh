source /opt/intel/oneapi/2025.1/oneapi-vars.sh
cd build
#rm *. -rf
cmake ..

make -j
# g++ -O3 src/main.cpp -o main -std=c++23 -m64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_rt -lpthread -fopenmp -lboost_program_options -lm -ldl

./LinScale-Cuda --m1 3 --n1 2 --m2 3 --n2 2 --compute 1
exit
