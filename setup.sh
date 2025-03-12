nvcc -O3 main.cu -Wno-deprecated-gpu-targets -lcusolver -lcudart -lcublas -lboost_program_options -o main

for i in {1..5} ; do
    ./main --m1 1000 --n1 4 --m2 1000 --n2 4
done
