nvcc -O3 main.cu -Wno-deprecated-gpu-targets -lcusolver -lcudart -lcublas -lboost_program_options -o main
m1_a=(100 200 400 800)
m2_a=(100 200 400 800)
n_a=(2 4 8 16 32 64)

for m1 in "${m1_a[@]}"; do
    for m2 in "${m2_a[@]}"; do
        for n1 in "${n_a[@]}"; do
            echo "**************************************************"
            echo "First table: $m1 x $n1 Second table: $m2 x $n1"
            for i in {1..5} ; do
                echo "Run $i"
                ./main --m1 $(($m1)) --n1 $n1 --m2 $m2 --n2 $n1
            done
            echo "**************************************************"
        done
    done
done
