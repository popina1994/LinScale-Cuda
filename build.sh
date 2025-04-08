source /opt/intel/oneapi/2025.1/oneapi-vars.sh
cd build
#rm *. -rf
cmake ..

make -j
m1_a=(100 200 400 800 1600)
n_a=(4 8 16 32 64)
# ./LinScale-Cuda --m1 1000 --n1 64 --m2 1000 --n2 64 --compute 0
# exit
for m1 in "${m1_a[@]}"; do
    m2=$m1
        for n1 in "${n_a[@]}"; do
            echo "***"
            echo "First table: $m1 x $n1 Second table: $m2 x $n1"
            for i in {1..5} ; do
                echo "Run $i"
                ./LinScale-Cuda --m1 $(($m1)) --n1 $n1 --m2 $m2 --n2 $n1 --compute 1
            done
            echo "***"
        done
done
