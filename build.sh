source /opt/intel/oneapi/2025.1/oneapi-vars.sh
cd build
#rm *. -rf
cmake -DMEMORY_USAGE=1 ..

make -j
# m1_a=(100 200 400 800 1600 3200)
# n_a=(4 8 16 32 64)
m1_a=(100 200 400 800 1600 3200 6400 12800)
n_a=(4 8)
# ./LinScale-Cuda --m1 1000 --n1 64 --m2 1000 --n2 64 --compute 0 --join_vals_domain_size 1
# exit
for m1 in "${m1_a[@]}"; do
    m2=$m1
        for n1 in "${n_a[@]}"; do
            echo "************************************************"
            echo "First table: $m1 x $n1 Second table: $m2 x $n1"
            for i in {1..1} ; do
                echo "Run $i"
                ./LinScale-Cuda --m1 $(($m1)) --n1 $n1 --m2 $m2 --n2 $n1 --compute 0 --join_vals_domain_size 1
            done
            echo "************************************************"
        done
done
