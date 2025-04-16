source /opt/intel/oneapi/2025.1/oneapi-vars.sh
cd build
#rm *. -rf
cmake -DMEMORY_USAGE=1 ..

make -j
m1_a=(100 200 400 800 1600 3200)
n_a=(4 8 16)
> log_run_times.txt
for m1 in "${m1_a[@]}"; do
    m2=$m1
        for n1 in "${n_a[@]}"; do
            echo "************************************************"
            echo "First table: $m1 x $n1 Second table: $m2 x $n1"
            for i in {1..1} ; do
                echo "Run $i"
                ./LinScale-Cuda --m1 $(($m1)) --n1 $n1 --m2 $m2 --n2 $n1 --compute 0 --join_vals_domain_size 1 --check_accuracy
            done
            echo "************************************************"
        done
done
