SRC_PATH=$(pwd)
BUILD_PATH=$SRC_PATH/build
LOGS_PATH="$BUILD_PATH/results"
INTEL_PATH="/opt/intel/"

source ${INTEL_PATH}/oneapi/2025.1/oneapi-vars.sh
mkdir -p "${BUILD_PATH}"
cd build
# rm  * -rf
mkdir -p "${LOGS_PATH}"

# LOG_RUN_TIMES="$LOGS_PATH/log_run_times.txt"
# touch $LOG_RUN_TIMES
# > $LOG_RUN_TIMES
# exec > ${LOG_RUN_TIMES}

cmake -DMEMORY_USAGE=0 ..
make -j

m1_a=(100 200 400 800 1600 3200 6400)
n_a=(4 8 16)

# for m1 in "${m1_a[@]}"; do
#     m2=$m1
#         for n1 in "${n_a[@]}"; do
#             echo "************************************************"
#             echo "First table: $m1 x $n1 Second table: $m2 x $n1"
#             for i in {1..5} ; do
#                 echo "Run $i"
#                 ./LinScale-Cuda --m1 $(($m1)) --n1 $n1 --m2 $m2 --n2 $n1 --compute 0 --join_vals_domain_size 1
#             done
#             echo "************************************************"
#         done
# done
# python3 $SRC_PATH/scripts/clean_results.py --runtimes --log_path $LOGS_PATH

LOG_ACCURACY="$LOGS_PATH/log_accuracy.txt"
touch $LOG_ACCURACY
> $LOG_ACCURACY
exec > ${LOG_ACCURACY}

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
python3 $SRC_PATH/scripts/clean_results.py --accuracy --log_path $LOGS_PATH

LOG_MEMORY="$LOGS_PATH/log_memory.txt"
touch $LOG_MEMORY
exec > ${LOG_MEMORY}
> $LOG_MEMORY

cmake -DMEMORY_USAGE=1 ..
make -j

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
python3 $SRC_PATH/scripts/clean_results.py --memory --log_path $LOGS_PATH
