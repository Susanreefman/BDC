#!/bin/bash
#SBATCH --job-name=assignment6
#SBATCH --account=hsreefman
#SBATCH --error=assignment6.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
#SBATCH --cpus-per-task=20

source /commons/conda/conda_load.sh;


export FILE1="Sign_MNIST_mini.dat"
export FILE2="mini_MNIST.dat"

TIMINGS_FILE="timings.csv"

echo "cores,time" > $TIMINGS_FILE

for CORES in 4 8 16 20; do
    START_TIME=$(date +%s)

    parallel -j $CORES python assignment6.py {} ::: $FILE1 $FILE2

    END_TIME=$(date +%s)

    ELAPSED_TIME=$((END_TIME - START_TIME))

    echo "$CORES,$ELAPSED_TIME" >> $TIMINGS_FILE

done

