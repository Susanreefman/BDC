#!/bin/bash
#SBATCH --job-name=assignment4
#SBATCH --account=hsreefman
#SBATCH --error=assignment4.err
#SBATCH --time=00:5:00
#SBATCH --nodes=1
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
#SBATCH --cpus-per-task=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5

export FILE="/data/datasets/rnaseq_data/Brazil_Brain/SPM11_R1.fastq"

source /commons/conda/conda_load.sh;

time mpiexec -np 5 python3 assignment4.py $FILE;  #-np is same as --ntasks in slurm