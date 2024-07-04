#!/bin/bash
#SBATCH --job-name=assignment3
#SBATCH --account=hsreefman
#SBATCH --error=assignment3.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
#SBATCH --cpus-per-task=20

source /commons/conda/conda_load.sh;

export FILE1="/data/datasets/rnaseq_data/Brazil_Brain/SPM11_R1.fastq"
export FILE2="/data/datasets/rnaseq_data/Brazil_Brain/SPM14_R1.fastq"

parallel -j 20 python assignment3.py {} ::: $FILE1 $FILE2
