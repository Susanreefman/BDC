#!/bin/bash
#SBATCH --job-name=assignment3
#SBATCH --account=hsreefman
#SBATCH --error=assignment3.err
#SBATCH --time=00:2:00
#SBATCH --nodes=1
#SBATCH --partition=assemblix
#SBATCH --nodelist=assemblix2019
#SBATCH --cpus-per-task=20

source /commons/conda/conda_load.sh;

export FILE="/commons/Themas/Thema12/HPC/rnaseq.fastq";

parallel --jobs 20 --pipepart --block -1 --regexp --recstart '@.*(/1| 1:.*)\n[A-Za-z\n\.~]' --recend '\n' -a $FILE python3 assignment3.py --chunk | python3 assignment3.py --combine;
