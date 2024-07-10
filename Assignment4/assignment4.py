#!/usr/bin/env python3

"""
Assignment 4: Big Data Computing

hsreefman
"""

import sys
from mpi4py import MPI


def get_header(lines):
    """
    Determines the header line number
    Param:
        lines (list): list with separate lines
    Return:
        index (int): index of start position fastq
    """
    for index, line in enumerate(lines):
        if line.startswith("@"):
            return index
    return -1


def get_phredscore(chunk_lines):
    """
    Calculates the phred mean scores and puts them in a list
    Param:
        chunk_lines (string):
    Return:
        vertical_list (list):
    """
    max_length = max(len(line) for line in chunk_lines)

    scores = []
    for i in range(max_length):
        score = []
        for line in chunk_lines:
            if i < len(line):
                score.append(ord(line[i]) - 33)
        scores.append(sum(score) / len(chunk_lines))

    return scores


def calculate_mean(scores):
    """
    Calculates the mean scores of all lines
    param:
        scores (list): list with scores
    return:
        means (list): list with mean scores
    """
    max_length = max(len(score) for score in scores)

    if not all(map(lambda x:len(x) == max_length, scores)):
        raise ValueError("Quality lines do not have equal length")

    means = []
    for i in range(max_length):
        mean = [score[i] for score in scores]
        means.append(sum(mean) / len(mean))
    return means



def process_file(filename, comm, rank, size):
    """
    Process files 
    Param:
        filename (str): The name of the FASTQ file to be processed.
        comm (MPI.Comm): The MPI communicator.
        rank (int): The rank of the current MPI process.
        size (int): The total number of MPI processes.
    Return:
        average (float) or None: The average PHRED score if the process rank is 0, otherwise None.
    """
    if rank == 0:
        with open(filename, 'r') as f:
            lines = f.readlines()

        start_index = get_header(lines)
        quality_lines = []
        for i in range(start_index + 3, len(lines), 4):
            quality_lines.append(lines[i])

        chunks = [quality_lines[i::size] for i in range(size)]
    else:
        chunks = None

    chunk = comm.scatter(chunks, root=0)
    local_scores = get_phredscore(chunk)
    all_scores = comm.gather(local_scores, root=0)



    if rank == 0:
        means = calculate_mean(all_scores)
        output= "\n".join([f"{index}, {score}" for index, score in enumerate(means)])
        return output
    return None


def main():
    """Main function"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        file_list = sys.argv[1:]
    else:
        file_list = None

    file_list = comm.bcast(file_list, root=0)

    results = []
    for filename in file_list:
        avg_phred = process_file(filename, comm, rank, size)
        if rank == 0:
            results.append((filename, avg_phred))

    return 0


if __name__ == '__main__':
    sys.exit(main())
