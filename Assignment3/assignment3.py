#!/usr/bin/python3

"""
Assignment 3 Big Data Computing
"""

__author__ = "hsreefman"

# Imports
import sys


CHUNK_SIZE = 60

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
    # Flatten the list of lists and compute the lengths
    lengths = [len(sublist[0]) for sublist in chunk_lines]

    # Find the maximum length
    max_length = max(lengths)

    scores = []
    for i in range(max_length):
        score = []
        for line in chunk_lines:
            line = line[0]
            if i < len(line):
                score.append(ord(line[i]) - 33)

            scores.append(sum(score) / len(score))

    return scores


def process_file(filename):
    """
    Process files 
    Param:
        filename (str): The name of the FASTQ file to be processed.
    Return:
        average (float) or None: The average PHRED score if the process rank is 0, otherwise None.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    start_index = get_header(lines)

    quality_lines = []
    for i in range(start_index + 3, len(lines), 4):
        quality_lines.append(lines[i])

    chunk_lines = []
    for span in range(0, len(quality_lines), 1):
        chunk_lines.append(quality_lines[span: span + 1])

    scores = get_phredscore(chunk_lines)

    return scores


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python assignment3.py <input_fastq_file>")
        sys.exit(1)

    input_fastq = sys.argv[1]

    avg_phred = process_file(input_fastq)

    print("\n".join([f"{index}, {score}" for index, score in enumerate(avg_phred)]))
    
