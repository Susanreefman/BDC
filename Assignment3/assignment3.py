#!/usr/bin/python3

"""
Assignment 3 Big Data Computing
"""

__author__ = "hsreefman"

# Imports
import sys
import os

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


def process_file(filename):
    """
    Process files 
    Param:
        filename (str): The name of the FASTQ file to be processed.
    Return:
        average (float) or None: The average PHRED score if the process rank is 0, otherwise None.
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    start_index = get_header(lines)

    quality_lines = []
    for i in range(start_index + 3, len(lines), 4):
        quality_lines.append(lines[i])

    chunk_lines = []
    for span in range(0, len(quality_lines), CHUNK_SIZE):
        chunk_lines.append(quality_lines[span: span + CHUNK_SIZE])

    scores = [get_phredscore(chunk) for chunk in chunk_lines]

    means = calculate_mean(scores)

    output= "\n".join([f"{index}, {score}" for index, score in enumerate(means)])

    f = os.path.basename(filename)

    print(f)
    print(output)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python assignment3.py <input_fastq_file>")
        sys.exit(1)

    input_fastq = sys.argv[1]

    process_file(input_fastq)
