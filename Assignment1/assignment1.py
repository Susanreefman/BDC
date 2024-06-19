#!/usr/bin/python3

"""
Assignment 1 Big Data Computing
"""

__author__ = "hsreefman"

# Imports
import sys
import multiprocessing as mp
import argparse as ap
import os

# Defaults
MAX_CORES = 5
CHUNK_SIZE = 60

# Functions
def arg_parser():
    """
    Parse commandline arguments
    return: 
        args (ap.ArgumentParser)
    """
    argparser = ap.ArgumentParser(description="Script voor Opdracht 1 van Big Data Computing")
    argparser.add_argument("-n", action="store",
                           dest="n", required=False, type=int,
                           help="Aantal cores om te gebruiken.")
    argparser.add_argument("-o", type=str,
                           help="CSV file om de output in op te slaan. Default is output naar "
                                "terminal STDOUT")
    argparser.add_argument("fastq_files", type=str, nargs='+',
                           help="Minstens 1 Illumina Fastq Format file om te verwerken")
    args = argparser.parse_args()
    return args


def read_file(filepath):
    """
    Read input files from input file path
    Param:
        filepath (str): string containing path to file
    Return:
        (list): separated lines in list
    """
    try:
        with open(filepath, "r") as readfile:
            return readfile.read().splitlines()
    except FileExistsError:
        print("File: {filepath} does not exist")


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


def write_output(outputfile, output_string, multiple, f):
    """
    Write output to command line as default or else to given file
    param: output file the output with scores multiple, int for amount of files header string
    """
    if outputfile:
        if multiple:
            print(f, file=outputfile)
        print(output_string, file=outputfile)
    else:
        if multiple:
            print(f)

        print(output_string)


def process_file(filepath, cores, outputfile=None, multiple=False):
    """
    Calculates the mean phred score for every position
    Write to output file if given, else print on command line
    param:
        filepath (str): string with path to file
        cores (int): number of cores to work with
        outputfile (str): string with path to output file
        multiple (boolean):
    """
    # Read lines from input
    lines = read_file(filepath)
    # Get index of header line
    start_index = get_header(lines)
    # make chunks
    chunks = []
    for i in range(start_index + 3, len(lines), 4):
        chunks.append(lines[i])

    chunk_lines = []
    for span in range(0, len(chunks), CHUNK_SIZE):
        chunk_lines.append(chunks[span: span + CHUNK_SIZE])
    # with given amount of cores calculate phredscore
    with mp.Pool(cores) as pool:
        results = pool.map(get_phredscore, chunk_lines)

    means = calculate_mean(results)
    output= "\n".join([f"{index}, {score}" for index, score in enumerate(means)])

    f = os.path.basename(filepath)

    # write output
    write_output(outputfile, output, multiple, f)


def main():
    # Command line arguments parsed
    args = arg_parser()
    fastq_files = args.fastq_files
    if args.n is not None:
        max_cores = args.n
    else:
        max_cores = MAX_CORES

    # process files, if output file is given, open file and process,
    if args.o is not None:
        with open(args.o, "w") as output:
            for f in fastq_files:
                process_file(f, max_cores, output, multiple=len(fastq_files)>1)
    else: # else proces file and write to command line.
        for f in fastq_files:
            process_file(f, max_cores, multiple=len(fastq_files)>1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
