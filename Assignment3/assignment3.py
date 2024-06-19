#!/usr/bin/env python3

"""
Assignment 3: Big Data Computing
"""

import sys
import argparse
import numpy as np


def argument_parser():
    """create an argument parser object"""
    parser = argparse.ArgumentParser(
        description="Script for Assignment 3 of Big Data Computing."
    )
    # Create exclusive group for the two modes
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--chunk",
        action="store_true",
        help="Run the program in chunk parsing mode;",
    )
    mode.add_argument(
        "--combine",
        action="store_true",
        help="Run the program in combine parsing mode;" 
    )
    return parser.parse_args()


def get_quality_line():
    """Get quality line"""
    with sys.stdin.buffer as fastq:
        i = 0
        while i < 2:
            # Skip 3 lines
            if not fastq.readline():
                i += 1
            fastq.readline()
            fastq.readline()
            # Quality line
            yield fastq.readline().strip()


def combine_list(array_list, is_phred=False):
    """Combine list of np arrays"""
    max_rlength = np.array([len(item) for item in array_list]).max()
    
    combine = np.zeros((len(array_list), max_rlength))

    for i, array in enumerate(array_list):
        combine[i, : len(array)] = array

    if is_phred:
        combine[np.nonzero(combine)] -= 33
    return combine


def get_sum_phred():
    """Get sum of phred scores"""
    quality = [np.frombuffer(line, dtype=np.uint8) for line in get_quality_line()]
    quality_array = combine_list(quality, is_phred=True)
    phred_sum = np.sum(quality_array, axis=0)
    phred_count = np.count_nonzero(quality_array, axis=0)
    return phred_sum, phred_count


def main() -> int:
    """Main"""
    args = argument_parser()

    # Chunk
    if args.chunk:
        sum, count = get_sum_phred()
        # Stdout
        print("sum:", list(sum))
        print("count:", list(count))
    # Combine
    elif args.combine:
        sum_array_list = []
        count_array_list = []
        with sys.stdin as lines:
            for line in lines:
                if line.startswith("sum:"):
                    sum_array_list.append(np.fromstring(line.strip()[6:-1], sep=", "))
                elif line.startswith("count:"):
                    count_array_list.append(np.fromstring(line.strip()[8:-1], sep=", "))
        # Combine the arrays
        sum_total = np.sum(combine_list(sum_array_list), axis=0)
        count_total = np.sum(combine_list(count_array_list), axis=0)
        
        # Calculate and stdout avg
        avg = np.divide(sum_total, count_total, dtype=np.float64)
        for i, value in enumerate(avg):
            print(f"{i},{value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
