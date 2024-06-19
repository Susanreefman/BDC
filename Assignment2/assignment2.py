#!/usr/bin/python3

"""
Assignment 2 Big Data Computing
"""
__author__ = "hsreefman"

import os
import queue
import sys
import time
import multiprocessing as mp
import argparse as ap
from multiprocessing.managers import BaseManager

POISONPILL = "MEMENTOMORI"
ERROR = "Encountered an error"
IP = ''
PORTNUM = 5382
AUTHKEY = b'0cec8b70230d94f5?'
CHUNK_COUNT = 25
NUM_PROCESSES = 4

# Functions
def arg_parser():
    """
    Parse commandline arguments
    return: 
        args (ap.ArgumentParser)
    """
    argparser = ap.ArgumentParser(description="Script voor Opdracht 2 van Big Data Computing;  Calculate PHRED scores over the network.")
    mode = argparser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-s", action="store_true", help="Run the program in Server mode; see extra options needed below")
    mode.add_argument("-c", action="store_true", help="Run the program in Client mode; see extra options needed below")
    server_args = argparser.add_argument_group(title="Arguments when run in server mode")
    server_args.add_argument("-o", action="store", dest="csvfile", type=ap.FileType('w', encoding='UTF-8'),
                           required=False, help="CSV file om de output in op te slaan. Default is output naar terminal STDOUT")
    # server_args.add_argument("fastq_files", type=str,  action="store", nargs='+',
                        #    help="Minstens 1 Illumina Fastq Format file om te verwerken")
    server_args.add_argument("fastq_files", action="store", type=str, nargs='*', help="Minstens 1 Illumina Fastq Format file om te verwerken")
    server_args.add_argument("--chunks", action="store", type=int, required=False)

    client_args = argparser.add_argument_group(title="Arguments when run in client mode")
    client_args.add_argument("-n", action="store",
                           dest="n", required=False, type=int,
                           help="Aantal cores om te gebruiken per host.")
    client_args.add_argument("--host", action="store", type=str, help="The hostname where the Server is listening")
    client_args.add_argument("--port", action="store", type=int, help="The port on which the Server is listening")

    args = argparser.parse_args()
    return args


def make_server_manager(host, port, authkey):
    """
    Create a manager for the server, listening on the given port.
    Return a manager object with get_job_q and get_result_q methods.
    """
    job_q = queue.Queue()
    result_q = queue.Queue()

    # This is based on the examples in the official docs of multiprocessing.
    # get_{job|result}_q return synchronized proxies for the actual Queue
    # objects.
    class QueueManager(BaseManager):
        """
        Renamed class of QueueManager
        """

    QueueManager.register('get_job_q', callable=lambda: job_q)
    QueueManager.register('get_result_q', callable=lambda: result_q)

    manager = QueueManager(address=(host, port), authkey=authkey)
    manager.start()
    # print('Server started at port %s' % port)
    return manager


def runserver(func, data, outfile_path, host, port):
    """
    Main function that runs for server
    """
    # Start a shared manager server and access its queues
    manager = make_server_manager(host, port, AUTHKEY)
    shared_job_q = manager.get_job_q()
    shared_result_q = manager.get_result_q()

    if not data:
        print("Give me something to do!")
        return
    print("Sending data!")

    for job_data in data:
        shared_job_q.put({'fn': func, 'arg': job_data})

    time.sleep(2)

    results = []
    while True:
        try:
            result = shared_result_q.get_nowait()
            results.append(result)
            if len(results) == len(data):
                break
        except queue.Empty:
            time.sleep(1)
            continue

    # Tell the client process no more data will be forthcoming
    shared_job_q.put(POISONPILL)
    # Sleep a bit before shutting down the server - to give clients time to
    # realize the job queue is empty and exit in an orderly way.
    time.sleep(5)

    if outfile_path:
        with open(outfile_path, "w") as file:
            write_results(results, file)
    else:
        write_results(results, sys.stdout)
    manager.shutdown()


def write_results(results, outfile):
    """
    Writes results to file or stdout
    """
    for result in results:
        name = os.path.basename(result["job"]["arg"])
        result = result["result"]
        if len(results) > 1:
            print(name, file=outfile)
        print(result, file=outfile)


def make_client_manager(host, port, authkey):
    """ Create a manager for a client. This manager connects to a server on the
        given address and exposes the get_job_q and get_result_q methods for
        accessing the shared queues from the server.
        Return a manager object.
    """

    class ServerQueueManager(BaseManager):
        """
        Renamed class of QueueManager
        """

    ServerQueueManager.register('get_job_q')
    ServerQueueManager.register('get_result_q')

    manager = ServerQueueManager(address=(host, port), authkey=authkey)
    manager.connect()

    return manager


def runclient(num_processes, chunksize, host, port):
    """
    Setup client
    """
    manager = make_client_manager(host, port, AUTHKEY)
    job_q = manager.get_job_q()
    result_q = manager.get_result_q()
    run_workers(job_q, result_q, num_processes, chunksize)


def run_workers(job_q, result_q, num_processes, chunksize):
    """
    Runs the worker
    """
    while True:
        try:
            job = job_q.get_nowait()
            if job == POISONPILL:
                job_q.put(POISONPILL)
                return
            else:
                try:
                    result = job["fn"](job["arg"], chunksize, num_processes)
                    result_q.put({'job': job, 'result': result})
                except NameError:
                    result_q.put({'job': job, 'result': ERROR})

        except queue.Empty:
            time.sleep(1)


def get_average_phred(filepath, chunksize, num_processes):
    """
    Calculates the mean phred score for every position
    """""
    file = read_file(filepath)
    start_ind = get_header(file)

    chunks = [file[x] for x in range(start_ind + 3, len(file), 4)]

    chunkssplit = [chunks[chunkspan: chunkspan + chunksize] for chunkspan in
                range(0, len(chunks), chunksize)]

    with mp.Pool(num_processes) as pool:
        results = pool.map(get_phredscore, chunkssplit)

    means = calculate_mean(results)
    output = create_output_string(means)
    return output


def create_output_string(average_score):
    """
    Creates an output string. Which can then be written to stout or file
    Param:
        average_score
    Return:
        (str)
    
    """
    return "\n".join([f"{ind},{score}" for ind, score in enumerate(average_score)])



def get_phredscore(chunk_lines):
    """
    Calculates the phred mean scores and puts them in a list
    Param:
        chunk_lines (string): lines from file
    Return:
        scores (list): calculated scores
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

    means = []
    for i in range(max_length):
        mean = [score[i] for score in scores]
        means.append(sum(mean) / len(mean))
    return means


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

def read_file(filepath):
    """
    Reads a file and parses as list containing individual lines
    param:
        filepath (str): path to file
    return: 
        (list): lines in file
    """
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            return file.read().splitlines()
    else:
        raise FileExistsError(f"File: {filepath} does not exist!")

def main():
    """Main"""
    args = arg_parser()
    host = args.host if args.host else IP
    port = args.port if args.port else PORTNUM
    chunks = args.chunks if args.chunks else CHUNK_COUNT
    num_processes = args.n if args.n else NUM_PROCESSES

    if args.s: #If in server mode
        server = mp.Process(target=runserver,
                            args=(get_average_phred, args.fastq_files,
                                  args.csvfile, host, port))
        server.start()
        time.sleep(1)
        server.join()
    else: #Client mode
        client = mp.Process(target=runclient, args=(num_processes, chunks, host, port))
        client.start()
        time.sleep(1)
        client.join()


if __name__ == '__main__':
    sys.exit(main())
