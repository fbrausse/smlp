#!/usr/bin/env python3
#
# This file is part of smlprover.
# It is a top level script to run smlprover (SMLP)
#
# Copyright 2019 Konstantin Korovin
# Copyright 2020 Franz Brauße <franz.brausse@manchester.ac.uk>
# See the LICENSE file for terms of distribution.

# coding: utf-8


import sys
from icecream import ic
from smlp_py.ext import plot
from smlp_py.smlp_flows import SmlpFlows
import threading
import time
import psutil
import os

ic.configureOutput(prefix=f'Debug | ', includeContext=True)

start_time = time.time()

plot_instance = plot.plot_exp()

log_file_path = 'memory_usage_log.txt'

interval = 5

def main(argv):

    plot_instance.copy_from()
    plot_instance.save_to_txt(argv)
    smlpInst = SmlpFlows(argv)
    smlpInst.smlp_flow()

    process = psutil.Process(os.getpid())
    max_memory = 0

    for _ in range(10):
        memory_usage = process.memory_info().rss / 1024 ** 2 
        max_memory = round(max(max_memory, memory_usage), 4)
        time.sleep(0.1)

    Memory = "Maximum memory usage: " + str(max_memory)
    plot_instance.save_to_txt(Memory)

def log_memory_usage():
    # Open the log file in append mode
    with open(log_file_path, 'a') as log_file:
        while True:
            # Get memory usage
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 ** 2  # Memory usage in MB

            # Format memory usage to 4 decimal places
            formatted_memory_usage = f"{memory_usage: .4f}"

            # Write the memory usage to the file
            log_file.write(f"Memory usage: {formatted_memory_usage} MB\n")

            # Flush the file buffer to ensure immediate write
            log_file.flush()

            # Wait for the next interval
            time.sleep(interval)

if __name__ == "__main__":

    max_memory = [0]

    # Start memory logging in a separate thread
    logging_thread = threading.Thread(target=log_memory_usage, daemon=True)
    logging_thread.start()

    try:
        main(sys.argv)
    except KeyboardInterrupt:
        print("Execution stopped.")

end_time = time.time()
total_time = end_time-start_time
plot_instance.save_to_txt(total_time)
