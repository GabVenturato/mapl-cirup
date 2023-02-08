import sys
import os
import datetime
import subprocess
import multiprocessing
import re
from collections import namedtuple

from problog.util import init_logger
from mapl_cirup import MaplCirup

TIMEOUT = 5 * 60  # seconds
RUNS = 10


def run_experiment(input_file, discount, error, res):
    mc = MaplCirup(input_file)
    res.put((mc.size(), mc.compile_time()))
    mc.value_iteration(discount=discount, error=error)
    res.put((mc.value_iteration_time(), mc.tot_time(), mc.iterations()))


def main(argv):
    args = argparser().parse_args(argv)

    # create directory to save results
    res_dir = 'results'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    if args.experiments:
        print("=== Experimental Evaluation ===")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        with open(res_dir + '/' + timestamp + '_exp-eval.csv', 'w') as fres:
            header = "run,solver,filename,family,var_num,timeout,discount,error," \
                     "circuit_size,compile_time,vi_time,tot_time,vi_iterations\n"
            fres.write(header)

            for run in range(1, RUNS+1):
                # mapl-cirup
                solver = "mapl-cirup"
                print("\n\nMAPL-CIRUP\n")
                for root, dirs, files in os.walk("./examples"):
                    family = os.path.basename(os.path.normpath(root))
                    for file in files:
                        if file.endswith(".pl"):
                            # retrieve var number
                            var_num = 'n'
                            var_num_pattern = re.compile("_v([0-9]+)")
                            match = re.search(var_num_pattern, file)
                            if match:
                                var_num = match.group(1)

                            # execute experiments
                            input_file = os.path.join(root, file)
                            print("\n\n-> Executing mapl-cirup on %s..." % input_file)
                            res = multiprocessing.Queue()
                            discount = 0.9
                            error = 0.1
                            p = multiprocessing.Process(target=run_experiment, args=(input_file, discount, error, res))
                            p.start()
                            p.join(TIMEOUT)

                            if p.is_alive():
                                p.kill()

                                if res.empty():
                                    # compilation didn't finish
                                    fres.write("%s,%s,%s,%s,%s,%s,%s,na,na,na,na,na\n" %
                                               (run,solver, input_file, family, TIMEOUT, discount, error))
                                else:
                                    # compilation finished, but value iteration not
                                    size, compile_time = res.get()
                                    fres.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,na,%s,na\n" %
                                               (run, solver, input_file, family, var_num, TIMEOUT, discount, error,
                                                size, compile_time, compile_time))
                            else:
                                # everything finished
                                size, compile_time = res.get()
                                vi_time, tot_time, iterations = res.get()
                                fres.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" %
                                           (run, solver, input_file, family, var_num, TIMEOUT, discount, error, size,
                                            compile_time, vi_time, tot_time, iterations))

                            fres.flush()

                # spudd
                solver = "spudd"
                print("\n\nSPUDD\n")
                for root, dirs, files in os.walk("./examples"):
                    for file in files:
                        if file.endswith(".dat"):
                            # retrieve var number
                            var_num = 'n'
                            var_num_pattern = re.compile("_v([0-9]+)")
                            match = re.search(var_num_pattern, file)
                            if match:
                                var_num = match.group(1)

                            # execute experiments
                            input_file = os.path.join(root, file)
                            print("\n\n-> Executing spudd on %s..." % input_file)

                            # find discount and error in input file
                            discount = "na"
                            error = "na"
                            discount_pattern = re.compile("discount\s+([0-9\.]+)")
                            error_pattern = re.compile("tolerance\s+([0-9\.]+)")
                            for line in open(input_file):
                                match = re.search(discount_pattern, line)
                                if match:
                                    discount = round(float(match.group(1)), 2)
                                match = re.search(error_pattern, line)
                                if match:
                                    error = match.group(1)

                            try:
                                spudd_proc = subprocess.Popen(
                                    ["./examples/executables/spudd-linux", input_file, "-o", res_dir + "/spudd"]
                                )
                                spudd_proc.communicate(timeout=TIMEOUT)

                                # find stats
                                size = "na"
                                vi_time = "na"
                                iterations = "na"
                                size_pattern = re.compile("Total number of nodes allocated:\s([0-9]+)")
                                vi_time_pattern = re.compile("\sFinal execution time:\s+([0-9\.]+)")
                                iterations_pattern = re.compile("\sIterations to convergence\s([0-9]+)")

                                for line in open(res_dir + "/spudd-stats.dat"):
                                    match = re.search(size_pattern, line)
                                    if match:
                                        size = match.group(1)
                                    match = re.search(vi_time_pattern, line)
                                    if match:
                                        vi_time = round(float(match.group(1)), 2)
                                    match = re.search(iterations_pattern, line)
                                    if match:
                                        iterations = match.group(1)

                                fres.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,na,%s,%s,%s\n" %
                                           (run,solver, input_file, family, var_num, TIMEOUT, discount, error, size,
                                            vi_time, vi_time, iterations))
                                os.remove(res_dir + "/spudd-stats.dat")
                                os.remove(res_dir + "/spudd-OPTDual.ADD")
                            except subprocess.TimeoutExpired:
                                spudd_proc.kill()
                                fres.write("%s,%s,%s,%s,%s,%s,%s,%s,na,na,na,na,na\n" %
                                           (run, solver, input_file, family, var_num, TIMEOUT, discount, error))

                            fres.flush()
    else:
        if args.file is None:
            input_file = input("File path: ")
        else:
            input_file = args.file
        init_logger(args.verbose)

        mc: MaplCirup = MaplCirup(input_file)

        print("Compile time: %s" % mc.compile_time())
        # print("Minimize time: %s" % mc.minimize_time())
        print("Circuit size: %s" % mc.size())

        # mc.value_iteration(discount=0.9, error=0.01)
        mc.value_iteration(discount=0.9, error=0.1)
        # mc.value_iteration(discount=0.9)
        # mc.value_iteration(discount=0.9, horizon=1)
        # mc.value_iteration(horizon=10)
        # mc.value_iteration()  # = immediate reward

        print("Value iteration time: %s" % mc.value_iteration_time())
        print("Total time: %s" % mc.tot_time())
        print('\nNumber of iterations: ' + str(mc.iterations()) + '\n')
        # mc.view_dot()
        # mc.print_explicit_policy()


def argparser():
    import argparse

    parser = argparse.ArgumentParser(description="MArkov PLanning with CIRcuit bellman UPdates.")
    parser.add_argument('-f', '--file', help='Path to the input file')
    # TODO: Implement the following parameter
    # parser.add_argument('-o', '--output', type=str, default=None,
    #                     help='Write output to given file (default: stdout)')
    parser.add_argument('-v', '--verbose', help='Verbose mode')
    parser.add_argument('--experiments', help='Run experimental evaluation', action='store_true')
    return parser


if __name__ == '__main__':
    main(sys.argv[1:])
