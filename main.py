import sys

from problog.util import init_logger
from ddc import DDC

MAX_ACTS = 10   # Max number of steps in the online algorithm


def main(argv):
    args = argparser().parse_args(argv)

    input_file = args.inputfile
    init_logger(args.verbose)

    ddc: DDC = DDC(input_file)

    print("Compile time: %s" % ddc.compile_time())
    print("Minimize time: %s" % ddc.minimize_time())
    print("Circuit size: %s" % ddc.size())

    ddc.value_iteration(discount=0.9, error=0.1)
    # ddc.value_iteration(horizon=3)

    print("Value iteration time: %s" % ddc.value_iteration_time())
    print("Total time: %s" % ddc.tot_time())
    print('\nNumber of iterations: ' + str(ddc.iterations()) + '\n')
    ddc.print_explicit_policy()


def argparser():
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Decision Circuits solver based on ProbLog.")
    parser.add_argument('inputfile')
    # TODO: Implement the following parameter
    # parser.add_argument('-o', '--output', type=str, default=None,
    #                     help='Write output to given file (default: stdout)')
    parser.add_argument('-v', '--verbose', help='Verbose mode')
    return parser


if __name__ == '__main__':
    main(sys.argv[1:])
