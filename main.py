import sys

from problog.util import init_logger
from mapl_cirup import MaplCirup

MAX_ACTS = 10   # Max number of steps in the online algorithm


def main(argv):
    args = argparser().parse_args(argv)

    input_file = args.inputfile
    init_logger(args.verbose)

    mc: MaplCirup = MaplCirup(input_file)

    print("Compile time: %s" % mc.compile_time())
    # print("Minimize time: %s" % mc.minimize_time())
    print("Circuit size: %s" % mc.size())

    # mc.value_iteration(discount=0.9, error=0.01)
    mc.value_iteration(discount=0.9, error=0.1)
    # mc.value_iteration(discount=0.9)
    # mc.value_iteration(discount=0.9, horizon=39)
    # mc.value_iteration(horizon=10)
    # mc.value_iteration()  # = immediate reward

    print("Value iteration time: %s" % mc.value_iteration_time())
    print("Total time: %s" % mc.tot_time())
    print('\nNumber of iterations: ' + str(mc.iterations()) + '\n')
    mc.print_explicit_policy()


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
