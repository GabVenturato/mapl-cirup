import sys

from problog.util import init_logger
from mapl_cirup import MaplCirup


def main(argv):
    args = argparser().parse_args(argv)

    if args.experiments:
        print("I'll run the experimental evaluation")
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
