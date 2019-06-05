import argparse

import torch

def positive_nonzero_float(x):
    x = float(x)
    if x < 0.0:
        raise argparse.ArgumentTypeError("%r not bigger than 0.0 "%(x,))
    return x

def positive_nonzero_int(x):
    x = int(x)
    if x < 0.0:
        raise argparse.ArgumentTypeError("%r not bigger than 0.0 "%(x,))
    return x

def probability_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not a probability "%(x,))
    return x


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--load-ga', type=bool, default=False,
                        help='Load a saved state from the last generation found in the specified directiory')
    parser.add_argument('--load-ga-dir', default='./trained_models/evolution/___LAST___',
                        help='The directory from which the saved generation will be loaded')
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

    """ Learning specific arguments """
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--deterministic', action='store_true', default=False)
    parser.add_argument('--ep-training', action='store_true', default=False)
    parser.add_argument("--init-sigma", type=positive_nonzero_float, default=1.0,
                        help="initialized value for the exploration sigma parameter")
    parser.add_argument("--module-outputs", type=positive_nonzero_int,
                             help="Define the number of outputs that a single module will have.")
    parser.add_argument("--unfreeze-modules", action='store_true',
                            help="unfreeze the module parameters for gradient optimization")
    args = parser.parse_args()
    args.cuda = False


    return args
