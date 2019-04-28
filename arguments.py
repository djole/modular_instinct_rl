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
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--load-ga', type=bool, default=False,
                        help='Load a saved state from the last generation found in the specified directiory')
    parser.add_argument('--load-ga-dir', default='./trained_models/evolution/',
                        help='The directory from which the saved generation will be loaded')
    parser.add_argument('--exp-idx', type=positive_nonzero_int, default=0,
                        help='experiment index: use only for testing the code')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
