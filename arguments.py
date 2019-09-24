""" Program arguments """
import argparse


def positive_nonzero_float(x):
    x = float(x)
    if x < 0.0:
        raise argparse.ArgumentTypeError("%r not bigger than 0.0 " % (x,))
    return x


def positive_nonzero_int(x):
    x = int(x)
    if x < 0.0:
        raise argparse.ArgumentTypeError("%r not bigger than 0.0 " % (x,))
    return x


def probability_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not a probability " % (x,))
    return x


def get_args():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--save-dir",
        default="./trained_models/",
        help="directory to save agent logs (default: ./trained_models/)",
    )
    parser.add_argument(
        "--load-ga",
        action="store_true",
        help="Load a saved state from the last generation found in the specified directiory",
    )
    parser.add_argument(
        "--load-ga-dir",
        default="./trained_models/evolution/___LAST___",
        help="The directory from which the saved generation will be loaded",
    )
    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument(
        "--seed", type=int, default=543, metavar="N", help="random seed (default: 543)"
    )
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="interval between training status logs (default: 10)",
    )

    """ Learning specific arguments """
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning rate (default: 0.01)"
    )
    parser.add_argument("--ep-training", action="store_true", default=False)

    parser.add_argument(
        "--init-sigma",
        type=positive_nonzero_float,
        default=1.0,
        help="initialized value for the exploration sigma parameter",
    )
    parser.add_argument(
        "--module-outputs",
        type=positive_nonzero_int,
        default=2,
        help="Define the number of outputs that a single module will have.",
    )
    parser.add_argument(
        "--hidden-size",
        type=positive_nonzero_int,
        default=100,
        help="Define the number of neurons in the hidden layer of a module.",
    )
    parser.add_argument(
        "--num-modules",
        type=positive_nonzero_int,
        default=4,
        help="Define the number of modules in the model.",
    )

    """Arguments that change the behaviour of the model """
    parser.add_argument(
        "--num-proc",
        type=positive_nonzero_int,
        default=5,
        help="Define over how many processes to parallelize the fitness evaluation.",
    )
    parser.add_argument(
        "--unfreeze-modules",
        action="store_true",
        default=False,
        help="unfreeze the module parameters for gradient optimization",
    )
    parser.add_argument(
        "--monolithic-baseline",
        action="store_true",
        default=False,
        help="run the experiment with a monolithic network model.",
    )
    parser.add_argument(
        "--sees-inputs",
        action="store_true",
        default=False,
        help=" If TRUE, the combinator sees the inputs to the model as well",
    )
    parser.add_argument(
        "--rm-nogo",
        action="store_true",
        default=False,
        help="if TRUE, the supported environment will not have no-go zones",
    )
    parser.add_argument(
        "--rm-exploration-fit",
        action="store_true",
        default=False,
        help="if TRUE, the exploration fitness will be removed",
    )
    parser.add_argument(
        "--load-instinct",
        action="store_true",
        default=False,
        help="if TRUE, the pretrained instinct will be loaded from a file",
    )
    model_type_group = parser.add_mutually_exclusive_group()
    model_type_group.add_argument(
        "--parametric-combinator",
        action="store_true",
        default=False,
        help="if TRUE, genetic algorithm will instantiate parametric combinator",
    )
    model_type_group.add_argument(
        "--instinct-sigma",
        action="store_true",
        default=False,
        help="If TRUE, initialize the model that uses an instinct network that outputs the sigma value of the exploration distribution",
    )

    parser.add_argument(
        "--reduce-goals",
        action="store_true",
        default=False,
        help="if TRUE, run meta-learning on two predetermined goals. Used for quick experiments.",
    )
    args = parser.parse_args()
    args.cuda = False

    return args
