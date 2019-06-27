"""the entry point into the program"""
import torch

import navigation_2d
from simpleGA import rollout

D_IN, D_OUT = 2, 2


def main():
    from arguments import get_args

    args = get_args()

    env = navigation_2d.Navigation2DEnv()
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    if args.debug:
        rollout(args, env, D_IN, D_OUT, device, pop_size=5, elite_prop=0.2)
    else:
        rollout(args, env, D_IN, D_OUT, device)


if __name__ == "__main__":
    main()
