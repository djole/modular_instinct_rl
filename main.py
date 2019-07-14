"""the entry point into the program"""
import torch
from simpleGA import rollout

D_IN, D_OUT = 2, 2


def main():
    from arguments import get_args

    args = get_args()
    device = torch.device("cpu")

    if args.debug:
        rollout(args, D_IN, D_OUT, device, pop_size=5, elite_prop=0.2)
    else:
        rollout(args, D_IN, D_OUT, device, pop_size=(args.num_proc * 2))


if __name__ == "__main__":
    main()
