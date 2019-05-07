'''the entry point into the program'''
from simpleGA import rollout
import navigation_2d

import torch

def main():
    from arguments import get_args
    args = get_args()
    
    env = navigation_2d.Navigation2DEnv()
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cpu')

    if args.debug:
        rollout(args, env, device, pop_size=10, elite_prop=0.1)
    else:
        rollout(args, env, device)


if __name__ == '__main__':
    main()
