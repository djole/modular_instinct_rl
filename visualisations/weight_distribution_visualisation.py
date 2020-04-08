import torch
import matplotlib.pyplot as plt

def vis_weights(model):
    dct = model.named_parameters()
    collect_tensor = torch.tensor([])
    for pkey, ptensor in dct:
        pt = ptensor.flatten()
        collect_tensor = torch.cat((collect_tensor, pt))

    print(collect_tensor)
    plt.hist(collect_tensor.detach().numpy(),bins=100)