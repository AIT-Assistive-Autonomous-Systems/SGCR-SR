from OmniSR.losses import CharbonnierLoss
import torch.nn as nn

def get_criterion(opt, device):
    # either CharbonnierLoss or MSELoss
    if opt.reconstruction_loss == "charbonnier":  # starting criterion is charbonnierLoss
        return CharbonnierLoss().to(device)
    elif opt.reconstruction_loss == "l2":
        return nn.MSELoss().to(device)
    else:
        raise ValueError("Choose either charbonnier or L2.")