import torch

def tv_loss_function(images):
    tv_h = torch.sum(torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1]))
    return tv_h + tv_w