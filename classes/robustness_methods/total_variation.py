from torch import sum, abs

def tv_loss_function(images):
    tv_h = sum(abs(images[:, :, 1:, :] - images[:, :, :-1, :]))
    tv_w = sum(abs(images[:, :, :, 1:] - images[:, :, :, :-1]))
    return tv_h + tv_w