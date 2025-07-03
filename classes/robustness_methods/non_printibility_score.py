from torch import abs, float32, tensor

nps_palette = [
    [26, 24, 52],
    [35, 39, 180],
    [59, 90, 44],
    [61, 90, 96],
    [65, 94, 161],
    [72, 91, 216],
    [80, 159, 63],
    [88, 24, 50],
    [88, 40, 180],
    [88, 160, 128],
    [94, 217, 85],
    [100, 160, 190],
    [105, 160, 235],
    [109, 222, 138],
    [125, 214, 204],
    [147, 91, 196],
    [153, 22, 50],
    [153, 88, 71],
    [155, 37, 169],
    [184, 215, 227],
    [185, 219, 131],
    [187, 216, 185],
    [190, 161, 203],
    [191, 218, 78],
    [193, 160, 127],
    [198, 161, 64],
    [204, 94, 177],
    [209, 38, 155],
    [210, 29, 51],
    [216, 89, 68]
]

palette = tensor(nps_palette, dtype=float32).view(1, 3, 1, 1, 30) / 255.0

def nps_loss_function(images, palette=palette):
    images = images.unsqueeze(dim=4)
    pixel_wise_nps = abs(images - palette).min(dim=4).values
    return pixel_wise_nps.sum() / images.shape[0]
