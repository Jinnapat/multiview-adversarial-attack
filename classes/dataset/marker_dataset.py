import os
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms.functional import perspective

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision.transforms.v2 import InterpolationMode
from kornia.utils import draw_convex_polygon

mask_color = torch.tensor([1.0], device="cuda")

class MarkerDataset(Dataset):
    def __init__(self, dataset_prefix, meta_df, corner_transform, image_size, patch):
        self.dataset_prefix = dataset_prefix
        self.meta_df = meta_df
        self.corner_transform = corner_transform
        self.patch = patch
        self.image_size = image_size

        self.corner_cols = ["x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"]

        patch_size = patch.shape[1]
        self.patch_starting_points = [[0, 0], [patch_size, 0], [patch_size, patch_size], [0, patch_size]]

        self.full_patch = torch.zeros(3, image_size, image_size, dtype=torch.float32, device="cuda")
        self.images = []

        for image_name in meta_df["image_name"]:
            image_path = os.path.join(self.dataset_prefix, image_name)
            image = read_image(image_path).cuda() / 255.0
            self.images.append(image)

    def __len__(self):
        return self.meta_df.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]

        corners_data = self.meta_df.loc[idx, self.corner_cols].values.astype(int).reshape(4, 2)
        corners = self.corner_transform(torch.tensor(corners_data, device="cuda"))

        full_patch = self.full_patch.clone()
        full_patch[:, :self.patch.shape[1], :self.patch.shape[2]] = self.patch

        transformed_patch = perspective(
            full_patch,
            self.patch_starting_points,
            corners,
            interpolation=InterpolationMode.BILINEAR
        )

        mask = draw_convex_polygon(torch.zeros(1, 3, self.image_size, self.image_size, dtype=torch.float32, device="cuda"), corners.unsqueeze(0), mask_color)
        image = transformed_patch * mask + image * (1.0 - mask)

        sample = {
            "image": image[0],
        }

        return sample