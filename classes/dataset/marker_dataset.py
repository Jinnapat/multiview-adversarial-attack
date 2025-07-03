from torch.utils.data import Dataset
import os
from torchvision.io import read_image
import torch
from torchvision.transforms.functional import perspective
from torchvision.transforms.v2 import InterpolationMode
import torch
from kornia.utils import draw_convex_polygon

class MarkerDataset(Dataset):
    def __init__(self, dataset_prefix, meta_df, corner_transform, patch):
        self.dataset_prefix = dataset_prefix
        self.meta_df = meta_df
        self.corner_transform = corner_transform
        self.patch = patch
        self.patch_starting_points = [[0, 0], [20, 0], [20, 20], [0, 20]]

    def __len__(self):
        return self.meta_df.shape[0]

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_prefix, self.meta_df["image_name"].iloc[idx])
        image = read_image(image_path) / 255.0
        corners = self.corner_transform(torch.tensor(self.meta_df.iloc[idx, 4:12].values.astype(int).reshape(4, 2)))
        
        image_height = image.shape[1]
        image_width = image.shape[2]

        full_patch = torch.zeros(3, image_height, image_width, dtype=torch.float32)
        full_patch[:, :self.patch.shape[1], :self.patch.shape[2]] = self.patch

        transformed_patch = perspective(
            full_patch,
            self.patch_starting_points,
            corners,
            interpolation=InterpolationMode.BILINEAR
        )

        mask = draw_convex_polygon(torch.zeros(1, 3, image_height, image_width, dtype=torch.float32), corners.unsqueeze(0), torch.tensor([1.0]))
        image = transformed_patch * mask + image * (1.0 - mask)

        sample = {
            "image": image[0],
        }

        return sample