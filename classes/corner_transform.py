from torch import tensor, rand, sin, cos, float32, matmul

def scale_and_rotate(target_patch_size):
    scaled_patch_size = (rand(1).item() * 0.2 + 0.9) * target_patch_size
    rotate_rad = (rand(1) * 2 - 1.0) * 0.3490658504
    rotation_matrix = tensor([[cos(rotate_rad), -sin(rotate_rad)], [sin(rotate_rad), cos(rotate_rad)]], dtype=float32)

    corners = tensor([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=float32) * scaled_patch_size
    corners = matmul(rotation_matrix, corners.T).T
    return corners

class RandomEot(object):
    def __init__(self, image_size, target_patch_size):
        self.target_patch_size = target_patch_size

        border_offset = target_patch_size * 2.0
        self.min_translate = border_offset
        self.max_translate = image_size - border_offset
        self.translate_range = self.max_translate - self.min_translate

    def __call__(self, _):
       corners = scale_and_rotate(self.target_patch_size)
       corners += rand(1, 2) * self.translate_range + self.min_translate
       return corners

class ObjectAwareEot(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        min_x = sample[:, 0].min()
        min_y = sample[:, 1].min()
        max_x = sample[:, 0].max()
        max_y = sample[:, 1].max()

        range_translate = tensor([[max_x - min_x, max_y - min_y]], dtype=float32)
        min_translate = tensor([[min_x, min_y]], dtype=float32)

        corners = scale_and_rotate(self.patch_size)
        corners += rand(1, 2) * range_translate + min_translate

        return corners
    
class Homography(object):
    def __call__(self, sample):
       return sample