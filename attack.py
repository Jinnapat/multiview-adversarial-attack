import torch
import pandas as pd
from torch.utils.data import DataLoader

from torch import no_grad
from classes.dataset.marker_dataset import MarkerDataset
from classes.corner_transform import RandomEot, ObjectAwareEot, Homography
from classes.evaluation import evaluate
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import pickle
import os
import argparse
from classes.model_selector import get_model
from torchvision.io import write_png
from classes.robustness_methods.non_printibility_score import nps_loss_function, palette_for_patch
from classes.robustness_methods.total_variation import tv_loss_function
from tqdm import tqdm

# parser
parser = argparse.ArgumentParser(description="Multiview adversarial attack")
parser.add_argument("--dataset_prefix", type=str, default="dataset")
parser.add_argument("--meta_file", type=str, default="meta.csv")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--patch_resolution", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=500)
parser.add_argument("--nps_coef", type=float, default=0.1)
parser.add_argument("--tv_coef", type=float, default=0.01)
parser.add_argument("--train_batch_size", type=int, default=10)
parser.add_argument("--test_batch_size", type=int, default=40)
parser.add_argument("--image_size", type=int, default=20)
args = parser.parse_args()

# setting
patch_resolution = args.patch_resolution
image_size = args.image_size
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size
dataset_prefix = args.dataset_prefix
affine_target_patch_size = 72
num_epochs = args.num_epochs
lr = args.lr
nps_coef = args.nps_coef
tv_coef = args.tv_coef
output_path = args.output_path
image_size = 720

meta_df = pd.read_csv(args.meta_file)

model, preprocess = get_model(args.model)
loss_function = CrossEntropyLoss()

def start_training(scene_data_train, scene_data_eval, transform, scene_name, suffix):
    gt_class = scene_data_eval["gt_class"].head(1).values[0]

    patch = torch.rand(3, patch_resolution, patch_resolution, dtype=torch.float32, device="cuda").requires_grad_(True)
    optimizer = Adam([patch], lr=lr)

    dataset_train = MarkerDataset(dataset_prefix, scene_data_train, transform, image_size, patch=patch)
    dataloader_train = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True)

    dataset_eval = MarkerDataset(dataset_prefix, scene_data_eval, transform, image_size, patch=patch)
    dataloader_eval = DataLoader(dataset_eval, batch_size=test_batch_size, shuffle=False)

    label = torch.zeros(1, 1000, device="cuda")
    label[0, gt_class] = 1.0

    conf_list = []
    top1_list = []
    top3_list = []
    top5_list = []

    conf, top1, top3, top5 = evaluate(model, preprocess, dataloader_eval, gt_class)
    conf_list.append(conf)
    top1_list.append(top1)
    top3_list.append(top3)
    top5_list.append(top5)

    best_conf = conf
    best_patch = patch.detach().clone()

    for _ in tqdm(range(num_epochs)):
        for batch in dataloader_train:
            optimizer.zero_grad()
            images = batch["image"]
            preds = model(preprocess(images))

            loss_obj = loss_function(preds, label.repeat(images.shape[0], 1))
            loss_nps = nps_loss_function(patch)
            loss_tv = tv_loss_function(patch)

            total_loss = (-loss_obj + (nps_coef * loss_nps) + (tv_coef * loss_tv))
            total_loss.backward()

            optimizer.step()

            with no_grad():
                dist = torch.square(patch.unsqueeze(3) - palette_for_patch).sum(dim=0)
                picked_color = dist.min(dim=2).indices
                patch = palette_for_patch[:, 0, 0, picked_color]
        
        conf, top1, top3, top5 = evaluate(model, preprocess, dataloader_eval, gt_class)
        conf_list.append(conf)
        top1_list.append(top1)
        top3_list.append(top3)
        top5_list.append(top5)

        if conf < best_conf:
            best_conf = conf
            best_patch = patch.detach().clone()
        
    int_patch_image = (best_patch * 255.0).to(torch.uint8).cpu()
    patch_save_path = os.path.join(output_path, "patch", f"{scene_name}_{suffix}.png")
    write_png(int_patch_image, patch_save_path)

    pickle_save_path = os.path.join(output_path, "pickle", f"{scene_name}_{suffix}.pkl")
    with open(pickle_save_path, "wb") as f:
        pickle.dump({
            "conf_list": conf_list,
            "top1_list": top1_list,
            "top3_list": top3_list,
            "top5_list": top5_list
        }, f)

if not os.path.exists(output_path):
    os.mkdir(output_path)
patch_folder_path = os.path.join(output_path, "patch")
if not os.path.exists(patch_folder_path):
    os.mkdir(patch_folder_path)
pickle_folder_path = os.path.join(output_path, "pickle")
if not os.path.exists(pickle_folder_path):
    os.mkdir(pickle_folder_path)

for scene_id in meta_df["scene_id"].unique():

    original_only_train = meta_df[(meta_df.scene_id == scene_id) & (meta_df.original) & ~(meta_df.is_eval_set)]
    original_only_eval = meta_df[(meta_df.scene_id == scene_id) & (meta_df.original) & (meta_df.is_eval_set)]

    with_novel_train = meta_df[(meta_df.scene_id == scene_id) & ~(meta_df.is_eval_set)]
    with_novel_eval = meta_df[(meta_df.scene_id == scene_id) & (meta_df.is_eval_set)]
    
    scene_name = original_only_eval["scene_id"].head(1).values[0]

    # single view attack
    transform = RandomEot(image_size, affine_target_patch_size)
    original_only_train_singleview = original_only_train.head(1)
    start_training(original_only_train_singleview, original_only_train_singleview, transform, scene_name, "singleview_original_only")
    
    # RandomEot attack
    transform = RandomEot(image_size, affine_target_patch_size)
    start_training(original_only_train, original_only_eval, transform, scene_name, "randomeot_original_only")
    start_training(with_novel_train, with_novel_eval, transform, scene_name, "randomeot_with_novel")

    # ObjectAware EOT attack
    transform = ObjectAwareEot(affine_target_patch_size)
    start_training(original_only_train, original_only_eval, transform, scene_name, "objecteot_original_only")
    start_training(with_novel_train, with_novel_eval, transform, scene_name, "objecteot_with_novel")

    # Homography attack
    transform = Homography()
    start_training(original_only_train, original_only_eval, transform, scene_name, "homography_original_only")
    start_training(with_novel_train, with_novel_eval, transform, scene_name, "homography_with_novel")