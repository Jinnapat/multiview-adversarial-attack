from torch.nn.functional import softmax
from torch import topk, no_grad

def evaluate(model, preprocess, dataloader, gt_class):
    conf = 0
    top1 = 0
    top3 = 0
    top5 = 0

    dataset_size = len(dataloader.dataset)

    with no_grad():
        for batch in dataloader:
            images = batch["image"]
            preds = softmax(model(preprocess(images)), dim=1)

            conf += preds[:, gt_class].sum().item()

            top5_classes = topk(preds, k=5).indices

            top1 += (top5_classes[:, 0] == gt_class).sum().item()
            top3 += top1 + (top5_classes[:, 1:3] == gt_class).any(dim=1).sum().item()
            top5 += top3 + (top5_classes[:, 3:] == gt_class).any(dim=1).sum().item()

    return conf / dataset_size, top1 / dataset_size, top3 / dataset_size, top5 / dataset_size






