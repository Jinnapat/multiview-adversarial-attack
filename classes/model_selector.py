from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights

def get_model(model_name):
    if model_name == "resnet":
        weight = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weight).cuda()
        model.eval()
        preprocess = weight.transforms(antialias=True)
    elif model_name == "vit":
        weight = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weight).cuda()
        model.eval()
        preprocess = weight.transforms(antialias=True)
    else:
        raise ValueError(f"Model {model_name} not found")
    return model, preprocess