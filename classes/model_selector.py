from torchvision.models import resnet50, Resnet50_Weights, vit_b_16, ViT_B_16_Weights

def get_model(model_name):
    if model_name == "resnet50":
        weight = Resnet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weight).cuda()
        model.eval()
        preprocess = weight.transforms()
    elif model_name == "vit_b_16":
        weight = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weight).cuda()
        model.eval()
        preprocess = weight.transforms()
    else:
        raise ValueError(f"Model {model_name} not found")
    return model, preprocess