from efficientnet_pytorch import EfficientNet


def get_model(pretrained, device):
    model_name = 'efficientnet-b7'

    if pretrained:
        model = EfficientNet.from_pretrained(model_name, num_classes=5)
    else:
        model = EfficientNet.from_name(model_name, num_classes=5)

    return model.to(device)