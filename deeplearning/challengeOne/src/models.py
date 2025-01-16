import timm
import torch.nn as nn
from torchvision import models


def dense_net(output=1):
    model = models.densenet161(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, output)
    return model


def mobilenet_v2(output=1):
    model = models.mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, output)
    return model


def efficientnetv2_l(output=1):
    model = timm.create_model("tf_efficientnetv2_l_in21k", pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, output)
    return model


def efficientnetv2_xl(output=1):
    model = timm.create_model("tf_efficientnetv2_xl_in21k", pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, output)
    return model


def resnext50(output=1):
    resnext = models.resnext50_32x4d(pretrained=True)
    num_ftrs = resnext.fc.in_features
    resnext.fc = nn.Linear(num_ftrs, output)
    return resnext


def resnext101(output=1):
    resnext = models.resnext101_64x4d(pretrained=True)
    num_ftrs = resnext.fc.in_features
    resnext.fc = nn.Linear(num_ftrs, output)
    return resnext


def convnext_base(output=1):
    model = timm.create_model("convnext_base", pretrained=True)
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, output)
    return model


def convnext_large(output=1):
    model = timm.create_model("convnext_large", pretrained=True)
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, output)
    return model


def convnextv2_base(output=1):
    model = timm.create_model("convnextv2_base", pretrained=True)
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, output)
    return model


def convnextv2_large(output=1):
    model = timm.create_model("convnextv2_large", pretrained=True)
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, output)
    return model


def convit_base(output=1):
    model = timm.create_model("convit_base", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def convit_small(output=1):
    model = timm.create_model("convit_small", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def deit_base_distilled_patch16_224(output=1):
    model = timm.create_model("deit_base_distilled_patch16_224", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def deit_base_patch16_224(output=1):
    model = timm.create_model("deit_base_patch16_224", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def vitbase16(output=1):
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def vitbase32(output=1):
    model = timm.create_model("vit_base_patch32_224", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def vitlarge16(output=1):
    model = timm.create_model("vit_large_patch16_224", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def vitlarge32(output=1):
    model = timm.create_model("vit_large_patch32_224", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def swin_base_patch4_window7_224(output=1):
    model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def swin_large_patch4_window7_224(output=1):
    model = timm.create_model("swin_large_patch4_window7_224", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def swinv2_base_window16_256(output=1):
    model = timm.create_model("swinv2_base_window16_256", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def swinv2_large_window12_192(output=1):
    model = timm.create_model("swinv2_large_window12_192", pretrained=True)
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, output)
    return model


def regnetx_002(output=1):
    model = timm.create_model("regnetx_002", pretrained=True)
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, output)
    return model


def regnetx_004(output=1):
    model = timm.create_model("regnetx_004", pretrained=True)
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, output)
    return model


def regnety_002(output=1):
    model = timm.create_model("regnety_002", pretrained=True)
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, output)
    return model


def regnety_004(output=1):
    model = timm.create_model("regnety_004", pretrained=True)
    num_ftrs = model.head.fc.in_features
    model.head.fc = nn.Linear(num_ftrs, output)
    return model
