import torch

from .wideResNet import WideResNet
from .resnet_CIFAR import ResNetCIFAR
from .resnet import ResNet
from .vit import SimpleViT
from .vgg import VGG
from utility.args import Args

# Define flags here that are used by multiple models to avoid double definitions.
Args.add_argument("--model", type=str, help="model name (WRN = WideResNet")
Args.add_argument("--dropout", type=float, help="Dropout rate.")
Args.add_argument("--BN", type=bool, help="use batch norm ?")
Args.add_argument("--depth", type=int, help="Number of layers.")



modelDict = {
    "WRN" : WideResNet,
    "resnet": ResNet,
    "resnetCIFAR": ResNetCIFAR,
    "ViT": SimpleViT,
    "VGG": VGG,
}


def getModel() -> torch.nn.Module:
    if Args.model in modelDict:
        return modelDict[Args.model]
    else:
        raise RuntimeError(f"Model '{Args.model}' not found. Available models: {', '.join(modelDict.keys())}")
