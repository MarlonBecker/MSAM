import torchvision.models as models
import torch.nn as nn

from utility.args import Args

class ResNet(models.ResNet):
    def __init__(self, num_classes = 10):

        blocks_layers = {
            18: (models.resnet.BasicBlock, [2, 2, 2, 2]),
            34: (models.resnet.BasicBlock, [3, 4, 6, 3]),
            50: (models.resnet.Bottleneck, [3, 4, 6, 3]),
            101: (models.resnet.Bottleneck, [3, 4, 23, 3]),
            152: (models.resnet.Bottleneck, [3, 8, 36, 3]),
        }
        
        if not Args.depth in blocks_layers:
            raise RuntimeError(f"Model depth {Args.depth} not supported for Resnet.")
    
        block, layers = blocks_layers[Args.depth]
        super().__init__(block=block, layers=layers, num_classes=num_classes)
    
    
    def setBatchNormTracking(self, track_running_stats: bool):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = track_running_stats
