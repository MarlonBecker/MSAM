from torchvision import models
import torch.nn as nn

from utility.args import Args

cfgs = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(models.VGG):
    def __init__(self, num_classes = 10):
        if Args.depth in cfgs:
            cfg = cfgs[Args.depth]
        else:
            raise RuntimeError(f"Depth {Args.depth} is not supported for VGG. Select one of {', '.join(map(str,cfgs.keys()))}.")

        super(VGG, self).__init__(models.vgg.make_layers(cfg, batch_norm=Args.BN), num_classes = num_classes)

    def setBatchNormTracking(self, track_running_stats: bool):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = track_running_stats
