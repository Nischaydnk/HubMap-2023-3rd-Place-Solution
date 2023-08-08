import timm
import numpy as np
import torch.nn as nn
from ..builder import BACKBONES

# from params import MEAN, STD


@BACKBONES.register_module()
class EfficientNetV2(nn.Module):
    def __init__(self, name, pretrained=True):
        """
        Constructor.

        Args:
            name (name): Model name as specified in timm.
            blocks_idx (list of ints): Blocks to output features at.
            pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        """
        super().__init__()

        self.effnet = getattr(timm.models, name)(
            pretrained=pretrained,
            drop_path_rate=0.2,
        )

        self.block_idx = [1,3,5,6]
        self.nb_fts = [self.effnet.blocks[b][-1].conv_pwl.out_channels for b in self.block_idx]
        self.nb_ft = self.nb_fts[-1]


        self.name = name

    def forward(self, x):  # should return a tuple
        """
        Extract features for an EfficientNet model.
        Args:
            x (torch tensor [BS x 3 x H x W]): Input image.
        Returns:
            list of torch tensors: features.
        """
        x = self.effnet.conv_stem(x)
        x = self.effnet.bn1(x)
        # x = self.effnet.act1(x)

        features = []
        for i, b in enumerate(self.effnet.blocks):
            x = b(x)
            if i in self.block_idx:
                features.append(x)
            print(i, x.size(), i in self.block_idx)

        return features