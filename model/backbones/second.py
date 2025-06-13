import torch
from torch import nn
from torch.nn import BatchNorm2d, Conv2d
import torch.nn.functional as F

class DropBlock2D(nn.Module):
    def __init__(self, block_size=3, drop_prob=0.1, start_epoch=3, max_epoch=12, progressive=True):
        super().__init__()
        self.block_size = block_size
        self.initial_drop_prob = drop_prob
        self.drop_prob = drop_prob
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.progressive = progressive
        self.current_epoch = 0

    def forward(self, x):
        if not self.training:
            return x

        if self.current_epoch < self.start_epoch:
            return x

        if self.progressive:
            # Linearly increase drop probability over epochs
            progress = min(1.0, (self.current_epoch - self.start_epoch) / max(1, self.max_epoch - self.start_epoch))
            drop_prob = self.initial_drop_prob * progress
        else:
            drop_prob = self.initial_drop_prob

        gamma = drop_prob / (self.block_size ** 2)
        mask = (torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) < gamma).float()
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2, ceil_mode=True)
        mask = 1 - mask
        # Crop to match original shape (in case of size mismatch)
        mask = mask[:, :, :x.shape[2], :x.shape[3]]
        return x * mask * (mask.numel() / (mask.sum() + 1e-6))

    def set_epoch(self, epoch):
        self.current_epoch = epoch


class SECOND(nn.Module):
    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 dropblock_prob=0.1,
                 dropblock_size=3,
                 dropblock_start_epoch=3,
                 dropblock_max_epoch=12,
                 dropblock_progressive=True):
        super().__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        self.dropblocks = []  # Keep track of DropBlock modules to update epoch later

        in_filters = [in_channels] + out_channels[:-1]
        blocks = []

        for i, num_layers in enumerate(layer_nums):
            layers = []
            use_dropblock = i >= 1  # Apply DropBlock only in stage 2 and 3

            # First conv layer with stride
            layers += [
                Conv2d(in_filters[i], out_channels[i], kernel_size=3, stride=layer_strides[i], padding=1, bias=False),
                BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
            ]
            if use_dropblock:
                db = DropBlock2D(
                    block_size=dropblock_size,
                    drop_prob=dropblock_prob,
                    start_epoch=dropblock_start_epoch,
                    max_epoch=dropblock_max_epoch,
                    progressive=dropblock_progressive
                )
                layers.append(db)
                self.dropblocks.append(db)

            # Additional conv layers
            for _ in range(num_layers):
                layers += [
                    Conv2d(out_channels[i], out_channels[i], kernel_size=3, padding=1, bias=False),
                    BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ]
                if use_dropblock:
                    db = DropBlock2D(
                        block_size=dropblock_size,
                        drop_prob=dropblock_prob,
                        start_epoch=dropblock_start_epoch,
                        max_epoch=dropblock_max_epoch,
                        progressive=dropblock_progressive
                    )
                    layers.append(db)
                    self.dropblocks.append(db)

            blocks.append(nn.Sequential(*layers))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return tuple(outs)

    def set_epoch(self, epoch):
        for db in self.dropblocks:
            db.set_epoch(epoch)
