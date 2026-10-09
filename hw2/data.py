import functools

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
PADDING = 4


def download(root):
    CIFAR10(root, train=True, download=True)
    CIFAR10(root, train=False, download=True)


def load_split(root, train, device):
    """Return images as float tensor (N, 3, 32, 32) in [0, 1] and labels, both on device."""
    dataset = CIFAR10(root, train=train, download=False)
    x = torch.from_numpy(dataset.data).to(device).permute(0, 3, 1, 2).float().div(255).contiguous()
    y = torch.tensor(dataset.targets, device=device)
    return x, y


@functools.lru_cache(maxsize=None)
def _stats(device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return mean, std


def prepare(x):
    """Normalize images in [0, 1] with CIFAR-10 channel statistics and convert to channels_last."""
    mean, std = _stats(x.device)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def augment(x, generator):
    """Random 32x32 crop from the image zero-padded by 4 pixels, then horizontal flip with probability 0.5."""
    batch, _, height, width = x.shape
    padded = F.pad(x, (PADDING, PADDING, PADDING, PADDING))
    top = torch.randint(0, 2 * PADDING + 1, (batch,), device=x.device, generator=generator)
    left = torch.randint(0, 2 * PADDING + 1, (batch,), device=x.device, generator=generator)
    rows = top[:, None, None] + torch.arange(height, device=x.device)[None, :, None]
    cols = left[:, None, None] + torch.arange(width, device=x.device)[None, None, :]
    batch_index = torch.arange(batch, device=x.device)[:, None, None]
    cropped = padded[batch_index, :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(batch, device=x.device, generator=generator) < 0.5
    return torch.where(flip[:, None, None, None], cropped.flip(3), cropped)
