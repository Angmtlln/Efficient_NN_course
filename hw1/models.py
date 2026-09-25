import torch
from torch import nn


class SmallCNN(nn.Sequential):
    """Conv -> ReLU after every convolution, no BatchNorm, no residuals.

    Output resolution: S/4 after the stem, S/8 after conv3, S/16 after conv5.
    """

    def __init__(self, num_classes: int = 100):
        super().__init__(
            
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )


def build_model() -> nn.Module:
    return SmallCNN(num_classes=100)


if __name__ == "__main__":
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"parameters: {n_params:,}")
    for S in (32, 224):
        y = model.eval()(torch.randn(2, 3, S, S))
        print(f"S={S}: output {tuple(y.shape)}")
