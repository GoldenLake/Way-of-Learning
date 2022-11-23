import torch
import torch.nn as nn


def ConvMixer(h, depth, kernel_size=9, patch_size=7, n_classes=1000):
    Seq, ActBn = nn.Sequential, lambda x: Seq(x, nn.GELU(), nn.BatchNorm2d(h))
    # print(ActBn(nn.Conv2d(3, h, patch_size, stride=patch_size)))
    Residual = type('Residual', (Seq,), {'forward': lambda self, x: self[0](x) + x})
    # print(Residual(ActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding="same"))))
    return Seq(ActBn(nn.Conv2d(3, h, patch_size, stride=patch_size)),
               *[Seq(Residual(ActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding="same"))),
                     ActBn(nn.Conv2d(h, h, 1))) for i in range(depth)],
               nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(h, n_classes))


if __name__ == "__main__":
    net = ConvMixer(224, 3)
    # print(net)
    x = torch.randn(1, 3, 224, 224)