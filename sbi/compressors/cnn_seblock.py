"""seblock ResNet3D compressor for raw cubes (Case 2). Math copied verbatim
from the user's model_mod1.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1, 1).expand_as(x)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, ci, co, k, s, p):
        super().__init__()
        self.depthwise = nn.Conv3d(ci, ci, k, s, p, groups=ci)
        self.pointwise = nn.Conv3d(ci, co, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResNet3DCompressor(nn.Module):
    def __init__(self, t_dim=8, n_params=4, direct=False):
        super().__init__()
        # direct=True: the summary head itself is regressed to the params (the
        # MSE supervises fc_summary, i.e. the EXPORTED t). Requires t_dim==n_params.
        # direct=False (default): separate fc_summary (-> t) and fc_aux (-> params);
        # the MSE supervises fc_aux only, so t is trained indirectly via shared features.
        self.direct = bool(direct)
        if self.direct and t_dim != n_params:
            raise ValueError(
                f"direct regression needs t_dim==n_params, got t_dim={t_dim}, n_params={n_params}"
            )
        self.conv1 = DepthwiseSeparableConv3d(3, 32, 3, 2, 1)
        self.bn1 = nn.GroupNorm(4, 32)
        self.se1 = SEBlock3d(32, 4)
        self.conv2 = DepthwiseSeparableConv3d(32, 64, 3, 2, 1)
        self.bn2 = nn.GroupNorm(8, 64)
        self.se2 = SEBlock3d(64, 8)
        self.conv3 = DepthwiseSeparableConv3d(64, 128, 3, 2, 1)
        self.bn3 = nn.GroupNorm(8, 128)
        self.se3 = SEBlock3d(128)
        self.conv4 = DepthwiseSeparableConv3d(128, 256, 3, 2, 1)
        self.bn4 = nn.GroupNorm(8, 256)
        self.se4 = SEBlock3d(256)
        self.conv5 = nn.Conv3d(256, 512, kernel_size=2, stride=1, padding=0)
        self.bn5 = nn.GroupNorm(8, 512)
        self.fc_summary = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, t_dim),
        )
        # fc_aux only needed in the indirect (two-head) design
        self.fc_aux = (
            None
            if self.direct
            else nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),
                nn.Linear(128, n_params),
            )
        )

    def forward(self, x):
        f = self.features(x)
        t = self.fc_summary(f)
        if self.direct:
            return t, t  # aux output IS the summary: MSE supervises t directly
        return t, self.fc_aux(f)

    def features(self, x):
        x = self.se1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.se2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.se3(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.se4(F.leaky_relu(self.bn4(self.conv4(x))))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        return torch.flatten(x, 1)
