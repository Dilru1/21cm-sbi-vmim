import torch
import torch.nn as nn
import torch.nn.functional as F

# CHANGES vs previous version
# ---------------------------
# 1. Multi-scale readout is mean+std per channel (std over spatial dims), not
#    mean alone. Global average pooling discards fluctuation amplitude; rHS
#    lives in temperature-fluctuation texture, so the std channels give the
#    weak parameter a direct path to the heads. combined dim: 960 -> 1920.
# 2. SE reduction 8 (was 16): reduction 16 on 32 channels left a 2-unit
#    bottleneck in the squeeze MLP.


class SEBlock3d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1, 1).expand_as(x)


class ResidualBlock3d(nn.Module):
    """3D residual block with optional downsampling and SE block."""
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(ci, co, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(min(8, co // 4), co)
        self.conv2 = nn.Conv3d(co, co, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(min(8, co // 4), co)
        self.se = SEBlock3d(co, reduction=8)

        self.shortcut = nn.Sequential()
        if stride != 1 or ci != co:
            self.shortcut = nn.Sequential(
                nn.Conv3d(ci, co, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(8, co // 4), co)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.leaky_relu(out, 0.1)


def _pool_mean_std(x):
    """(B, C, D, H, W) -> (B, 2C): per-channel spatial mean and std."""
    flat = x.flatten(2)                       # (B, C, DHW)
    return torch.cat([flat.mean(dim=2), flat.std(dim=2)], dim=1)


class ResNet3DCompressor(nn.Module):
    def __init__(self, t_dim=8, n_params=4, direct=False):
        super().__init__()
        self.direct = bool(direct)
        if self.direct and t_dim != n_params:
            raise ValueError(f"direct regression needs t_dim==n_params, got t_dim={t_dim}, n_params={n_params}")

        # full dense stem (not depthwise-separable), stride 1
        self.stem = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.1)
        )

        # stride 1 early to keep small scales
        self.layer1 = ResidualBlock3d(32,  32, stride=1)   # ONE block, narrow, full res — texture only
        self.layer2 = ResidualBlock3d(32,  64, stride=1)   # downsample immediately after
        self.layer3 = ResidualBlock3d(64, 128, stride=2)
        self.layer4 = ResidualBlock3d(128, 256, stride=2)
        self.layer5 = ResidualBlock3d(256, 512, stride=2)

        # mean+std readout from stages 2..5: 2*(64+128+256+512) = 1920
        combined_features_dim = 2 * (64 + 128 + 256 + 512)

        self.fc_summary = nn.Sequential(
            nn.Linear(combined_features_dim, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), nn.Dropout(0.05),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.1), nn.Dropout(0.05),
            nn.Linear(128, t_dim)
        )

        self.fc_aux = None if self.direct else nn.Sequential(
            nn.Linear(combined_features_dim, 256), nn.LayerNorm(256), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.LeakyReLU(0.1), nn.Dropout(0.2),
            nn.Linear(128, n_params)
        )

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        f = torch.cat([_pool_mean_std(x2), _pool_mean_std(x3),
                       _pool_mean_std(x4), _pool_mean_std(x5)], dim=1)

        t = self.fc_summary(f)
        if self.direct:
            return t, t
        return t, self.fc_aux(f)