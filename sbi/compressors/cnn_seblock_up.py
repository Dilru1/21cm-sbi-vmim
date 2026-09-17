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
# 3. NEW -- architecture is now width/depth configurable via constructor kwargs
#    (channels, strides, readout_from, width_mult, se_reduction) for capacity
#    ablations. The DEFAULTS reproduce the previous fixed architecture EXACTLY:
#    same module names (stem, layer1..layer5, fc_summary, fc_aux), same shapes,
#    same 15,477,608 params -- so old checkpoints load without changes. Only
#    when you pass reducing knobs (e.g. channels=[32,32,64,128,256]) does the
#    graph shrink; a checkpoint is then only loadable by a model built with the
#    SAME knobs (state-dict keys/shapes track the schedule).


# Fixed schedule that reproduces the historical architecture when no knobs are
# passed. channels[0] is the stem output; block i maps channels[i]->channels[i+1].
DEFAULT_CHANNELS = [32, 32, 64, 128, 256, 512]   # stem32; blocks 32,64,128,256,512
DEFAULT_STRIDES  = [1, 1, 2, 2, 2]               # two stride-1 (full-res texture) then stride-2


def _gn_groups(channels, target):
    """Largest group count <= target that divides `channels` (>=1).
    For the default widths this returns the same counts the old code hardcoded
    (stem: 4 groups on 32ch; residual blocks: min(8, co//4))."""
    g = min(int(target), int(channels))
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)


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
    def __init__(self, ci, co, stride=1, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv3d(ci, co, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(min(8, co // 4), co)
        self.conv2 = nn.Conv3d(co, co, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(min(8, co // 4), co)
        self.se = SEBlock3d(co, reduction=reduction)

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
    """3D residual multi-scale CNN compressor.

    Architecture knobs (all optional; defaults == historical fixed net):

      channels     : list, channel schedule. channels[0] is the stem output;
                     residual block i maps channels[i] -> channels[i+1]. There
                     are len(channels)-1 blocks. Default DEFAULT_CHANNELS.
                     If None, DEFAULT_CHANNELS scaled by width_mult is used.
      strides      : list len == n_blocks. Default DEFAULT_STRIDES when the
                     block count matches, else [1] + [2]*(n_blocks-1) (keep the
                     first block at full resolution for small-scale texture).
      readout_from : 1-based stage indices feeding the mean+std readout (stage i
                     = output of block i). Default None => stages 2..N (skip the
                     first, full-resolution texture block -- matches the old net,
                     which read x2..x5 and dropped x1).
      width_mult   : convenience scalar applied to DEFAULT_CHANNELS when
                     `channels` is None (e.g. 0.5 -> ~4M params). Ignored if
                     `channels` is given explicitly.
      se_reduction : squeeze-excitation reduction ratio (default 8).

    Examples
    --------
      # identical to the old model (15.48M params, old checkpoints load):
      ResNet3DCompressor(t_dim=4, n_params=4)

      # drop the 512 stage  (~4.1M params):
      ResNet3DCompressor(t_dim=4, n_params=4,
                         channels=[32,32,64,128,256], strides=[1,1,2,2],
                         readout_from=[2,3,4])

      # half width          (~4.2M params):
      ResNet3DCompressor(t_dim=4, n_params=4, width_mult=0.5)
    """
    def __init__(self, t_dim=8, n_params=4, direct=False,
                 channels=None, strides=None, readout_from=None,
                 width_mult=1.0, se_reduction=8):
        super().__init__()
        self.direct = bool(direct)
        if self.direct and t_dim != n_params:
            raise ValueError(f"direct regression needs t_dim==n_params, got t_dim={t_dim}, n_params={n_params}")

        # ---- resolve the channel/stride schedule ----
        if channels is None:
            channels = [max(1, int(round(c * float(width_mult)))) for c in DEFAULT_CHANNELS]
        else:
            channels = [int(c) for c in channels]
        n_blocks = len(channels) - 1
        if n_blocks < 1:
            raise ValueError(f"channels must have >=2 entries (stem + >=1 block), got {channels}")

        if strides is None:
            strides = list(DEFAULT_STRIDES) if n_blocks == len(DEFAULT_STRIDES) \
                      else [1] + [2] * (n_blocks - 1)
        else:
            strides = [int(s) for s in strides]
        if len(strides) != n_blocks:
            raise ValueError(f"strides must have length n_blocks={n_blocks}, got {len(strides)}")

        # ---- resolve which stages feed the readout ----
        if readout_from is None:
            readout_stages = list(range(2, n_blocks + 1))   # skip stage 1 (matches old net)
        else:
            readout_stages = [int(s) for s in readout_from]
        for s in readout_stages:
            if not (1 <= s <= n_blocks):
                raise ValueError(f"readout_from stage {s} out of range 1..{n_blocks}")
        if not readout_stages:
            raise ValueError("readout_from resolved to an empty set of stages")
        self.readout_stages = readout_stages
        self.channels, self.strides = channels, strides

        # ---- stem ----
        c0 = channels[0]
        self.stem = nn.Sequential(
            nn.Conv3d(3, c0, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(c0, 4), c0),
            nn.LeakyReLU(0.1)
        )

        # ---- residual stages, registered as layer1..layerN so that the DEFAULT
        #      schedule reproduces the old state-dict keys exactly ----
        self._stage_blocks = []
        prev = c0
        for i in range(n_blocks):
            blk = ResidualBlock3d(prev, channels[i + 1], stride=strides[i], reduction=se_reduction)
            setattr(self, f"layer{i + 1}", blk)     # -> state_dict keys layer1.*, layer2.*, ...
            self._stage_blocks.append(blk)
            prev = channels[i + 1]

        # mean+std readout from the selected stages
        combined_features_dim = 2 * sum(channels[s] for s in readout_stages)

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
        outs = []
        for blk in self._stage_blocks:
            x = blk(x)
            outs.append(x)                      # outs[i] == stage (i+1) output

        f = torch.cat([_pool_mean_std(outs[s - 1]) for s in self.readout_stages], dim=1)

        t = self.fc_summary(f)
        if self.direct:
            return t, t
        return t, self.fc_aux(f)