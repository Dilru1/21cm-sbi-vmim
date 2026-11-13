#!/usr/bin/env python3
"""Wire the two missing Phase-A ablation knobs into cnn_seblock_up.py.

    python apply_arch_patch.py            # from the repo root
    python apply_arch_patch.py --revert

Idempotent, keeps a .bak, and compiles the result before writing.

WHAT IT ADDS
  pool_stats : ("mean","std") -> which halves of the readout exist.
      _pool_stats() is already defined in your file but NOTHING CALLS IT:
      forward() still uses _pool_mean_std, combined_features_dim is hard-coded
      to 2*sum(...), and "pool_stats" is not in _ARCH_KEYS. Without this patch
      configs A1/A2 run silently as A0.

      Why it matters now: with the cube mean removed from the input, the `mean`
      half of the readout is no longer a near-direct path to the global signal
      (which carried 93-97% of the cube variance). Which half the network
      actually needs is an open question.

  use_se : bool -> SE block present or replaced by nn.Identity.
      se_reduction CANNOT switch SE off: hidden = max(4, channels//reduction)
      floors at 4, so even reduction=1024 leaves a working squeeze MLP. A real
      ablation needs the module gone. nn.Identity keeps forward() unchanged and
      leaves no se.* keys in the state dict.

DEFAULTS ARE UNCHANGED. pool_stats=("mean","std") and use_se=True reproduce the
historical 15,477,608-parameter graph and state dict exactly, so A0 and every
existing checkpoint still load.
"""
import argparse
import os
import shutil
import sys

MARK = "# [arch-patch]"
TARGET = os.path.join("sbi", "compressors", "cnn_seblock_up.py")


def sub(text, old, new, what):
    if old not in text:
        sys.exit(f"ANCHOR NOT FOUND ({what}).\nApply this hunk by hand:\n"
                 f"---- find ----\n{old}\n---- replace ----\n{new}")
    if text.count(old) != 1:
        sys.exit(f"ANCHOR NOT UNIQUE ({what}): {text.count(old)} matches")
    return text.replace(old, new)


def patch(t):
    # --- 1. ResidualBlock3d takes use_se ---------------------------------
    t = sub(t,
        '    def __init__(self, ci, co, stride=1, reduction=8, padding_mode="zeros"):',
        '    def __init__(self, ci, co, stride=1, reduction=8, padding_mode="zeros",\n'
        '                 use_se=True):',
        "ResidualBlock3d signature")

    t = sub(t,
        "        self.se = SEBlock3d(co, reduction=reduction)\n",
        f"        {MARK} se_reduction cannot switch SE off -- hidden floors at\n"
        "        # max(4, co//reduction), so even reduction=1024 leaves a working\n"
        "        # squeeze MLP. A real ablation needs the module gone. Identity\n"
        "        # keeps forward() unchanged and leaves no se.* state-dict keys.\n"
        "        self.se = SEBlock3d(co, reduction=reduction) if use_se else nn.Identity()\n",
        "SE construction")

    # --- 2. compressor signature -----------------------------------------
    t = sub(t,
        "                 width_mult=1.0, se_reduction=8,\n"
        "                 in_channels=3, circular_padding=False,",
        "                 width_mult=1.0, se_reduction=8, use_se=True,\n"
        '                 pool_stats=("mean", "std"),\n'
        "                 in_channels=3, circular_padding=False,",
        "ResNet3DCompressor signature")

    # --- 3. validate + store pool_stats -----------------------------------
    t = sub(t,
        "        self.readout_stages = readout_stages\n"
        "        self.channels, self.strides = channels, strides",
        "        self.readout_stages = readout_stages\n"
        "        self.channels, self.strides = channels, strides\n"
        "\n"
        f"        {MARK} which readout statistics exist. Default == historical 2C.\n"
        "        self.pool_stats = tuple(pool_stats)\n"
        "        bad = set(self.pool_stats) - {\"mean\", \"std\"}\n"
        "        if bad or not self.pool_stats:\n"
        "            raise ValueError(f\"pool_stats must be a non-empty subset of \"\n"
        "                             f\"('mean','std'), got {pool_stats}\")",
        "pool_stats validation")

    # --- 4. readout width follows pool_stats ------------------------------
    t = sub(t,
        "        combined_features_dim = 2 * sum(channels[s] for s in readout_stages)",
        f"        {MARK} was hard-coded 2*; now follows pool_stats\n"
        "        combined_features_dim = len(self.pool_stats) * sum(\n"
        "            channels[s] for s in readout_stages)",
        "combined_features_dim")

    # --- 5. pass use_se to each block --------------------------------------
    t = sub(t,
        "            blk = ResidualBlock3d(prev, channels[i + 1], stride=strides[i], reduction=se_reduction,\n"
        "                                  padding_mode=pad_mode)",
        "            blk = ResidualBlock3d(prev, channels[i + 1], stride=strides[i],\n"
        "                                  reduction=se_reduction, padding_mode=pad_mode,\n"
        "                                  use_se=use_se)",
        "block construction")

    # --- 6. forward uses _pool_stats ---------------------------------------
    t = sub(t,
        "        f = torch.cat([_pool_mean_std(outs[s - 1]) for s in self.readout_stages], dim=1)",
        f"        {MARK} _pool_stats was defined but never called\n"
        "        f = torch.cat([_pool_stats(outs[s - 1], self.pool_stats)\n"
        "                       for s in self.readout_stages], dim=1)",
        "forward readout")

    # --- 7. expose both knobs to the yaml ----------------------------------
    t = sub(t,
        '_ARCH_KEYS = ("channels", "strides", "readout_from", "width_mult", "se_reduction",\n'
        '              "in_channels", "circular_padding", "spectral_readout", "cov_readout", "cov_dim")',
        f'{MARK} use_se and pool_stats added\n'
        '_ARCH_KEYS = ("channels", "strides", "readout_from", "width_mult", "se_reduction",\n'
        '              "use_se", "pool_stats",\n'
        '              "in_channels", "circular_padding", "spectral_readout", "cov_readout", "cov_dim")',
        "_ARCH_KEYS")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    os.chdir(a.root)

    if not os.path.exists(TARGET):
        sys.exit(f"missing {TARGET} -- run from the repo root")

    if a.revert:
        if os.path.exists(TARGET + ".bak"):
            shutil.copy(TARGET + ".bak", TARGET)
            print(f"reverted {TARGET}")
        else:
            print("no backup")
        return

    src = open(TARGET).read()
    if MARK in src:
        print(f"{TARGET}: already patched")
    else:
        if not os.path.exists(TARGET + ".bak"):
            shutil.copy(TARGET, TARGET + ".bak")
        out = patch(src)
        compile(out, TARGET, "exec")
        open(TARGET, "w").write(out)
        print(f"patched {TARGET}   (backup: {TARGET}.bak)")

    # --- the check that actually matters ----------------------------------
    s1 = "stage1_compress.py"
    if os.path.exists(s1):
        txt = open(s1).read()
        if "build_cnn_from_cfg" not in txt:
            print("\n*** WARNING ***")
            print(f"{s1} does not call build_cnn_from_cfg, so the compressor.arch")
            print("block in every ablation config is IGNORED and all eight runs")
            print("would train the identical default network. Replace")
            print('    compressor = comp(t_dim=cfg["t_dim"], n_params=cfg["n_params"],')
            print('                      direct=bool(c.get("direct_regression", False)))')
            print("with")
            print("    from sbi.compressors.cnn_seblock_up import build_cnn_from_cfg")
            print('    compressor = build_cnn_from_cfg(c, cfg["t_dim"], cfg["n_params"],')
            print('                                    direct=bool(c.get("direct_regression", False)))')
        else:
            print(f"{s1}: build_cnn_from_cfg present -- arch block will be honoured")

    print("\nverify:\n  python tools/count_params.py configs/abl_/*.yaml")


if __name__ == "__main__":
    main()
