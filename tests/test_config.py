"""Config loading + dotted-override behaviour (the -o key=value mechanism)."""

import textwrap

from sbi.config import load_config


def test_dotted_override_and_coercion(tmp_path):
    p = tmp_path / "arm.yaml"
    p.write_text(
        textwrap.dedent("""
        arm_name: demo/arm
        scratch_root: /tmp/scratch
        compressor:
          init_seed: 0
          lr: 0.001
    """)
    )
    cfg = load_config(str(p), overrides=["compressor.init_seed=3", "compressor.lr=2e-4"])
    assert cfg["arm_name"] == "demo/arm"
    # -o values are coerced from strings to real types
    assert cfg["compressor"]["init_seed"] == 3
    assert isinstance(cfg["compressor"]["init_seed"], int)
    assert abs(cfg["compressor"]["lr"] - 2e-4) < 1e-12
