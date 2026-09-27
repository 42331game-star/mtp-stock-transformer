"""CLI <-> config wiring.

``--layers`` used to set a throwaway ``cfg.layers`` attribute while the model
read ``cfg.num_layers``, so the flag in the README silently did nothing. These
tests make that class of bug loud.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mtp import MTPConfig  # noqa: E402
from mtp.cli.train import apply_args, build_parser  # noqa: E402


def test_every_train_flag_maps_to_a_config_field():
    known = {f.name for f in fields(MTPConfig)}
    dests = [a.dest for a in build_parser()._actions if a.dest != "help"]
    assert dests, "parser has no flags"
    unmapped = [d for d in dests if d not in known]
    assert unmapped == [], f"flags that would be silently ignored: {unmapped}"


def test_layers_flag_sets_num_layers():
    ns, _ = build_parser().parse_known_args(
        ["--layers", "8", "--d-model", "128", "--seed", "7"]
    )
    cfg = apply_args(MTPConfig(), ns)
    assert cfg.num_layers == 8, "--layers must reach num_layers"
    assert cfg.d_model == 128
    assert cfg.seed == 7


def test_unknown_field_is_rejected():
    cfg = MTPConfig()
    try:
        apply_args(cfg, argparse.Namespace(not_a_field=123))
    except SystemExit as exc:
        assert "not_a_field" in str(exc)
    else:
        raise AssertionError("unknown flag must not be applied silently")
