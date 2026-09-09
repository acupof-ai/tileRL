"""--deterministic must wire decode_graph=False: the captured decode graph is
the cross-process nondeterminism source
(errors/2026-09-09-a-flag-reverted-by-a-stale-branch.md)."""
import inspect

from tilerl import cli


def test_deterministic_flag_wires_decode_graph():
    parser = cli._build_parser()
    args_on = parser.parse_args(["train", "--deterministic"])
    args_off = parser.parse_args(["train"])
    assert args_on.deterministic is True
    assert args_off.deterministic is False
    # The wiring in _train_adapters: decode_graph=not args.deterministic
    assert "not args.deterministic" in inspect.getsource(cli._train_adapters)
